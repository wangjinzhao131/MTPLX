"""Decoded SSD donors must not remain alive through suffix prefill."""
import weakref

import mlx.core as mx
import pytest

import mtplx.generation as generation
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.session_bank import SessionBank
from test_checkpoint_anchors import _boundary, _put, _runtime


@pytest.mark.parametrize("fail_suffix", [False, True])
def test_cold_donor_released_before_suffix(tmp_path, monkeypatch, fail_suffix):
    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    bank = SessionBank(cold_tier=tier)
    entry = _put(bank, 1600, [_boundary(512), _boundary(1496)])
    assert tier.put_entry(entry, capabilities=["ar_insert"])
    assert tier.flush(timeout_s=5)
    bank.clear()
    del entry
    donors = []
    candidates = bank.near_prefix_candidates

    def observe(*args, **kwargs):
        rows = candidates(*args, **kwargs)
        donors.extend(weakref.ref(entry) for entry, _ in rows)
        return rows

    monkeypatch.setattr(bank, "near_prefix_candidates", observe)

    def suffix(_rt, restored, tokens, **kwargs):
        assert donors and all(ref() is None for ref in donors)
        assert restored.cache[0].offset == 1496
        assert float(restored.cache[1].state[0][0, 0]) == 1496.0
        restored.cache[1].state[0][0, 0] = -1
        mx.eval(restored.cache[1].state)
        if fail_suffix:
            raise RuntimeError("suffix interrupted")
        return mx.zeros((1, 4)), None, 0.0, 0.0

    monkeypatch.setattr(generation, "_prefill_restored_prompt_suffix", suffix)
    prompt = list(range(1498)) + [9999] * 64
    try:
        for _ in range(2):  # A failed or successful borrower must leave SSD reusable.
            def run():
                return generation._restore_near_prefix_prompt_state(
                    _runtime(), prompt, base_hidden_variant=None, mtp_hidden_variant=None,
                    mtp_history_policy="cycle", session_bank=bank, template_hash=None,
                    draft_head_identity=None, policy_fingerprint=None,
                    allow_block_prefix=True, chunk_started_s=0.0,
                    cache_factory=_runtime().make_cache,
                )
            if fail_suffix:
                with pytest.raises(RuntimeError, match="suffix interrupted"):
                    run()
            else:
                state = run()
                assert state.cached_tokens == 1496
                assert state.ssd_cache_hit
    finally:
        tier.close()
