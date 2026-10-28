"""Held native prefills retain rewind anchors without full KV/MTP snapshots."""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mtplx import generation as gen
from mtplx.session_bank import SessionBank
from test_checkpoint_pre_image import _runtime


@pytest.mark.parametrize("policy", ["cycle", "committed"])
@pytest.mark.parametrize("stable_branch", [False, True])
def test_held_prefill_owns_native_state_and_proves_actual_extension(monkeypatch, policy, stable_branch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "64")
    monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", "1")
    rt = _runtime()
    rt.model.make_mtp_cache = lambda: [KVCache()]

    def update(hidden_states, next_token_ids, *, mtp_cache, **kwargs):
        rows = next_token_ids.astype(mx.float32)[:, None, :, None]
        mtp_cache[0].update_and_fetch(rows, rows)
        return hidden_states

    rt.model.mtp_update_cache = update
    snapshots = []
    original = gen.snapshot_cache

    def snapshot(cache):
        snapshots.append(cache)
        return original(cache)

    monkeypatch.setattr(gen, "snapshot_cache", snapshot)
    bank = SessionBank()
    bank.hold_session_persistence("tools")
    ids = [index % 14 for index in range(193)]
    state = gen.restore_or_prefill_prompt_state(
        rt, ids, session_bank=bank, session_id="tools", mtp_history_policy=policy,
        store_prefix_snapshot=True, explicit_cache_boundary=37, stable_prefix_len=149,
    )
    entry = bank.longest_prefix(ids)
    assert state.prefill_store_snapshot["stored"]
    assert entry.live_ref_only and entry.native_exclusive
    assert entry.cache_ref is state.trunk_cache
    assert entry.mtp_history_cache_ref is state.committed_mtp_cache
    assert entry.mtp_history_snapshot is None
    assert entry.native_stable_prefix_len == 149
    assert not snapshots
    assert len(ids) in [record[0] for record in entry.gdn_boundaries]
    assert all(record[1].states[0] is None for record in entry.gdn_boundaries)

    extended = (ids[:149] if stable_branch else ids) + [3, 4, 5]
    prepared = bank.prepare_native_continuation(session_id="tools", token_ids=extended)
    if stable_branch:
        assert prepared is entry
        assert entry.prefix_len == 149 and entry.extension_only
        assert entry.logits is None
    else:
        assert entry.prefix_len == len(ids) and not entry.extension_only
    warm = gen.restore_or_prefill_prompt_state(
        rt, extended, session_bank=bank, session_id="tools", mtp_history_policy=policy,
        restore_mode="reference", store_prefix_snapshot=False,
    )
    assert warm.cached_tokens == (149 if stable_branch else len(ids))
    assert warm.restore_served["native_ownership_proven"] is True
    assert warm.trunk_cache is state.trunk_cache
    assert warm.committed_mtp_cache is state.committed_mtp_cache
    expected = rt.forward_ar(mx.array([extended]), cache=rt.make_cache(), logits_keep=1)
    assert mx.array_equal(warm.logits, expected[:, -1, :]).item()
    assert warm.explicit_cache_boundary == 37
