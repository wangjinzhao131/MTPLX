"""Explicit cache boundaries survive warm branches without shortening exact hits."""

import mlx.core as mx
import pytest

from mtplx import generation
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.checkpoint_anchors import AnchorPlan, record_nbytes, retain_checkpoints
from mtplx.session_bank import SessionBank
from test_checkpoint_anchors import _boundary
from test_checkpoint_pre_image import _runtime


@pytest.fixture(autouse=True)
def checkpoint_prefill(monkeypatch):
    for name, value in {
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_PREFILL_CHUNK_SIZE": "64",
        "MTPLX_SESSION_PREFIX_BLOCK_SIZE": "64",
        "MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS": "64",
        "MTPLX_GDN_BOUNDARY_CAPTURE": "1",
    }.items():
        monkeypatch.setenv(name, value)


def _state(rt, bank, ids, marker=None):
    state = generation.restore_or_prefill_prompt_state(
        rt, ids, session_bank=bank, mtp_history_policy="cycle",
        store_prefix_snapshot=False, explicit_cache_boundary=marker,
    )
    expected = rt.forward_ar(mx.array([ids]), cache=rt.make_cache(), logits_keep=1)
    assert mx.array_equal(state.logits, expected[:, -1, :]).item()
    assert state.trunk_cache[0].offset == len(ids)
    return state


def _publish(rt, bank, state):
    return bank.put(
        runtime=rt, token_ids=state.token_prefix, cache=state.trunk_cache,
        logits=state.logits, hidden=state.hidden, mtp_history_policy="cycle",
        hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
        gdn_boundaries=state.gdn_boundaries,
        explicit_cache_boundary=state.explicit_cache_boundary,
    )


def test_explicit_boundary_has_first_claim_on_existing_checkpoint_budget():
    records = [_boundary(at) for at in (64, 149, 384, 512)]
    kept = retain_checkpoints(records, AnchorPlan(
        explicit_cache_boundary=149, system_prefix=64, prompt_end=512,
        restore_point=384, budget_bytes=record_nbytes(records[0]),
    ))
    assert [record[0] for record in kept] == [149]


@pytest.mark.parametrize("sustained", ["0", "1"])
def test_explicit_committed_prefill_captures_with_existing_streaming_loop(monkeypatch, sustained):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", sustained)
    rt = _runtime()
    ids = [index % 14 for index in range(93)]
    state = generation.restore_or_prefill_prompt_state(
        rt, ids, session_bank=SessionBank(), mtp_history_policy="committed",
        store_prefix_snapshot=False, explicit_cache_boundary=37,
    )
    assert state.explicit_cache_boundary == 37
    assert 37 in [record[0] for record in state.gdn_boundaries]
    expected = rt.forward_ar(mx.array([ids]), cache=rt.make_cache(), logits_keep=1)
    assert mx.array_equal(state.logits, expected[:, -1, :]).item()
    assert state.trunk_cache[0].offset == len(ids)


@pytest.mark.parametrize("marker", [149, 512])
def test_explicit_cold_capture_and_ssd_restart(tmp_path, marker):
    rt = _runtime()
    ids = [index % 14 for index in range(513)]
    bank = SessionBank()
    state = _state(rt, bank, ids, marker)
    assert state.explicit_cache_boundary == marker
    assert marker in [record[0] for record in state.gdn_boundaries]
    entry = _publish(rt, bank, state)
    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    try:
        assert tier.put_entry(entry, capabilities=["ar_insert"])
        assert tier.flush(timeout_s=5)
    finally:
        tier.close()
    bank.clear()
    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    try:
        restarted = SessionBank(cold_tier=tier)
        exact = _state(rt, restarted, ids, marker)
        assert exact.cached_tokens == len(ids)
        assert exact.explicit_cache_boundary == marker
        assert marker in [record[0] for record in exact.gdn_boundaries]
        _publish(rt, restarted, exact)
        extended = _state(rt, restarted, ids + [14, 14])
        assert extended.cached_tokens == len(ids)
        assert extended.explicit_cache_boundary == marker
        _publish(rt, restarted, extended)
        branch = _state(rt, restarted, ids[:marker] + [15] * 65)
        assert branch.cached_tokens == marker
        assert branch.explicit_cache_boundary == marker
    finally:
        tier.close()


def test_new_marker_behind_warm_frontier_replays_once_then_keeps_full_hits():
    rt = _runtime()
    ids = [index % 14 for index in range(513)]
    bank = SessionBank()
    old = _state(rt, bank, ids)
    assert 149 not in [record[0] for record in old.gdn_boundaries]
    _publish(rt, bank, old)
    learned = _state(rt, bank, ids, 149)
    assert 0 < learned.cached_tokens < 149
    assert learned.explicit_cache_boundary == 149
    assert 149 in [record[0] for record in learned.gdn_boundaries]
    _publish(rt, bank, learned)
    repeated = _state(rt, bank, ids, 149)
    assert repeated.cached_tokens == len(ids)
    continued = _state(rt, bank, ids + [14, 14], 149)
    assert continued.cached_tokens == len(ids)
    assert continued.explicit_cache_boundary == 149
    _publish(rt, bank, continued)
    changed_earlier = _state(rt, bank, ids[:100] + [15] * 200)
    assert changed_earlier.cached_tokens <= 100
    assert changed_earlier.explicit_cache_boundary is None


def test_new_marker_behind_ssd_frontier_replays_then_restores_full_prompt(tmp_path):
    rt = _runtime()
    ids = [index % 14 for index in range(513)]
    bank = SessionBank()
    entry = _publish(rt, bank, _state(rt, bank, ids))
    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    try:
        assert tier.put_entry(entry, capabilities=["ar_insert"])
        assert tier.flush(timeout_s=5)
        bank.clear()
        restarted = SessionBank(cold_tier=tier)
        learned = _state(rt, restarted, ids, 149)
        assert learned.ssd_cache_hit
        assert 0 < learned.cached_tokens < 149
        assert 149 in [record[0] for record in learned.gdn_boundaries]
        _publish(rt, restarted, learned)
        repeated = _state(rt, restarted, ids, 149)
        assert repeated.cached_tokens == len(ids)
        assert repeated.explicit_cache_boundary == 149
    finally:
        tier.close()
