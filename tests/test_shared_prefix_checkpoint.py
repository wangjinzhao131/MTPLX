"""A replayed common prefix becomes a usable checkpoint for the next branch."""

import mlx.core as mx
import pytest

from mtplx import generation
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_state import snapshot_untrimmable_cache
from mtplx.checkpoint_anchors import record_nbytes
from mtplx.session_bank import SessionBank
from test_checkpoint_anchors import _boundary, _put
from test_checkpoint_pre_image import _runtime


def test_shared_prefix_keeps_distinct_anchors_within_the_existing_budget():
    budget = 8 * record_nbytes(_boundary(1))
    bank = SessionBank(checkpoint_budget_bytes=budget)
    sink = generation._checkpoint_sink(
        [_boundary(64), _boundary(8192)], session_bank=bank, prompt_len=16122,
        restore_point=8192, shared_prefix_len=15072,
        system_prefix_len=15783, stable_prefix_len=16087,
    )
    for at in sorted({*range(8256, 16122, 64), 15072, 15783, 16087, 16121}):
        generation._append_gdn_boundary_record(sink, at, *_boundary(at)[1:])
        assert len(sink) <= 8
        assert sum(map(record_nbytes, sink)) <= budget
    assert {8192, 15072, 15783, 16087} <= {row[0] for row in sink}
    entry = _put(bank, 16122, list(sink))
    assert [row[0] for row in entry.gdn_boundaries] == [row[0] for row in sink]


def test_shared_edge_respects_image_spans_and_forward_policy():
    sink = generation._checkpoint_sink(
        [], session_bank=SessionBank(), prompt_len=100, restore_point=40,
        shared_prefix_len=48, image_spans=((43, 49),),
    )
    assert generation._checkpoint_edges(sink, None, 59, 40) == ()
    generation._capture_gdn_boundary(sink, 48, [])
    assert not sink
    sink = generation._checkpoint_sink(
        [], session_bank=SessionBank(), prompt_len=50, restore_point=40,
        shared_prefix_len=49, image_spans=((43, 49),),
    )
    assert generation._checkpoint_edges(sink, None, 9, 40) == (9,)
    sink.cuts_forwards = False
    assert generation._checkpoint_edges(sink, None, 9, 40) == ()


def _restore(rt, bank, prompt):
    return generation._restore_near_prefix_prompt_state(
        rt, prompt, base_hidden_variant=None, mtp_hidden_variant=None,
        mtp_history_policy="cycle", session_bank=bank, template_hash=None,
        draft_head_identity=None, policy_fingerprint=None,
        allow_block_prefix=True, chunk_started_s=0.0,
    )


@pytest.mark.parametrize("restore,matched,prompt_len,donor_len", [
    (8192, 15072, 16122, 16153),
    (128, 190, 191, 256),  # The shared edge is the body end of a fused-size suffix.
])
def test_replayed_match_restores_exactly_after_ssd_restart(
    tmp_path, monkeypatch, restore, matched, prompt_len, donor_len,
):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2048")
    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "64")
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "64")
    monkeypatch.setenv("MTPLX_SMALL_SUFFIX_FUSED_MAX", "512")
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_CAPTURE", "1")
    rt = _runtime()
    donor = [i % 14 for i in range(donor_len)]
    cache = rt.make_cache()
    records = []
    start = 0
    for end in (restore, donor_len):
        logits = rt.forward_ar(mx.array([donor[start:end]]), cache=cache, logits_keep=1)
        mx.eval(logits, cache[1].state)
        if end == restore:
            records.append((end, snapshot_untrimmable_cache(cache), None))
        start = end
    budget = 8 * record_nbytes(records[0])
    bank = SessionBank(checkpoint_budget_bytes=budget)
    bank.put(
        runtime=rt, token_ids=donor, cache=cache, logits=logits[:, -1, :],
        hidden=None, mtp_history_policy="cycle", gdn_boundaries=records,
    )
    prompt = donor[:matched] + [14] * (prompt_len - matched)
    state = _restore(rt, bank, prompt)
    assert state.cached_tokens == restore
    assert state.restore_served["requested_matched"] == matched
    assert state.restore_served["actual_restore_point"] == restore
    assert state.trunk_cache[0].offset == len(prompt)
    assert state.gdn_boundaries.plan.shared_prefix == matched
    assert matched in [row[0] for row in state.gdn_boundaries]
    assert sum(map(record_nbytes, state.gdn_boundaries)) <= budget
    entry = bank.put(
        runtime=rt, token_ids=prompt, cache=state.trunk_cache, logits=state.logits,
        hidden=state.hidden, mtp_history_policy="cycle",
        gdn_boundaries=state.gdn_boundaries,
    )
    assert entry.token_ids == tuple(prompt)
    assert entry.recurrent_boundary_at_or_below(matched)[0] == matched
    assert sum(map(record_nbytes, entry.gdn_boundaries)) <= budget

    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    try:
        assert tier.put_entry(entry, capabilities=["ar_insert"])
        assert tier.flush(timeout_s=5)
    finally:
        tier.close()
    bank.clear()
    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    try:
        bank = SessionBank(cold_tier=tier, checkpoint_budget_bytes=budget)
        next_prompt = donor[:matched] + [15] * (prompt_len - matched)
        warm = _restore(rt, bank, next_prompt)
        assert warm.ssd_cache_hit
        assert warm.cached_tokens == matched > restore
        assert warm.restore_served["requested_matched"] == matched
        assert warm.restore_served["boundary_restore"]
        assert warm.trunk_cache[0].offset == len(next_prompt)
        assert warm.gdn_boundaries.plan.shared_prefix is None
        expected = rt.forward_ar(
            mx.array([next_prompt]), cache=rt.make_cache(), logits_keep=1,
        )
        assert mx.array_equal(warm.logits, expected[:, -1, :]).item()
        wrong_history = [(next_prompt[0] + 1) % 14, *next_prompt[1:]]
        assert _restore(rt, bank, wrong_history) is None
    finally:
        tier.close()
