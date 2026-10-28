"""A hybrid SSD entry with no recurrent boundary at or below a partial match
is refused before any tensor is read, and the next ranked row is tried.

The restore serves a hybrid entry's prefix only at a stored recurrent
boundary (SessionBank.restore_entry_prefix_cache, tiny gaps included since
2026-09-08). On 2026-10-01 a Pi turn shared its tool definitions with a
131,735-token conversation on the SSD tier that had no boundaries: every
attempt decoded that entry's 4.17 GB for a candidate the restore refused,
and the decoded copy stayed resident through the cold prefill that followed.
"""

from __future__ import annotations

import time

import mlx.core as mx
import pytest

import mtplx.cache_bank.cold_tier as cold_tier
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot


def _entry(tokens, *, recurrent: bool):
    class Entry:
        token_ids = tuple(tokens)
        nbytes = 2048
        cache_snapshot = CacheSnapshot(
            states=[mx.zeros((1, 2, 8, 4), dtype=mx.float16)],
            meta_states=[{"offset": len(tokens)}],
        )
        logits = mx.zeros((1, 8), dtype=mx.float16)
        hidden = mx.zeros((1, 8), dtype=mx.float16)
        mtp_history_snapshot = None
        gdn_boundaries = ()
        has_recurrent = recurrent
        session_id = "s1"
        token_hash = f"hash-{len(tokens):04d}" * 2
        prefix_len = len(tokens)
        model_path = "model"
        mtp_enabled = False
        hidden_variant = None
        template_hash = None
        mtp_history_policy = None
        policy_fingerprint = None

    return Entry()


def _tier(tmp_path):
    return SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)


def _store(tier, *entries):
    for entry in entries:
        assert tier.put_entry(entry, capabilities=["ar_insert"]) is True
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if tier.stats()["writes_completed"] >= len(entries):
            return
        time.sleep(0.05)
    raise AssertionError("writer did not complete")


def _lookup(tier, tokens):
    return tier.lookup_prefix_boundary(
        tokens,
        model_path="model",
        mtp_enabled=False,
        max_token_gap=8,
        min_matched_tokens=8,
        block_size=16,
        block_min_matched_tokens=16,
    )


def _no_decode(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a refused row must not be decoded")

    monkeypatch.setattr(cold_tier, "decode_payload", refuse)
    monkeypatch.setattr(cold_tier, "decode_payload_prefix", refuse)


def test_a_hybrid_entry_without_a_boundary_is_refused_before_decoding(tmp_path, monkeypatch):
    tier = _tier(tmp_path)
    stored = list(range(2000))
    _store(tier, _entry(stored, recurrent=True))
    _no_decode(monkeypatch)

    query = tuple(stored[:1500] + [9999] * 64)
    assert _lookup(tier, query) is None
    stats = tier.stats()
    assert stats["last_miss_reason"] == "ssd_prefix_no_recurrent_boundary"
    assert stats["prefix_restores_without_boundary"] == 1
    assert stats.get("restore_hits", 0) == 0


def test_the_next_ranked_row_serves_when_the_longest_cannot(tmp_path):
    tier = _tier(tmp_path)
    hybrid = list(range(2000))
    attention = list(range(1200)) + [5555] * 100
    _store(tier, _entry(hybrid, recurrent=True), _entry(attention, recurrent=False))

    query = tuple(hybrid[:1500] + [9999] * 64)
    hit = _lookup(tier, query)
    assert hit is not None
    assert hit.record.token_ids == tuple(attention)
    assert hit.matched_tokens == 1200
    assert tier.stats()["prefix_restores_without_boundary"] == 1


def test_with_boundary_true_restores_off_the_full_entry_still_decodes(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BOUNDARY_TRUE_RESTORE", "0")
    tier = _tier(tmp_path)
    stored = list(range(2000))
    _store(tier, _entry(stored, recurrent=True))

    query = tuple(stored[:1500] + [9999] * 64)
    hit = _lookup(tier, query)
    assert hit is not None
    assert hit.record.token_ids == tuple(stored)


def test_longest_recoverable_prefix_wins_before_tensor_decode(tmp_path, monkeypatch):
    tier = _tier(tmp_path)
    prompt = tuple(range(1500))
    entries = []
    for matched, boundary in ((1000, 512), (900, 768)):
        entry = _entry(prompt[:matched] + (9999,), recurrent=True)
        entry.gdn_boundaries = [(boundary, CacheSnapshot(states=entry.cache_snapshot.states, meta_states=[{"offset": boundary}]), None)]
        entries.append(entry)
    _store(tier, *entries)
    decoded = []
    original = tier._restore_row

    def restore(row, *args, **kwargs):
        decoded.append(row["prefix_len"])
        return original(row, *args, **kwargs)

    monkeypatch.setattr(tier, '_restore_row', restore)
    hit = _lookup(tier, prompt)
    assert hit is not None
    assert hit.record.token_ids == entries[1].token_ids
    assert decoded == [entries[1].prefix_len]
    assert hit.record.gdn_boundaries[0][0] == 768


def test_unreachable_raw_match_does_not_hydrate_below_ram_floor(tmp_path, monkeypatch):
    tier = _tier(tmp_path)
    prompt = tuple(range(1500))
    entry = _entry(prompt[:1000] + (9999,), recurrent=True)
    entry.gdn_boundaries = [(512, CacheSnapshot(states=(), meta_states=()), None)]
    _store(tier, entry)
    _no_decode(monkeypatch)
    assert tier.lookup_prefix_boundary(
        prompt, model_path='model', mtp_enabled=False,
        min_useful_matched_tokens=768,
    ) is None
    assert tier.stats()['last_miss_reason'] == 'ssd_prefix_not_better_than_ram'


def test_unusable_metadata_does_not_hide_a_later_recoverable_prefix(tmp_path):
    tier = _tier(tmp_path)
    prompt = tuple(range(1500))
    entries = [_entry(prompt[:matched] + (9999,), recurrent=True)
               for matched in (1200, 1100, 1000, 900)]
    usable = _entry(prompt[:800] + (9999,), recurrent=False)
    _store(tier, *entries, usable)
    hit = _lookup(tier, prompt)
    assert hit is not None
    assert hit.record.token_ids == usable.token_ids


@pytest.mark.parametrize("ram_floor", [0, 1490])
def test_block_match_keeps_unaligned_recurrent_checkpoint(tmp_path, ram_floor):
    tier = _tier(tmp_path)
    entry = _entry(range(2000), recurrent=True)
    kv = mx.arange(4000, dtype=mx.float32).reshape(1, 1, 2000, 2)
    entry.cache_snapshot = CacheSnapshot(states=(kv, mx.zeros((1, 4))), meta_states=(None, None))
    entry.mtp_history_snapshot = CacheSnapshot(states=(kv,), meta_states=(None,))
    entry.mtp_history_policy = "committed"
    entry.gdn_boundaries = [
        (p, CacheSnapshot(states=(None, mx.full((1, 4), p)), meta_states=(None, None)), None)
        for p in (1024, 1496, 1550)
    ]
    _store(tier, entry)
    try:
        hit = tier.lookup_prefix_boundary(
            tuple(range(1500)) + (9999,) * 64,
            model_path="model", mtp_enabled=False, max_token_gap=8,
            block_size=16, block_min_matched_tokens=16,
            min_useful_matched_tokens=ram_floor,
        )
        assert hit is not None
        assert hit.record.cache_snapshot_prefix_len == 1496
        assert hit.record.cache_snapshot.states[0].tolist() == kv[:, :, :1496].tolist()
        assert hit.record.mtp_history_snapshot_prefix_len == 1495
        assert hit.record.mtp_history_snapshot.states[0].tolist() == kv[:, :, :1495].tolist()
        assert hit.record.gdn_boundaries[0][0] == 1496
        assert hit.record.gdn_boundaries[0][1].states[1].tolist() == [[1496] * 4]
    finally:
        tier.close()
