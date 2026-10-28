"""Tool-turn persistence holds keep only identifiers until the idle release."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.session_bank import SessionBank


RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=False)


class _KV:
    def __init__(self, rows):
        self.state = mx.ones((1, 1, rows, 2), dtype=mx.float16)
        self.meta_state = ("kv", str(rows))

    def is_trimmable(self):
        return True


class _Tier:
    def __init__(self):
        self.encoded = []
        self.spilled = []

    def put_entry(self, entry, **kwargs):
        self.encoded.append(
            (entry.token_ids, entry.snapshot_epoch, entry.snapshot_settled_at is not None)
        )
        return True

    def spill_entry(self, entry, **kwargs):
        self.spilled.append((entry.token_ids, entry.snapshot_epoch))
        return True

    def is_published(self, entry):
        return False


class _Lane:
    def __init__(self):
        self.pending = {}
        self.cancelled = []

    def dispatch(self, job):
        self.pending[job.coalesce_key] = job

    def cancel(self, key):
        self.cancelled.append(key)
        return int(self.pending.pop(key, None) is not None)

    def drain(self):
        while self.pending:
            self.pending.pop(next(iter(self.pending)))()


@pytest.fixture(autouse=True)
def _lazy_settle(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_LAZY_SNAPSHOT", "1")
    monkeypatch.setenv("MTPLX_SESSION_SNAPSHOT_SETTLE", "1")


def _bank(lane=None, **kwargs):
    bank = SessionBank(
        max_entries=8, max_bytes=1 << 20, cold_tier=_Tier(), **kwargs
    )
    if lane is not None:
        bank.cold_enqueue_dispatch = lane.dispatch
        bank.cold_enqueue_cancel = lane.cancel
    return bank


def _put(bank, tokens=(1, 2, 3), *, session_id="tool", epoch=1, **kwargs):
    entry = bank.put(
        runtime=RUNTIME,
        token_ids=tokens,
        cache=[_KV(len(tokens))],
        logits=None,
        hidden=None,
        session_id=session_id,
        snapshot_epoch=epoch,
        **kwargs,
    )
    assert entry is not None
    return entry


@pytest.mark.parametrize("tokens", [(1, 2, 3), (1, 2, 3, 4)])
def test_held_put_releases_only_latest_tokens_and_epoch_in_settle_order(tokens):
    lane = _Lane()
    bank = _bank(lane)
    bank.hold_session_persistence("tool")
    old = _put(bank)
    latest = _put(bank, tokens, epoch=2)

    assert bank._entries[latest.token_ids] is latest
    assert latest.lazy_kv and latest.snapshot_settled_at is None
    assert lane.pending == {}
    assert bank.cold_tier.encoded == []
    assert bank._persistence_pending == {}
    # The delay owns token IDs and an epoch, never an entry or tensor.
    assert bank._persistence_deferred == {"tool": (tokens, 2)}
    assert all(type(token) is int for token in bank._persistence_deferred["tool"][0])
    assert type(bank._persistence_deferred["tool"][1]) is int
    assert all(entry is not old for entry in bank._entries.values())

    assert bank.release_session_persistence("tool") is True
    assert list(lane.pending) == ["snapshot_settle:tool", "ssd_cold:tool"]
    assert bank._persistence_deferred == {}
    assert "tool" not in bank._persistence_held
    lane.drain()

    assert bank.cold_tier.encoded == [(tokens, 2, True)]
    assert latest.cold_encode_completed_at is not None
    assert bank._persistence_pending == {}
    assert bank.release_session_persistence("tool") is False


@pytest.mark.parametrize("broken_dispatch", [False, True])
def test_hold_prevents_synchronous_encode_fallback(broken_dispatch):
    bank = _bank()
    dispatches = []
    if broken_dispatch:
        def reject(job):
            dispatches.append(job.coalesce_key)
            raise RuntimeError("idle lane unavailable")

        bank.cold_enqueue_dispatch = reject
    bank.hold_session_persistence("tool")
    entry = _put(bank)

    assert dispatches == []
    assert bank.cold_tier.encoded == []
    assert entry.snapshot_settled_at is None
    assert bank.release_session_persistence("tool") is True
    assert bank.cold_tier.encoded == [(entry.token_ids, 1, False)]
    if broken_dispatch:
        assert dispatches == ["snapshot_settle:tool", "ssd_cold:tool"]


def test_hold_blocks_live_reference_spill_until_release():
    lane = _Lane()
    bank = _bank(lane, per_session_max_bytes=1)
    bank.hold_session_persistence("tool")
    entry = _put(bank, keep_live_ref=True)

    assert entry.live_ref_only and entry.cache_ref is not None
    assert lane.pending == {}
    assert bank.run_live_ref_spill(entry.token_ids, entry.snapshot_epoch) is False
    assert bank.cold_tier.spilled == []
    assert bank.release_session_persistence("tool") is True
    assert list(lane.pending) == ["ssd_cold:tool"]
    lane.drain()

    assert bank.cold_tier.spilled == [(entry.token_ids, 1)]
    assert bank.cold_tier.encoded == []
    assert entry.cold_encode_completed_at is not None


def test_hold_cancels_own_queued_jobs_and_preserves_foreign_jobs():
    lane = _Lane()
    bank = _bank(lane)
    own = _put(bank)
    foreign = _put(bank, (8, 9), session_id="foreign")
    foreign_jobs = {
        key: job for key, job in lane.pending.items() if key.endswith(":foreign")
    }

    bank.hold_session_persistence("tool")

    assert set(lane.cancelled) == {"snapshot_settle:tool", "ssd_cold:tool"}
    assert lane.pending == foreign_jobs
    assert set(bank._persistence_pending) == set(foreign_jobs)
    assert bank._persistence_deferred == {"tool": (own.token_ids, 1)}
    lane.drain()
    assert bank.cold_tier.encoded == [(foreign.token_ids, 1, True)]

    assert bank.release_session_persistence("tool") is True
    lane.drain()
    assert bank.cold_tier.encoded[-1] == (own.token_ids, 1, True)


@pytest.mark.parametrize("clear_all", [False, True])
def test_clear_during_request_keeps_later_publication_held(clear_all):
    lane = _Lane()
    bank = _bank(lane)
    bank.hold_session_persistence("tool")
    _put(bank)
    bank.clear(session_id=None if clear_all else "tool")
    latest = _put(bank, (1, 2, 3, 4), epoch=2)
    assert lane.pending == {}
    assert bank._persistence_deferred == {"tool": (latest.token_ids, 2)}
    assert bank.release_session_persistence("tool") is True


@pytest.mark.parametrize("clear_all", [False, True])
def test_clear_during_hold_does_not_resurrect_superseded_entry(clear_all):
    lane = _Lane()
    bank = _bank(lane)
    bank.hold_session_persistence("tool")
    _put(bank)
    latest = _put(bank, (1, 2, 3, 4), epoch=2)
    assert bank._persistence_deferred == {"tool": (latest.token_ids, 2)}

    assert bank.clear(session_id=None if clear_all else "tool") == 1
    assert bank.release_session_persistence("tool") is False

    assert bank._entries == {}
    assert bank._persistence_deferred == {}
    assert "tool" not in bank._persistence_held
    assert lane.pending == {}
    assert bank.cold_tier.encoded == []
    assert bank.cold_tier.spilled == []
