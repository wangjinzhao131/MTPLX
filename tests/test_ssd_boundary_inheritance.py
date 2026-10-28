"""Restored checkpoints survive a new capture, publication, and SSD restart."""

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_bank.codec import ColdEncodeInterrupted
from mtplx.cache_state import CacheSnapshot, snapshot_cache, snapshot_untrimmable_cache
from mtplx.checkpoint_anchors import AnchorPlan
from mtplx.generation import _inherited_gdn_boundaries
from mtplx.session_bank import SessionBank, SessionBankEntry


class _KV:
    meta_state = None

    def __init__(self, tokens=0):
        self.keys = mx.arange(tokens, dtype=mx.float32).reshape(1, 1, tokens, 1)
        self.offset = tokens

    @property
    def state(self):
        return (self.keys[:, :, : self.offset, :],)

    @state.setter
    def state(self, value):
        (self.keys,) = value
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, count):
        count = min(self.offset, count)
        self.offset -= count
        return count


class _Recurrent:
    meta_state = None

    def __init__(self, tokens=0):
        self.state = [mx.array([float(tokens)]), mx.array([tokens + 0.25])]

    def is_trimmable(self):
        return False

    def replace_state(self, state):
        self.state = list(state)


def _boundary(tokens):
    return (
        tokens,
        snapshot_untrimmable_cache([_KV(tokens), _Recurrent(tokens)]),
        mx.full((1, 1, 1), 10_000.0 + tokens),
    )


def _publish(bank, runtime, tokens, boundaries):
    count = len(tokens)
    entry = bank.put(
        runtime=runtime,
        token_ids=tokens,
        cache=[_KV(count), _Recurrent(count)],
        logits=mx.zeros((1, 4)),
        hidden=mx.full((1, 1, 1), 10_000.0 + count),
        hidden_variant="post_norm",
        session_id="boundary-inheritance",
        mtp_history_policy="committed",
        mtp_history_snapshot=snapshot_cache([_KV(count - 1)]),
        snapshot_epoch=count,
        mtp_snapshot_epoch=count,
        gdn_boundaries=boundaries,
    )
    assert entry is not None
    return entry


def _candidate(bank, runtime, tokens):
    candidates = bank.near_prefix_candidates(
        tokens,
        block_size=256,
        block_min_matched_tokens=256,
        allow_block_prefix=True,
        model_path=str(runtime.model_path),
        mtp_enabled=True,
        hidden_variant="post_norm",
        mtp_history_policy="committed",
    )
    assert candidates
    return candidates[0]


def _assert_restored_boundary(result, point):
    assert result is not None
    cache, history, _mode, restore_point, hidden = result
    assert restore_point == point
    assert cache[0].offset == point
    assert float(cache[1].state[0][0]) == point
    assert float(cache[1].state[1][0]) == point + 0.25
    assert float(hidden[0, 0, 0]) == 10_000.0 + point
    assert history[0].offset == point - 1


@pytest.mark.parametrize("partial", [False, True], ids=["exact", "partial"])
def test_ssd_boundaries_survive_new_capture_and_restart(tmp_path, monkeypatch, partial):
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    monkeypatch.setenv("MTPLX_SESSION_BOUNDARY_TRUE_RESTORE", "1")
    runtime = SimpleNamespace(
        model_path=Path("boundary-inheritance-tiny"),
        mtp_enabled=True,
        make_cache=lambda: [_KV(), _Recurrent()],
        make_mtp_cache=lambda: [_KV()],
    )
    base_dir = tmp_path / "session-bank"
    cold = SessionBankColdTier(base_dir=base_dir, mode="on", min_prefix_tokens=2)
    try:
        bank = SessionBank(cold_tier=cold, checkpoint_budget_bytes=1 << 20)
        original_tokens = list(range(1200))
        source = _publish(
            bank, runtime, original_tokens, [_boundary(n) for n in (256, 512, 1024)]
        )
        source_id = cold.entry_id_for(source)
        assert cold.flush(timeout_s=10.0)
        bank.clear()

        if partial:
            descendant_tokens = original_tokens[:600] + list(range(20_000, 20_200))
            restored_entry, matched = _candidate(bank, runtime, descendant_tokens)
            assert restored_entry.cache_source == "ssd"
            assert [r[0] for r in restored_entry.gdn_boundaries] == [512]
            assert restored_entry.gdn_boundary_loader is not None
            _assert_restored_boundary(
                bank.restore_entry_prefix_cache(runtime, restored_entry, matched), 512
            )
            restore_point, new_checkpoint = 512, 768
            expected_inherited = [256, 512]
        else:
            restored = bank.restore(
                runtime,
                original_tokens,
                hidden_variant="post_norm",
                mtp_history_policy="committed",
            )
            assert restored is not None and restored.cache_source == "ssd"
            restored_entry = restored.entry
            assert restored_entry.gdn_boundaries == []
            assert restored_entry.gdn_boundary_loader is not None
            assert restored.cache[0].offset == 1200
            assert restored.mtp_history_cache[0].offset == 1199
            descendant_tokens = original_tokens + list(range(20_000, 20_100))
            restore_point, new_checkpoint = 1200, 1280
            expected_inherited = [256, 512, 1024]

        inherited = _inherited_gdn_boundaries(
            restored_entry, restore_point, budget_bytes=bank.checkpoint_budget_bytes
        )
        assert [r[0] for r in inherited] == expected_inherited
        descendant = _publish(
            bank, runtime, descendant_tokens, inherited + [_boundary(new_checkpoint)]
        )
        expected_persisted = expected_inherited + [new_checkpoint]
        assert [r[0] for r in descendant.gdn_boundaries] == expected_persisted
        assert cold.flush(timeout_s=10.0)

        with cold._connect() as connection:
            source_row = connection.execute(
                "SELECT * FROM entries WHERE entry_id = ?", (source_id,)
            ).fetchone()
            descendant_row = connection.execute(
                "SELECT * FROM entries WHERE entry_id = ?", (cold.entry_id_for(descendant),)
            ).fetchone()
        assert source_row is not None and descendant_row is not None
        payload = json.loads(
            (base_dir / descendant_row["entry_dir"] / "payload.json").read_text()
        )
        assert [
            r["tokens"] for r in payload["payload_spec"]["gdn_boundaries"]
        ] == expected_persisted

        # The original donor must not hide a boundary-less descendant on lookup.
        cold._delete_entry_row(source_row)
        bank.clear()
        cold.close()
        cold = SessionBankColdTier(base_dir=base_dir, mode="on", min_prefix_tokens=2)
        restarted = SessionBank(cold_tier=cold, checkpoint_budget_bytes=1 << 20)
        branch = original_tokens[:300] + [30_000, 30_001]
        entry, matched = _candidate(restarted, runtime, branch)
        assert entry.cache_source == "ssd"
        assert entry.token_ids == tuple(descendant_tokens)
        assert matched == 300
        _assert_restored_boundary(
            restarted.restore_entry_prefix_cache(runtime, entry, matched), 256
        )
    finally:
        cold.close()


def _entry_with_selected_boundary(loader):
    return SessionBankEntry(
        token_ids=tuple(range(1200)),
        token_hash="boundary-inheritance",
        model_path="boundary-inheritance-tiny",
        mtp_enabled=False,
        hidden_variant=None,
        cache_snapshot=CacheSnapshot(states=(), meta_states=()),
        logits=None,
        hidden=None,
        has_recurrent=True,
        gdn_boundaries=[_boundary(512)],
        gdn_boundary_loader=loader,
    )


def test_interrupted_boundary_merge_keeps_selected_and_retries():
    calls = []

    def loader(should_abort=None):
        calls.append(True)
        if should_abort is not None and should_abort():
            raise ColdEncodeInterrupted()
        return [_boundary(256)]

    entry = _entry_with_selected_boundary(loader)
    plan = AnchorPlan(prompt_end=1200, tail_backoff=0)
    with pytest.raises(ColdEncodeInterrupted):
        entry._ensure_boundaries_loaded(plan=plan, should_abort=lambda: True)
    assert entry.gdn_boundary_loader is loader
    assert [r[0] for r in entry.gdn_boundaries] == [512]

    entry._ensure_boundaries_loaded(plan=plan, should_abort=lambda: False)
    assert len(calls) == 2
    assert entry.gdn_boundary_loader is None
    assert [r[0] for r in entry.gdn_boundaries] == [256, 512]
    assert float(entry.gdn_boundaries[1][2][0, 0, 0]) == 10_512.0


def test_corrupt_boundary_loader_preserves_selected():
    calls = []

    def loader():
        calls.append(True)
        raise FileNotFoundError("missing older boundary")

    entry = _entry_with_selected_boundary(loader)
    entry._ensure_boundaries_loaded(plan=AnchorPlan(prompt_end=1200, tail_backoff=0))
    assert len(calls) == 1
    assert [r[0] for r in entry.gdn_boundaries] == [512]
    assert float(entry.gdn_boundaries[0][2][0, 0, 0]) == 10_512.0
    assert float(entry.gdn_boundaries[0][1].states[1][0][0]) == 512.0
