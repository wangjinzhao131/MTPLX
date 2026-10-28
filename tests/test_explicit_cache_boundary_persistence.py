"""Explicit checkpoints survive bounded retention and SSD descendants."""

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mtplx.cache_bank.cold_tier as cold_module
from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_bank.codec import decode_payload, decode_payload_prefix, encode_payload
from mtplx.cache_state import snapshot_cache
from mtplx.session_bank import SessionBank

from test_ssd_boundary_inheritance import (
    _KV,
    _Recurrent,
    _assert_restored_boundary,
    _boundary,
    _candidate,
)


@pytest.fixture
def runtime():
    return SimpleNamespace(
        model_path=Path("explicit-boundary-tiny"),
        mtp_enabled=True,
        make_cache=lambda: [_KV(), _Recurrent()],
        make_mtp_cache=lambda: [_KV()],
    )


def _put(bank, runtime, tokens, boundaries=(), explicit=None):
    count = len(tokens)
    return bank.put(
        runtime=runtime,
        token_ids=tokens,
        cache=[_KV(count), _Recurrent(count)],
        logits=mx.zeros((1, 4)),
        hidden=mx.full((1, 1, 1), 10_000.0 + count),
        hidden_variant="post_norm",
        session_id="explicit-boundary",
        mtp_history_policy="committed",
        mtp_history_snapshot=snapshot_cache([_KV(count - 1)]),
        snapshot_epoch=count,
        mtp_snapshot_epoch=count,
        gdn_boundaries=list(boundaries),
        explicit_cache_boundary=explicit,
    )


def test_put_inherits_single_explicit_marker_and_current_request_overrides(runtime):
    bank = SessionBank(checkpoint_budget_bytes=12)
    source = _put(bank, runtime, list(range(1200)), [_boundary(n) for n in (256, 512, 1024)], 256)
    assert source.explicit_cache_boundary == 256
    assert [r[0] for r in source.gdn_boundaries] == [256]

    same = _put(bank, runtime, list(range(1200)))
    assert same.explicit_cache_boundary == 256
    descendant = _put(bank, runtime, list(range(1300)), [_boundary(1280)])
    assert descendant.explicit_cache_boundary == 256
    assert [r[0] for r in descendant.gdn_boundaries] == [256]
    assert descendant.prefix_len == 1300

    current = _put(bank, runtime, list(range(1400)), [_boundary(256), _boundary(1280)], 1280)
    assert current.explicit_cache_boundary == 1280
    assert [r[0] for r in current.gdn_boundaries] == [1280]
    other = _put(bank, runtime, [99_999, 99_998])
    assert other.explicit_cache_boundary is None


def test_put_does_not_inherit_boundary_past_materialized_frontier(runtime):
    bank = SessionBank(checkpoint_budget_bytes=24)
    source = _put(bank, runtime, list(range(1200)), [_boundary(256), _boundary(1024)], 1024)
    source.cache_snapshot_prefix_len = 512
    descendant = _put(bank, runtime, list(range(1300)))
    assert descendant.explicit_cache_boundary is None
    assert [r[0] for r in descendant.gdn_boundaries] == [256]


def test_put_does_not_import_an_incompatible_marked_donor(runtime):
    bank = SessionBank(checkpoint_budget_bytes=24)
    _put(bank, runtime, list(range(1200)), [_boundary(256)], 256)
    runtime.model_path = Path("different-model")
    descendant = _put(bank, runtime, list(range(1300)), [_boundary(1280)])
    assert descendant.explicit_cache_boundary is None
    assert [r[0] for r in descendant.gdn_boundaries] == [1280]


def test_admission_shedding_keeps_explicit_last_and_clears_dropped_marker(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BANK_SHED_BOUNDARIES", "1")
    runtime = SimpleNamespace(model_path=Path("explicit-shed"), mtp_enabled=False)
    for budget, expected in ((40, [100, 300]), (16, [])):
        bank = SessionBank(per_session_max_bytes=budget, max_bytes=1024)
        entry = bank.put(
            runtime=runtime,
            token_ids=list(range(400)),
            cache=[],
            logits=None,
            hidden=mx.zeros((16,), dtype=mx.uint8),
            gdn_boundaries=[_boundary(n) for n in (100, 200, 300)],
            explicit_cache_boundary=100,
        )
        assert entry is not None and entry.nbytes <= budget
        assert [r[0] for r in entry.gdn_boundaries] == expected
        assert entry.explicit_cache_boundary == (100 if expected else None)


def test_codec_marker_is_optional_and_requires_an_available_proven_checkpoint():
    encoded = encode_payload(
        cache_snapshot=snapshot_cache([_KV(1200), _Recurrent(1200)]),
        logits=None,
        hidden=None,
        mtp_history_snapshot=None,
        gdn_boundaries=[_boundary(256), _boundary(1024)],
        explicit_cache_boundary=1024,
    )
    assert decode_payload(encoded.spec, encoded.tensors.__getitem__, include_gdn_boundaries=False).explicit_cache_boundary == 1024
    partial = decode_payload_prefix(
        encoded.spec, encoded.tensors.__getitem__, cache_prefix_len=256, boundary_prefix_len=256
    )
    assert partial.explicit_cache_boundary is None
    for marker in (None, True, "256", -1, 512):
        spec = {**encoded.spec, "explicit_cache_boundary": marker}
        assert decode_payload(spec, encoded.tensors.__getitem__, include_gdn_boundaries=False).explicit_cache_boundary is None
    legacy = dict(encoded.spec)
    legacy.pop("explicit_cache_boundary")
    assert decode_payload(legacy, encoded.tensors.__getitem__, include_gdn_boundaries=False).explicit_cache_boundary is None


@pytest.mark.parametrize("writer", ["put_entry", "spill_entry"])
def test_explicit_marker_survives_bounded_ssd_inheritance_and_restart(tmp_path, monkeypatch, runtime, writer):
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    monkeypatch.setenv("MTPLX_SESSION_BOUNDARY_TRUE_RESTORE", "1")
    base_dir = tmp_path / "session-bank"
    cold = SessionBankColdTier(base_dir=base_dir, mode="on", min_prefix_tokens=2)
    try:
        old = _put(SessionBank(), runtime, list(range(1200)), [_boundary(512)])
        assert getattr(cold, writer)(old) and cold.flush(timeout_s=10.0)
        source = _put(SessionBank(), runtime, list(range(1200)), [_boundary(n) for n in (256, 512, 1024)], 256)
        assert cold.entry_id_for(source) != cold.entry_id_for(old)
        assert getattr(cold, writer)(source)
        assert cold.flush(timeout_s=10.0)
        source_id = cold.entry_id_for(source)
        bank = SessionBank(cold_tier=cold, checkpoint_budget_bytes=12)
        restored = bank.restore(runtime, list(range(1200)), required_boundary=256)
        assert restored is not None and restored.cache_source == "ssd"
        entry = restored.entry
        assert entry.explicit_cache_boundary == 256
        assert entry.gdn_boundaries == []
        assert entry.has_recurrent_boundary(256)
        bank._hydrate_boundaries_for_persistence(entry)
        assert [r[0] for r in entry.gdn_boundaries] == [256]

        descendant = _put(bank, runtime, list(range(1300)), [_boundary(1280)])
        assert descendant.explicit_cache_boundary == 256
        assert [r[0] for r in descendant.gdn_boundaries] == [256]
        assert cold.flush(timeout_s=10.0)
        with cold._connect() as connection:
            old_row = connection.execute("SELECT * FROM entries WHERE entry_id = ?", (cold.entry_id_for(old),)).fetchone()
            source_row = connection.execute("SELECT * FROM entries WHERE entry_id = ?", (source_id,)).fetchone()
            descendant_row = connection.execute("SELECT * FROM entries WHERE entry_id = ?", (cold.entry_id_for(descendant),)).fetchone()
        payload = json.loads((base_dir / descendant_row["entry_dir"] / "payload.json").read_text())
        assert payload["payload_spec"]["explicit_cache_boundary"] == 256
        cold._delete_entry_row(old_row)
        cold._delete_entry_row(source_row)
        bank.clear()
        cold.close()
        cold = SessionBankColdTier(base_dir=base_dir, mode="on", min_prefix_tokens=2)
        restarted = SessionBank(cold_tier=cold, checkpoint_budget_bytes=12)
        branch = list(range(350)) + [90_000, 90_001]
        entry, matched = _candidate(restarted, runtime, branch)
        assert entry.explicit_cache_boundary == 256
        assert entry.token_ids == tuple(range(1300))
        _assert_restored_boundary(restarted.restore_entry_prefix_cache(runtime, entry, matched), 256)
    finally:
        cold.close()


def test_missing_required_boundary_does_not_consume_ram_reference(runtime):
    bank = SessionBank()
    source = _put(bank, runtime, list(range(1200)), [_boundary(256)])
    source.cache_ref = [_KV(1200), _Recurrent(1200)]
    original_ref = source.cache_ref
    assert bank.restore(runtime, list(range(1200)), mode="reference", required_boundary=512) is None
    assert source.cache_ref is original_ref
    assert bank.last_miss_reason == "explicit_cache_boundary_missing"
    assert bank.restore(runtime, list(range(1200)), required_boundary=256) is not None


def test_ssd_exact_lookup_selects_variant_with_required_boundary(tmp_path, monkeypatch, runtime):
    base_dir = tmp_path / "session-bank"
    cold = SessionBankColdTier(base_dir=base_dir, mode="on", min_prefix_tokens=2)
    try:
        for last_access, boundary in enumerate((256, 512), 1):
            source = _put(
                SessionBank(checkpoint_budget_bytes=12), runtime,
                list(range(1200)), [_boundary(256), _boundary(512)], boundary,
            )
            assert [record[0] for record in source.gdn_boundaries] == [boundary]
            assert cold.put_entry(source) and cold.flush(timeout_s=10.0)
            with cold._connect() as connection:
                connection.execute(
                    "UPDATE entries SET last_access_s = ? WHERE entry_id = ?",
                    (last_access, cold.entry_id_for(source)),
                )
        cold.close()
        cold = SessionBankColdTier(base_dir=base_dir, mode="on", min_prefix_tokens=2)
        decoded_markers = []
        original_decode = cold_module.decode_payload

        def decode(spec, *args, **kwargs):
            decoded_markers.append(spec["explicit_cache_boundary"])
            return original_decode(spec, *args, **kwargs)

        monkeypatch.setattr(cold_module, "decode_payload", decode)
        bank = SessionBank(cold_tier=cold, checkpoint_budget_bytes=12)
        restored = bank.restore(runtime, list(range(1200)), required_boundary=256)
        assert restored is not None and restored.cache_source == "ssd"
        assert restored.entry.prefix_len == 1200
        assert restored.entry.explicit_cache_boundary == 256
        assert restored.entry.has_recurrent_boundary(256)
        assert decoded_markers == [256]
    finally:
        cold.close()


def test_partial_ssd_restore_drops_unproven_marker(tmp_path, monkeypatch, runtime):
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    monkeypatch.setenv("MTPLX_SESSION_BOUNDARY_TRUE_RESTORE", "1")
    cold = SessionBankColdTier(base_dir=tmp_path / "session-bank", mode="on", min_prefix_tokens=2)
    try:
        source = _put(SessionBank(), runtime, list(range(1200)), [_boundary(256), _boundary(1024)], 1024)
        assert cold.put_entry(source) and cold.flush(timeout_s=10.0)
        bank = SessionBank(cold_tier=cold)
        assert bank.restore(runtime, list(range(1200)), required_boundary=512) is None
        entry, _matched = _candidate(bank, runtime, list(range(350)) + [90_000, 90_001])
        assert entry.explicit_cache_boundary is None
        assert entry.has_recurrent_boundary(256)
        assert not entry.has_recurrent_boundary(1024)
    finally:
        cold.close()
