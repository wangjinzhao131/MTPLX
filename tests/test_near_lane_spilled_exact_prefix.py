"""A spilled session is restored whole from SSD, not from a neighbour's head.

Measured on 2026-10-09 (Qwen3.8-27B, 7 GB bank, two conversations sharing a
4,019-token system prompt): after the bank spilled conversation A to SSD to
make room for B, A's next turn found no exact prefix in RAM and took the
near-prefix lane first. That lane asked the SSD tier for its best candidate,
which was A's own spilled entry, a whole prefix of the prompt; it hydrated
it (5.4 GB), refused it (the lane serves only matched < prefix_len; whole
prefixes are the exact lane's), and served B's checkpoint at the shared
system prompt instead. The turn restored 4,019 tokens and re-read 54,510
(143 s), while session_bank.restore() would have read A's 57,989 tokens back
from SSD. The near lane now has to beat the longest SSD whole prefix, read
from the manifest without hydrating anything.

CPU-sized: toy runtime, real SessionBank and cold tier on tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import mlx.core as mx
import pytest

from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot
from mtplx.generation import restore_or_prefill_prompt_state
from mtplx.session_bank import SessionBank

MODEL = str(Path("models/example"))
SPILLED = list(range(1, 42))  # conversation A, on SSD only
PROMPT = tuple(SPILLED + [70, 71, 72])  # A's next turn
NEIGHBOUR = SPILLED[:16] + [90 + i for i in range(30)]  # B shares 16 tokens


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


def _tier(tmp_path) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=tmp_path / "session-bank", mode="on", min_prefix_tokens=2
    )


def _bank(cold) -> SessionBank:
    return SessionBank(
        max_entries=8, max_bytes=4096, per_session_max_bytes=4096, cold_tier=cold
    )


def _put(bank: SessionBank, token_ids, session_id: str):
    return bank.put_snapshot(
        runtime=_runtime(),
        token_ids=token_ids,
        cache_snapshot=CacheSnapshot(states=(), meta_states=()),
        logits=None,
        hidden=None,
        session_id=session_id,
        snapshot_epoch=len(token_ids),
        nbytes_override=128,
    )


def _spill_a_keep_b(tmp_path):
    cold = _tier(tmp_path)
    bank = _bank(cold)
    _put(bank, SPILLED, "a")
    assert cold.flush(timeout_s=5.0) is True
    bank.clear()
    _put(bank, NEIGHBOUR, "b")
    return cold, bank


def _count_hydrations(cold, monkeypatch) -> list:
    calls: list = []
    original = cold._restore_row

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cold, "_restore_row", spy)
    return calls


def test_the_probe_reads_metadata_without_lookup_side_effects(tmp_path, monkeypatch):
    cold, bank = _spill_a_keep_b(tmp_path)
    try:
        calls = _count_hydrations(cold, monkeypatch)
        before = cold.stats()
        probe = {"model_path": MODEL, "mtp_enabled": True}
        assert bank.cold_exact_prefix_len(PROMPT, **probe) == len(SPILLED)
        # B's prompt shares only the opening: no whole prefix on SSD.
        assert bank.cold_exact_prefix_len(tuple(NEIGHBOUR), **probe) == 0
        # Another identity never matches.
        assert bank.cold_exact_prefix_len(PROMPT, model_path="other", mtp_enabled=True) == 0
        assert calls == []
        after = cold.stats()
        for key in ("restore_hits", "restore_misses", "restore_failures"):
            assert after[key] == before[key]
        assert after["last_miss_reason"] == before["last_miss_reason"]
    finally:
        cold.close()


def test_the_probe_is_zero_without_a_readable_tier(tmp_path):
    assert SessionBank(max_entries=2, max_bytes=4096).cold_exact_prefix_len(
        PROMPT, model_path=MODEL, mtp_enabled=True
    ) == 0
    cold = SessionBankColdTier(
        base_dir=tmp_path / "write-only", mode="write-only", min_prefix_tokens=2
    )
    try:
        assert cold.exact_prefix_len(PROMPT, model_path=MODEL, mtp_enabled=True) == 0
    finally:
        cold.close()


@pytest.mark.parametrize("field,value", [
    ("format_version", -1), ("mtp_snapshot_epoch", 0), ("prefix_len", 40),
    ("token_ids_json", "[1, 2, 999]"), ("token_ids_json", "not-json"),
    ("model_path", "other"), ("mtp_enabled", 0),
    ("hidden_variant", "other"), ("template_hash", "other"),
    ("mtp_history_policy", "cycle"), ("draft_head_identity", "other"),
    ("policy_fingerprint", "other"),
])
def test_probe_preserves_exact_restore_identity_and_epoch_gates(tmp_path, field, value):
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        identity = dict(
            hidden_variant="hidden", template_hash="template",
            mtp_history_policy="committed", draft_head_identity="draft",
            policy_fingerprint="policy",
        )
        bank.put_snapshot(
            runtime=_runtime(), token_ids=SPILLED,
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=mx.zeros((1, 4)), hidden=mx.zeros((1, 1, 2)),
            mtp_history_snapshot=CacheSnapshot(states=(), meta_states=()),
            snapshot_epoch=41, mtp_snapshot_epoch=41, nbytes_override=128,
            **identity,
        )
        assert cold.flush(timeout_s=5)
        probe = dict(model_path=MODEL, mtp_enabled=True, **identity)
        assert bank.cold_exact_prefix_len(PROMPT, **probe) == 41
        with cold._connect() as conn:
            conn.execute(f"UPDATE entries SET {field} = ?", (value,))
        assert bank.cold_exact_prefix_len(PROMPT, **probe) == 0
    finally:
        cold.close()


@pytest.mark.parametrize("damage", [
    "missing", "json", "root", "snapshot", "states", "logits", "hidden", "mtp",
])
def test_probe_refuses_incomplete_or_corrupt_payload_metadata(tmp_path, damage):
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        bank.put_snapshot(
            runtime=_runtime(), token_ids=SPILLED,
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=mx.zeros((1, 4)), hidden=mx.zeros((1, 1, 2)),
            hidden_variant="hidden", mtp_history_policy="committed",
            mtp_history_snapshot=CacheSnapshot(states=(), meta_states=()),
            snapshot_epoch=41, mtp_snapshot_epoch=41, nbytes_override=128,
        )
        assert cold.flush(timeout_s=5)
        probe = dict(model_path=MODEL, mtp_enabled=True, hidden_variant="hidden",
                     mtp_history_policy="committed")
        assert bank.cold_exact_prefix_len(PROMPT, **probe) == 41
        with cold._connect() as conn:
            row = conn.execute("SELECT entry_dir FROM entries").fetchone()
        path = cold.base_dir / row["entry_dir"] / "payload.json"
        payload = json.loads(path.read_text())
        spec = payload["payload_spec"]
        if damage == "missing":
            path.unlink()
        elif damage == "json":
            path.write_text("{")
        elif damage == "root":
            path.write_text("[]")
        else:
            if damage == "snapshot":
                spec["cache_snapshot"] = None
            elif damage == "states":
                spec["cache_snapshot"]["states"] = {"kind": "none"}
            elif damage == "mtp":
                spec["mtp_history_snapshot"] = None
            else:
                spec[damage] = {"kind": "none"}
            path.write_text(json.dumps(payload))
        assert bank.cold_exact_prefix_len(PROMPT, **probe) == 0
    finally:
        cold.close()


def test_probe_respects_required_recurrent_checkpoint(tmp_path):
    cold = _tier(tmp_path)
    try:
        bank = _bank(None)
        snapshot = CacheSnapshot(states=(), meta_states=())
        entry = bank.put_snapshot(
            runtime=_runtime(), token_ids=SPILLED, cache_snapshot=snapshot,
            logits=None, hidden=None, nbytes_override=128,
        )
        entry.has_recurrent = True
        entry.gdn_boundaries = [(16, snapshot)]
        assert cold.put_entry(entry)
        assert cold.flush(timeout_s=5)
        bank.cold_tier = cold
        probe = dict(model_path=MODEL, mtp_enabled=True)
        assert bank.cold_exact_prefix_len(PROMPT, required_boundary=16, **probe) == 41
        assert bank.cold_exact_prefix_len(PROMPT, required_boundary=17, **probe) == 0
        assert bank.cold_exact_prefix_len(PROMPT, required_boundary=41, **probe) == 41
    finally:
        cold.close()


def _near(bank: SessionBank, floor: int):
    return bank.near_prefix_candidates(
        PROMPT,
        min_matched_tokens=4,
        block_size=8,
        block_min_matched_tokens=8,
        model_path=MODEL,
        mtp_enabled=True,
        min_restore_tokens=floor,
    )


def test_without_the_floor_the_spilled_session_is_hydrated_for_nothing(
    tmp_path, monkeypatch
):
    """The 2026-10-09 shape: the best candidate is A's whole prefix (which the
    near lane refuses), hydrated on the request path."""
    cold, bank = _spill_a_keep_b(tmp_path)
    try:
        calls = _count_hydrations(cold, monkeypatch)
        matches = _near(bank, 0)
        assert calls == [1]
        entry, matched = matches[0]
        assert getattr(entry, "cache_source", None) == "ssd"
        assert matched == entry.prefix_len == len(SPILLED)
    finally:
        cold.close()


def test_at_the_floor_nothing_is_hydrated_and_no_candidate_beats_it(
    tmp_path, monkeypatch
):
    cold, bank = _spill_a_keep_b(tmp_path)
    try:
        calls = _count_hydrations(cold, monkeypatch)
        floor = bank.cold_exact_prefix_len(PROMPT, model_path=MODEL, mtp_enabled=True)
        matches = _near(bank, floor)
        assert calls == []
        assert all(int(matched) <= floor for _entry, matched in matches)
    finally:
        cold.close()


@pytest.fixture
def selected_ssd(tmp_path, monkeypatch):
    from test_checkpoint_pre_image import _runtime as hybrid_runtime
    from mtplx import generation

    for name, value in {
        "MTPLX_SUSTAINED_PREFILL": "1", "MTPLX_PREFILL_CHUNK_SIZE": "64",
        "MTPLX_SESSION_PREFIX_BLOCK_SIZE": "64",
        "MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS": "64",
        "MTPLX_GDN_BOUNDARY_CAPTURE": "1",
    }.items():
        monkeypatch.setenv(name, value)
    rt = hybrid_runtime()
    bank = SessionBank()
    cold = _tier(tmp_path)
    ids = [index % 14 for index in range(128)]

    def publish(tokens, session_id):
        state = restore_or_prefill_prompt_state(
            rt, tokens, session_bank=bank, mtp_history_policy="cycle",
            store_prefix_snapshot=False,
            explicit_cache_boundary=64 if len(tokens) > 64 else None,
        )
        return bank.put(
            runtime=rt, token_ids=tokens, cache=state.trunk_cache,
            logits=state.logits, hidden=state.hidden, session_id=session_id,
            hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
            mtp_history_policy="cycle", gdn_boundaries=state.gdn_boundaries,
        )

    entry = publish(ids, "spilled")
    assert cold.put_entry(entry)
    assert cold.flush(timeout_s=5)
    bank.clear()
    publish(ids[:16], "short")
    publish(ids[:64] + [15] * 80, "near")
    bank.cold_tier = cold
    identity = dict(
        model_path=str(rt.model_path), mtp_enabled=bool(rt.mtp_enabled),
        hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
        mtp_history_policy="cycle",
    )
    try:
        yield rt, bank, cold, ids + [14, 14], identity
    finally:
        cold.close()


@pytest.mark.parametrize("full_hit", [False, True])
def test_long_ssd_exact_beats_short_ram_exact_and_middle_ram_near(selected_ssd, monkeypatch, full_hit):
    from mtplx import generation

    rt, bank, cold, prompt, identity = selected_ssd
    if full_hit:
        prompt = prompt[:-2]
    ram_plan = bank.restore_plan(prompt, **identity)
    assert ram_plan["reuse_tokens"] == 64
    plan = bank.restore_plan(prompt, include_cold=True, **identity)
    assert plan["mode"] == "ssd_exact"
    assert plan["reuse_tokens"] == 128
    assert plan["restore_donor_nbytes"] > 0
    assert plan["source"] is None
    assert ram_plan["keys"] <= plan["keys"]
    floors = []
    original = generation._restore_near_prefix_prompt_state

    def observe(*args, **kwargs):
        floors.append(kwargs["min_restore_tokens"])
        return original(*args, **kwargs)

    monkeypatch.setattr(generation, "_restore_near_prefix_prompt_state", observe)
    state = restore_or_prefill_prompt_state(
        rt, prompt, session_bank=bank, store_prefix_snapshot=False,
    )
    assert floors == ([] if full_hit else [128])
    assert state.cached_tokens == 128
    assert state.ssd_cache_hit
    expected = rt.forward_ar(mx.array([prompt]), cache=rt.make_cache(), logits_keep=1)
    assert mx.array_equal(state.logits, expected[:, -1, :]).item()


@pytest.mark.parametrize("full_hit", [False, True])
@pytest.mark.parametrize("ram_exact", [16, 96])
def test_missing_tensor_after_probe_keeps_best_ram_fallback(selected_ssd, monkeypatch, full_hit, ram_exact):
    from mtplx import generation

    rt, bank, cold, prompt, identity = selected_ssd
    if full_hit:
        prompt = prompt[:-2]
    if ram_exact == 96:
        state = restore_or_prefill_prompt_state(rt, prompt[:96], store_prefix_snapshot=False)
        bank.cold_tier = None
        bank.put(
            runtime=rt, token_ids=prompt[:96], cache=state.trunk_cache,
            logits=state.logits, hidden=state.hidden, session_id="longer-exact",
            hidden_variant=identity["hidden_variant"], mtp_history_policy="cycle",
        )
        bank.cold_tier = cold
    assert bank.restore_plan(prompt, include_cold=True, **identity)["reuse_tokens"] == 128
    with cold._connect() as conn:
        row = conn.execute("SELECT entry_dir FROM entries").fetchone()
    payload = json.loads((cold.base_dir / row["entry_dir"] / "payload.json").read_text())
    blob = next(iter(payload["tensor_blobs"].values()))
    cold._blob_path(blob["sha256"]).unlink()
    floors = []
    original = generation._restore_near_prefix_prompt_state

    def observe(*args, **kwargs):
        floors.append(kwargs["min_restore_tokens"])
        return original(*args, **kwargs)

    monkeypatch.setattr(generation, "_restore_near_prefix_prompt_state", observe)
    state = restore_or_prefill_prompt_state(
        rt, prompt, session_bank=bank, store_prefix_snapshot=False,
    )
    assert floors == ([ram_exact] if full_hit else [128, ram_exact])
    assert state.cached_tokens == max(64, ram_exact)
    assert not state.ssd_cache_hit
    expected = rt.forward_ar(mx.array([prompt]), cache=rt.make_cache(), logits_keep=1)
    assert mx.array_equal(state.logits, expected[:, -1, :]).item()


def test_full_ram_exact_skips_cold_metadata_probe(selected_ssd, monkeypatch):
    rt, bank, cold, prompt, identity = selected_ssd
    state = restore_or_prefill_prompt_state(
        rt, prompt, session_bank=bank, store_prefix_snapshot=False,
    )
    bank.put(
        runtime=rt, token_ids=prompt, cache=state.trunk_cache, logits=state.logits,
        hidden=state.hidden, hidden_variant=identity["hidden_variant"],
        mtp_history_policy="cycle", gdn_boundaries=state.gdn_boundaries,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("full RAM hit must not read payload.json")

    monkeypatch.setattr(cold, "exact_prefix_plan", forbidden)
    assert bank.restore_plan(prompt, include_cold=True, **identity)["mode"] == "exact"
    again = restore_or_prefill_prompt_state(
        rt, prompt, session_bank=bank, store_prefix_snapshot=False,
    )
    assert again.cached_tokens == len(prompt)
    assert not again.ssd_cache_hit


def test_selected_ssd_plan_cannot_hydrate_a_new_larger_donor(selected_ssd, monkeypatch):
    import mtplx.cache_bank.cold_tier as cold_module

    rt, bank, cold, prompt, identity = selected_ssd
    plan = bank.restore_plan(prompt, include_cold=True, **identity)
    cap = plan["restore_donor_nbytes"]
    assert bank.restore_plan(prompt, include_cold=True, ssd_restore_max_bytes=0,
                             **identity)["mode"] == "near_prefix"
    with cold._connect() as conn:
        conn.execute("UPDATE entries SET nbytes = ?, logical_nbytes = ?", (cap + 1, cap + 1))
    assert bank.restore_plan(prompt, include_cold=True, ssd_restore_max_bytes=cap,
                             **identity)["mode"] == "near_prefix"

    def forbidden(*args, **kwargs):
        raise AssertionError("unpriced donor must be refused before tensor decode")

    monkeypatch.setattr(cold_module, "decode_payload", forbidden)
    restore_identity = {key: value for key, value in identity.items()
                        if key not in {"model_path", "mtp_enabled"}}
    assert bank.restore(rt, prompt, ssd_exact_plan=plan, **restore_identity) is None


@pytest.mark.parametrize("disabled", [
    "MTPLX_SESSION_NEAR_PREFIX_RESTORE", "MTPLX_SESSION_BLOCK_PREFIX_RESTORE",
])
def test_disabled_near_lane_cannot_shadow_ssd_exact(tmp_path, monkeypatch, disabled):
    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "8")
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "8")
    cold, bank = _spill_a_keep_b(tmp_path)
    try:
        _put(bank, list(PROMPT[:43]) + [99] * 20, "long-near")
        identity = dict(model_path=MODEL, mtp_enabled=True)
        assert bank.restore_plan(PROMPT, **identity)["reuse_tokens"] == 43
        monkeypatch.setenv(disabled, "0")
        plan = bank.restore_plan(PROMPT, include_cold=True, **identity)
        assert (plan["mode"], plan["reuse_tokens"]) == ("ssd_exact", 41)
    finally:
        cold.close()


def test_lazy_ram_checkpoint_reach_does_not_shadow_ssd_exact(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "8")
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "8")
    cold, bank = _spill_a_keep_b(tmp_path)
    try:
        entry = _put(bank, list(PROMPT[:43]) + [99] * 20, "lazy-near")
        entry.has_recurrent = True
        entry.gdn_boundaries = []

        def forbidden(*args, **kwargs):
            raise AssertionError("selection must not hydrate checkpoint tensors")

        forbidden.positions = (16,)
        entry.gdn_boundary_loader = forbidden
        identity = dict(model_path=MODEL, mtp_enabled=True)
        assert bank.restore_plan(PROMPT, **identity)["reuse_tokens"] == 43
        plan = bank.restore_plan(PROMPT, include_cold=True, **identity)
        assert (plan["mode"], plan["reuse_tokens"]) == ("ssd_exact", 41)
    finally:
        cold.close()
