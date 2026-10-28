"""Native tool leases carry explicit provenance through transfer and rollback."""

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from mtplx.cache_state import snapshot_cache, snapshot_untrimmable_cache
from mtplx.one_copy import lease_return
from mtplx.session_bank import SessionBank


TOKENS = (1, 2, 3)
EXTENDED = (*TOKENS, 4)


class _Runtime:
    model_path = Path("models/native-tool-test")
    mtp_enabled = True

    def make_cache(self):
        return [ArraysCache(2), KVCache()]

    def make_mtp_cache(self):
        return [KVCache()]


RUNTIME = _Runtime()


def _append(kv, rows=1, value=1):
    kv.update_and_fetch(
        mx.full((1, 1, rows, 2), value, dtype=mx.float16),
        mx.full((1, 1, rows, 2), value + 1, dtype=mx.float16),
    )
    mx.eval(kv.keys, kv.values)


def _cache():
    cache = RUNTIME.make_cache()
    cache[0][0] = mx.full((1, 2, 3), 7, dtype=mx.float32)
    cache[0][1] = mx.full((1, 2, 2), 8, dtype=mx.float32)
    _append(cache[1], len(TOKENS))
    mx.eval(cache[0].state)
    return cache


def _mtp():
    cache = RUNTIME.make_mtp_cache()
    _append(cache[0], len(TOKENS) - 1)
    return cache


def _put(bank, *, cache=None, tokens=TOKENS, native_owned=False, logits=None, hidden=None, **kwargs):
    values = dict(
        keep_live_ref=True, prefer_live_ref=True, session_id="tool",
        snapshot_epoch=1,
    )
    values.update(kwargs)
    entry = bank.put(
        runtime=RUNTIME, token_ids=tokens,
        cache=_cache() if cache is None else cache,
        logits=logits, hidden=hidden, native_owned=native_owned, **values,
    )
    assert entry is not None
    return entry


def _owned(bank, **kwargs):
    bank.hold_session_persistence("tool")
    return _put(bank, native_owned=True, **kwargs)


def _can_extend(bank, entry, *, session_id="tool", tokens=EXTENDED):
    return bank.native_lease_can_extend(
        entry, session_id=session_id, token_ids=tokens
    )


@pytest.fixture(autouse=True)
def _native_enabled(monkeypatch):
    monkeypatch.setenv("MTPLX_ONE_COPY", "1")
    monkeypatch.setenv("MTPLX_SESSION_LAZY_SNAPSHOT", "1")
    monkeypatch.setenv("MTPLX_SESSION_SNAPSHOT_SETTLE", "0")


def test_fresh_held_native_grant_has_no_attention_snapshot():
    bank = SessionBank()
    cache, mtp = _cache(), _mtp()
    assert not bank.native_live_ref_enabled(cache, session_id="tool")
    bank.hold_session_persistence("tool")
    assert bank.native_live_ref_enabled(cache, session_id="tool")
    assert not bank.native_live_ref_enabled([KVCache()], session_id="tool")

    # native_owned is the caller's trusted fresh-cold-cache attestation.
    entry = _put(bank, cache=cache, mtp_history_cache_ref=mtp, native_owned=True)

    assert entry.live_ref_only and entry.native_exclusive
    assert entry.cache_ref is cache and entry.mtp_history_cache_ref is mtp
    assert all(state is None for state in entry.cache_snapshot.states)
    assert entry.mtp_history_snapshot is None
    assert all(layer._mtplx_native_owned for layer in [*cache, *mtp])
    assert _can_extend(bank, entry)


@pytest.mark.parametrize(
    "held,prefer,native_owned,live",
    [(True, True, False, True), (False, True, True, True), (False, False, False, False)],
)
def test_default_and_legacy_publications_do_not_gain_provenance(
    held, prefer, native_owned, live
):
    bank = SessionBank()
    if held:
        bank.hold_session_persistence("tool")
    entry = _put(bank, prefer_live_ref=prefer, native_owned=native_owned)

    assert entry.live_ref_only is live
    assert not entry.native_exclusive
    assert not _can_extend(bank, entry)
    if not live:
        assert any(state is not None for state in entry.cache_snapshot.states)


@pytest.mark.parametrize(
    "session_id,tokens",
    [(None, EXTENDED), ("foreign", EXTENDED), ("tool", TOKENS),
     ("tool", TOKENS[:-1]), ("tool", (1, 2, 9, 4))],
)
def test_proof_requires_named_owner_and_strict_monotonic_extension(session_id, tokens):
    bank = SessionBank()
    entry = _owned(bank)
    assert not _can_extend(bank, entry, session_id=session_id, tokens=tokens)


@pytest.mark.parametrize("advance_mtp", [False, True])
def test_advanced_live_offsets_refuse_copy_free_proof(advance_mtp):
    bank = SessionBank()
    entry = _owned(bank, mtp_history_cache_ref=_mtp())
    target = entry.mtp_history_cache_ref[0] if advance_mtp else entry.cache_ref[1]
    _append(target)
    assert entry.native_exclusive
    assert not _can_extend(bank, entry)


def test_full_mtp_snapshot_prevents_fresh_grant():
    bank = SessionBank()
    entry = _owned(bank, mtp_history_snapshot=snapshot_cache(_mtp()))
    assert entry.live_ref_only and entry.mtp_history_snapshot is not None
    assert not entry.native_exclusive
    assert not _can_extend(bank, entry)


def test_native_publication_rebinds_current_recurrent_state_without_rewinding_anchor():
    bank = SessionBank()
    cache = _cache()
    anchor = snapshot_untrimmable_cache(cache)
    mx.eval(anchor.states)
    captured = mx.full((8, 1, 2, 3), 17, dtype=mx.float32)[-1]
    mx.eval(captured)
    cache[0][0] = captured
    entry = _owned(bank, cache=cache, gdn_boundaries=[(2, anchor, None)])
    assert entry.cache_ref is cache
    assert cache[0][0] is not captured
    assert mx.all(cache[0][0] == 17).item()
    assert mx.all(entry.gdn_boundaries[0][1].states[0][0] == 7).item()


def test_pending_persistence_and_second_session_snapshot_block_proof():
    bank = SessionBank()
    entry = _owned(bank)
    jobs = []
    bank.cold_enqueue_dispatch = jobs.append
    bank._dispatch_persistence(entry, lambda: None)
    assert not _can_extend(bank, entry)
    jobs.pop()()
    assert _can_extend(bank, entry)

    _put(bank, tokens=(9, 8, 7), keep_live_ref=False)
    assert not _can_extend(bank, entry)


@pytest.mark.parametrize("holder", ["snapshot", "reference"])
def test_other_mtp_only_holder_blocks_proof(holder):
    bank = SessionBank()
    entry = _owned(bank, mtp_history_cache_ref=_mtp())
    other = replace(
        entry, token_ids=(9, 8, 7), cache_ref=None,
        mtp_history_snapshot=(snapshot_cache(entry.mtp_history_cache_ref) if holder == "snapshot" else None),
        mtp_history_cache_ref=(entry.mtp_history_cache_ref if holder == "reference" else None),
    )
    bank._entries[other.token_ids] = other
    assert not _can_extend(bank, entry)


def test_republication_inherits_only_owned_containers():
    bank = SessionBank()
    cache = _cache()
    first = _owned(bank, cache=cache)
    inherited = _put(bank, cache=cache, snapshot_epoch=2)
    assert inherited is not first and inherited.native_exclusive
    assert not _can_extend(bank, first)
    assert _can_extend(bank, inherited)

    unrelated = _put(bank, cache=_cache(), snapshot_epoch=3)
    assert not unrelated.native_exclusive
    assert not _can_extend(bank, unrelated)


@pytest.mark.parametrize("mode,session_id", [("clone", "tool"), ("reference", "foreign")])
def test_clone_borrower_preserves_owner_and_cannot_mutate_its_state(mode, session_id):
    bank = SessionBank()
    cache, mtp = _cache(), _mtp()
    entry = _owned(bank, cache=cache, mtp_history_cache_ref=mtp)
    borrower = bank.restore(RUNTIME, EXTENDED, mode=mode, session_id=session_id)

    assert borrower is not None and borrower.restore_mode == "clone"
    assert not borrower.native_ownership_proven
    assert entry.cache_ref is cache and entry.mtp_history_cache_ref is mtp
    assert borrower.cache[1].keys is not cache[1].keys
    assert borrower.mtp_history_cache[0].keys is not mtp[0].keys
    borrower.cache[1].keys[..., 0, :] = -9
    borrower.cache[0][0] = borrower.cache[0][0] - 16
    borrower.mtp_history_cache[0].keys[..., 0, :] = -9
    mx.eval(borrower.cache[1].keys, borrower.cache[0].state, borrower.mtp_history_cache[0].keys)
    assert mx.all(cache[1].keys[..., 0, :] == 1).item()
    assert mx.all(cache[0][0] == 7).item()
    assert mx.all(mtp[0].keys[..., 0, :] == 1).item()
    cache[1].keys[..., 1, :] = 6
    mx.eval(cache[1].keys)
    assert mx.all(borrower.cache[1].keys[..., 1, :] == 1).item()
    assert entry.native_exclusive and _can_extend(bank, entry)


def test_exact_reference_restore_does_not_claim_monotonic_extension_proof():
    bank = SessionBank()
    _owned(bank)
    restored = bank.restore(RUNTIME, TOKENS, mode="reference", session_id="tool")
    assert restored is not None and restored.restore_mode == "reference_lease"
    assert not restored.native_ownership_proven


def test_reference_transfer_and_aborted_prefill_restore_recurrent_anchor():
    bank = SessionBank()
    cache, mtp = _cache(), _mtp()
    entry = _owned(bank, cache=cache, mtp_history_cache_ref=mtp)
    restored = bank.restore(RUNTIME, EXTENDED, mode="reference", session_id="tool")

    assert restored is not None and restored.native_ownership_proven
    assert restored.restore_mode == "reference_lease"
    assert restored.cache is cache and restored.mtp_history_cache is mtp
    assert entry.cache_ref is None and entry.mtp_history_cache_ref is None
    recovery = lease_return(
        bank, RUNTIME, restore_mode=restored.restore_mode, cache=cache,
        mtp_history_cache=mtp, token_ids=TOKENS, hidden=None, source=entry,
    )
    assert recovery is not None and recovery.anchor[1].states[1] is None
    _append(cache[1], rows=2, value=9)
    _append(mtp[0], rows=2, value=9)
    cache[0][0] = cache[0][0] + 20
    cache[0][1] = cache[0][1] + 20
    mx.eval(cache[0].state)
    returned = recovery.give_back()

    assert returned is not None and returned.native_exclusive and returned.extension_only
    assert returned.cache_ref is cache and returned.mtp_history_cache_ref is mtp
    assert not _can_extend(bank, returned)
    assert mx.all(recovery.anchor[1].states[0][0] == 7).item()
    retried = bank.restore(RUNTIME, EXTENDED, mode="reference", session_id="tool")
    assert retried is not None and not retried.native_ownership_proven
    assert retried.cache is cache and retried.mtp_history_cache is mtp
    assert cache[1].offset == len(TOKENS) and mtp[0].offset == len(TOKENS) - 1
    assert mx.all(cache[0][0] == 7).item()
    assert mx.all(cache[0][1] == 8).item()
    assert mx.all(cache[1].keys[..., :len(TOKENS), :] == 1).item()


@pytest.mark.parametrize("fail", [False, True])
def test_spill_revokes_provenance_without_rearming_on_republication(fail):
    bank = SessionBank()
    cache, mtp = _cache(), _mtp()
    entry = _owned(bank, cache=cache, mtp_history_cache_ref=mtp)
    calls = []

    class Tier:
        def spill_entry(self, view, **kwargs):
            assert not entry.native_exclusive
            assert all(not layer._mtplx_native_owned for layer in [*cache, *mtp])
            calls.append(view.token_ids)
            if fail:
                raise RuntimeError("test spill failure")
            return True

    bank.cold_tier = Tier()
    jobs = []
    bank.cold_enqueue_dispatch = jobs.append
    assert bank.release_session_persistence("tool")
    assert len(jobs) == 1
    assert jobs.pop()() is (not fail)
    assert calls == [TOKENS]
    assert not entry.native_exclusive
    bank.hold_session_persistence("tool")
    republished = _put(bank, cache=cache, mtp_history_cache_ref=mtp, snapshot_epoch=2)
    assert not republished.native_exclusive
    assert not _can_extend(bank, republished)


_STABLE_IDENTITY = dict(
    hidden_variant="native-test", template_hash="template-test",
    mtp_history_policy="committed", draft_head_identity="draft-test",
    policy_fingerprint="policy-test",
)
_STABLE_INCOMING = (1, 2, 9, 4)


def _stable_lease(bank, *, with_mtp=True, refusal=None):
    cache, mtp = _cache(), _mtp() if with_mtp else None
    stable_hidden = mx.full((1, 1, 2), 11, dtype=mx.float32)
    stable = snapshot_untrimmable_cache(cache)
    mx.eval(stable.states, stable_hidden)
    cache[0][0] = cache[0][0] + 10
    cache[0][1] = cache[0][1] + 10
    current = snapshot_untrimmable_cache(cache)
    current_hidden = mx.full((1, 1, 2), 21, dtype=mx.float32)
    boundaries = [(3, current, current_hidden)]
    if refusal != "missing_boundary":
        boundaries.insert(0, (2, stable, None if refusal == "missing_hidden" else stable_hidden))
    if refusal == "windowed_mtp":
        mtp[0].trim(1)
    bank.hold_session_persistence("tool")
    entry = _put(
        bank, cache=cache, mtp_history_cache_ref=mtp,
        native_owned=refusal != "unproven", native_stable_prefix_len=2,
        gdn_boundaries=boundaries, explicit_cache_boundary=3, snapshot_epoch=7,
        logits=mx.ones((1, 4)), hidden=current_hidden,
        extra_state={"position": 3} if refusal == "extra_state" else None,
        **_STABLE_IDENTITY,
    )
    if refusal == "current_advance":
        _append(cache[1])
        _append(mtp[0])
    return entry, cache, mtp


@pytest.mark.parametrize("with_mtp", [False, True])
def test_prepare_stable_continuation_rekeys_then_restore_rewinds_without_copy(with_mtp):
    bank = SessionBank()
    entry, cache, mtp = _stable_lease(bank, with_mtp=with_mtp)
    trunk_buffers = (cache[1].keys, cache[1].values)
    mtp_buffers = (mtp[0].keys, mtp[0].values) if mtp else None
    current_recurrent = cache[0][0]
    original_hash = entry.token_hash

    prepared = bank.prepare_native_continuation(session_id="tool", token_ids=_STABLE_INCOMING)

    assert prepared is entry and bank._entries == {TOKENS[:2]: entry}
    assert entry.token_hash != original_hash
    assert entry.token_ids == TOKENS[:2] and entry.snapshot_epoch == 2
    assert entry.logits is None and entry.extension_only
    assert entry.explicit_cache_boundary is None
    assert [record[0] for record in entry.gdn_boundaries] == [2]
    assert bank._persistence_deferred == {"tool": (TOKENS[:2], 2)}
    assert all(getattr(entry, key) == value for key, value in _STABLE_IDENTITY.items())
    assert cache[1].offset == 3 and cache[0][0] is current_recurrent
    assert cache[1].keys is trunk_buffers[0] and cache[1].values is trunk_buffers[1]
    assert mx.all(cache[0][0] == 17).item()
    if mtp:
        assert entry.mtp_snapshot_epoch == 2 and entry.lease_mtp_offset == 1
        assert mtp[0].offset == 2
        assert mtp[0].keys is mtp_buffers[0] and mtp[0].values is mtp_buffers[1]
    assert _can_extend(bank, entry, tokens=_STABLE_INCOMING)
    # The anchor has no end-of-prefix logits, so it may only seed an extension.
    assert bank.restore(RUNTIME, TOKENS[:2], mode="reference", session_id="tool") is None
    assert entry.cache_ref is cache

    restored = bank.restore(RUNTIME, _STABLE_INCOMING, mode="reference", session_id="tool")

    assert restored is not None and restored.native_ownership_proven
    assert restored.cache is cache and restored.mtp_history_cache is mtp
    assert entry._native_preparation is None
    assert cache[1].offset == 2 and cache[1].keys is trunk_buffers[0]
    assert mx.all(cache[0][0] == 7).item() and mx.all(cache[0][1] == 8).item()
    assert mx.all(restored.hidden == 11).item()
    if mtp:
        assert mtp[0].offset == 1 and mtp[0].keys is mtp_buffers[0]

    # A subsequent tool answer inherits the retained stable anchor/provenance.
    _append(cache[1], rows=2, value=9)
    if mtp:
        _append(mtp[0], rows=2, value=9)
    cache[0][0] = cache[0][0] + 20
    next_entry = _put(
        bank, cache=cache, tokens=_STABLE_INCOMING, mtp_history_cache_ref=mtp,
        snapshot_epoch=9, **_STABLE_IDENTITY,
    )
    assert next_entry.native_exclusive and next_entry.native_stable_prefix_len == 2
    assert next_entry.recurrent_boundary_at_or_below(2)[0] == 2
    assert bank.prepare_native_continuation(session_id="tool", token_ids=(1, 2, 8, 5)) is next_entry
    assert bank._persistence_deferred == {"tool": (TOKENS[:2], 2)}
    assert _can_extend(bank, next_entry, tokens=(1, 2, 8, 5))


def test_prepare_strict_full_raw_continuation_preserves_entry_and_metadata():
    bank = SessionBank()
    entry, cache, mtp = _stable_lease(bank)
    logits, hidden, token_hash = entry.logits, entry.hidden, entry.token_hash
    assert bank.prepare_native_continuation(session_id="tool", token_ids=EXTENDED) is entry
    assert bank._entries == {TOKENS: entry}
    assert entry.token_ids == TOKENS and entry.token_hash == token_hash
    assert entry.logits is logits and entry.hidden is hidden and not entry.extension_only
    assert entry.snapshot_epoch == 7 and entry.explicit_cache_boundary == 3
    assert cache[1].offset == 3 and mtp[0].offset == 2
    assert bank._persistence_deferred == {"tool": (TOKENS, 7)}
    assert _can_extend(bank, entry)


@pytest.mark.parametrize(
    "refusal",
    ["different_prefix", "no_suffix", "missing_boundary", "missing_hidden",
     "windowed_mtp", "extra_state", "unproven", "foreign", "current_advance"],
)
def test_prepare_stable_continuation_refuses_without_mutating_source(refusal):
    bank = SessionBank()
    entry, cache, mtp = _stable_lease(bank, refusal=refusal)
    incoming = (8, 2, 9, 4) if refusal == "different_prefix" else _STABLE_INCOMING
    if refusal == "no_suffix":
        incoming = TOKENS[:2]
    offsets = (cache[1].offset, mtp[0].offset)
    buffers = (cache[1].keys, mtp[0].keys, cache[0][0])
    logits, hidden, token_hash = entry.logits, entry.hidden, entry.token_hash

    assert bank.prepare_native_continuation(
        session_id="foreign" if refusal == "foreign" else "tool", token_ids=incoming
    ) is None

    assert bank._entries == {TOKENS: entry}
    assert entry.token_hash == token_hash and entry.snapshot_epoch == 7
    assert entry.logits is logits and entry.hidden is hidden and not entry.extension_only
    assert (cache[1].offset, mtp[0].offset) == offsets
    assert cache[1].keys is buffers[0] and mtp[0].keys is buffers[1]
    assert cache[0][0] is buffers[2]
    assert bank._persistence_deferred == {"tool": (TOKENS, 7)}


def test_unconsumed_preparation_release_restores_raw_frontier_before_spill():
    bank = SessionBank()
    entry, cache, mtp = _stable_lease(bank)
    logits, hidden, boundaries = entry.logits, entry.hidden, entry.gdn_boundaries
    aux_bytes = entry.lease_aux_nbytes
    jobs, spilled = [], []

    class Tier:
        def spill_entry(self, view, **kwargs):
            spilled.append((view.token_ids, view.snapshot_epoch))
            assert view.cache_snapshot.states[1][0].shape[2] == len(TOKENS)
            assert view.mtp_history_snapshot.states[0][0].shape[2] == len(TOKENS) - 1
            assert mx.all(view.cache_snapshot.states[0][0] == 17).item()
            return True

    bank.cold_tier = Tier()
    bank.cold_enqueue_dispatch = jobs.append
    assert bank.prepare_native_continuation(session_id="tool", token_ids=_STABLE_INCOMING) is entry
    assert entry.lease_aux_nbytes == aux_bytes
    assert entry._native_preparation is not None
    assert "cache_ref" not in entry._native_preparation
    assert "mtp_history_cache_ref" not in entry._native_preparation

    assert bank.release_session_persistence("tool")

    assert bank._entries == {TOKENS: entry}
    assert entry._native_preparation is None
    assert entry.logits is logits and entry.hidden is hidden
    assert entry.gdn_boundaries is boundaries and not entry.extension_only
    assert entry.snapshot_epoch == 7 and entry.mtp_snapshot_epoch == 7
    assert entry.explicit_cache_boundary == 3 and entry.lease_aux_nbytes == aux_bytes
    assert entry.lease_kv_offset == 3 and entry.lease_mtp_offset == 2
    assert cache[1].offset == 3 and mtp[0].offset == 2
    assert bank._persistence_deferred == {} and len(jobs) == 1
    assert jobs.pop()() is True
    assert spilled == [(TOKENS, 7)]
    assert entry.cold_encode_completed_at is not None


def test_clear_discards_unconsumed_preparation_without_resurrection():
    bank = SessionBank()
    entry, _, _ = _stable_lease(bank)
    aux_bytes = entry.lease_aux_nbytes
    assert bank.prepare_native_continuation(session_id="tool", token_ids=_STABLE_INCOMING) is entry
    assert entry._native_preparation is not None

    assert bank.clear(session_id="tool") == 1

    assert entry._native_preparation is None
    assert entry.cache_ref is None and entry.mtp_history_cache_ref is None
    assert entry.lease_aux_nbytes < aux_bytes
    assert not bank.release_session_persistence("tool")
    assert bank.prepare_native_continuation(session_id="tool", token_ids=EXTENDED) is None
    assert bank._entries == {} and bank._persistence_deferred == {}
