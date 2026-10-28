"""Issue #247: restores must never install the bank entry's own state objects.

druide67's report: a byte-identical temperature-0 call was answered according
to a *different* request's system prompt after an interleaved near-prefix
request. Root cause: lazy (kvcache-v2) snapshots restored with
``clone_states=False`` installed the entry's stored array objects directly
into the borrower's cache. The borrower's suffix prefill then wrote into those
very objects (setitem mutates the installed object; a rebind write may donate
its buffer because the lone shared object counts as uniquely referenced), so
the lender's banked span was silently rewritten and every later match served
the poisoned pages.

These tests are the regression pin built from that reproducer's geometry:
commit A -> near-prefix borrow for B -> B's suffix write -> A's entry (and
every later restore from it) must be byte-identical to a cache that was never
borrowed from. They run on real ``mx.array`` state through the mlx-lm-style
setitem write path, the exact container class the report reproduced on.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mtplx.cache_state import restore_cache, snapshot_cache_lazy_hybrid
from mtplx.session_bank import SessionBank
from mtplx.vendored_arrays_cache import FixedArraysCache

HEADS = 2
DIM = 4
CAPACITY = 32
PREFIX = 10
MATCH = 8  # near-prefix borrow point; gap 2 stays under the tiny-gap limit


@pytest.mark.parametrize("slice_backed", [False, True])
def test_live_entry_releases_snapshot_producer_graph(monkeypatch, slice_backed) -> None:
    monkeypatch.setenv("MTPLX_SESSION_LAZY_SNAPSHOT", "1")
    source = mx.ones((1, 64, 16, 128, 128) if slice_backed else (4096, 4096), dtype=mx.float32)
    mx.eval(source)
    source_bytes = source.nbytes
    recurrent = FixedArraysCache(1)
    recurrent[0] = source[:, -1] if slice_backed else source.sum(axis=0)
    del source
    before = mx.get_active_memory()
    bank = SessionBank()
    entry = bank.put(
        runtime=RuntimeWithMxCaches(), token_ids=[1, 2], cache=[recurrent],
        logits=None, hidden=None, keep_live_ref=True,
    )
    assert entry is not None and entry.cache_ref is not None
    assert mx.get_active_memory() < before - source_bytes // 2
    assert mx.all(recurrent[0] == (1 if slice_backed else 4096)).item()
    recurrent[0][0] = -1
    assert mx.all(entry.cache_snapshot.states[0][0] == (1 if slice_backed else 4096)).item()


class MxKVCache:
    """mlx-lm-style KV container: preallocated buffers, in-place setitem
    appends, trimmed-slice ``state``. The write pattern #247 hinges on."""

    def __init__(self, capacity: int = CAPACITY):
        self.keys = mx.zeros((1, HEADS, capacity, DIM), dtype=mx.float16)
        self.values = mx.zeros((1, HEADS, capacity, DIM), dtype=mx.float16)
        self.offset = 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(int(n), self.offset)
        self.offset -= n
        return n

    def append(self, k: mx.array, v: mx.array) -> None:
        steps = int(k.shape[2])
        if self.offset + steps > int(self.keys.shape[2]):
            raise RuntimeError("test cache capacity exceeded")
        self.keys[..., self.offset : self.offset + steps, :] = k
        self.values[..., self.offset : self.offset + steps, :] = v
        self.offset += steps

    @property
    def state(self):
        if self.offset == int(self.keys.shape[2]):
            return self.keys, self.values
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = int(self.keys.shape[2])

    @property
    def meta_state(self):
        return (str(self.offset),)

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])


class RuntimeWithMxCaches:
    model_path = Path("models/example")
    mtp_enabled = True

    def make_cache(self):
        return [MxKVCache()]

    def make_mtp_cache(self):
        return [MxKVCache()]


def _step(seed: int, steps: int = 1) -> tuple[mx.array, mx.array]:
    k = mx.random.normal((1, HEADS, steps, DIM), key=mx.random.key(seed)).astype(
        mx.float16
    )
    v = mx.random.normal(
        (1, HEADS, steps, DIM), key=mx.random.key(seed + 10_000)
    ).astype(mx.float16)
    return k, v


def _fill(cache: MxKVCache, tokens: int, *, base_seed: int = 0) -> None:
    for i in range(tokens):
        cache.append(*_step(base_seed + i))


def _keys_prefix(cache: MxKVCache, tokens: int) -> mx.array:
    return cache.keys[..., :tokens, :]


def test_restore_cache_lazy_install_is_immune_to_borrower_writes() -> None:
    """Primitive pin: clone_states=False must not hand out the stored objects."""
    src = MxKVCache()
    control = MxKVCache()
    _fill(src, PREFIX)
    _fill(control, PREFIX)

    snapshot = snapshot_cache_lazy_hybrid([src])

    borrower = MxKVCache()
    restore_cache([borrower], snapshot, clone_states=False)
    assert borrower.offset == PREFIX

    # Near-prefix borrow: trim back, then setitem-write a divergent suffix
    # inside the span the snapshot still covers.
    borrower.trim(PREFIX - MATCH)
    borrower.append(*_step(500, steps=2))
    mx.eval(borrower.keys, borrower.values)

    snap_keys, snap_values = snapshot.states[0]
    assert mx.array_equal(
        snap_keys, _keys_prefix(control, PREFIX)
    ).item(), "borrower setitem write reached the snapshot's stored keys (#247)"
    assert mx.array_equal(
        snap_values, control.values[..., :PREFIX, :]
    ).item(), "borrower setitem write reached the snapshot's stored values (#247)"

    # A second restore from the same snapshot must serve pristine pages.
    second = MxKVCache()
    restore_cache([second], snapshot, clone_states=False)
    assert mx.array_equal(
        _keys_prefix(second, PREFIX), _keys_prefix(control, PREFIX)
    ).item(), "restore after a borrower write served poisoned pages (#247)"


def test_bank_near_prefix_borrow_does_not_poison_the_lender_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full bank flow from the reproducer: put A, near-prefix borrow, write,
    then exact restore of A — byte-identical to a never-borrowed cache."""
    monkeypatch.setenv("MTPLX_SESSION_LAZY_SNAPSHOT", "1")
    runtime = RuntimeWithMxCaches()
    bank = SessionBank()
    tokens = list(range(PREFIX))

    src = MxKVCache()
    control = MxKVCache()
    _fill(src, PREFIX)
    _fill(control, PREFIX)

    entry = bank.put(
        runtime=runtime,
        token_ids=tokens,
        cache=[src],
        logits=None,
        hidden=None,
        session_id="lender",
    )
    assert entry is not None
    assert entry.lazy_kv is True, "test requires the lazy-snapshot geometry"

    # Interleaved request B: borrows A's entry at a near-prefix boundary and
    # prefills its own divergent suffix. B stores nothing — in the report B's
    # uncovered suffix was under the store minimum, and A was poisoned anyway.
    borrowed = bank.restore_entry_prefix_cache(runtime, entry, MATCH)
    assert borrowed is not None
    borrower_cache = borrowed[0]
    borrower_cache[0].append(*_step(700, steps=2))
    mx.eval(borrower_cache[0].keys, borrower_cache[0].values)

    # The entry's stored span must be untouched by B's writes...
    snap_keys, _ = entry.cache_snapshot.states[0]
    assert mx.array_equal(
        snap_keys, _keys_prefix(control, PREFIX)
    ).item(), "near-prefix borrower poisoned the lender's banked span (#247)"

    # ...and A's own exact restore must serve the original bytes.
    restored = bank.restore(runtime, tokens, session_id="lender")
    assert restored is not None
    assert mx.array_equal(
        _keys_prefix(restored.cache[0], PREFIX), _keys_prefix(control, PREFIX)
    ).item(), "exact restore after an interleaved borrow served poisoned pages (#247)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize("keep_live_ref", [False, True])
@pytest.mark.parametrize("snapshot_kind", ["eager", "lazy", "settled"])
def test_native_rewind_discards_tail_without_changing_donor(
    monkeypatch, keep_live_ref, snapshot_kind
):
    from types import SimpleNamespace
    from mlx_lm.models.cache import KVCache
    from mtplx.cache_state import snapshot_cache, snapshot_untrimmable_cache

    monkeypatch.setenv("MTPLX_SESSION_LAZY_SNAPSHOT", str(int(snapshot_kind != "eager")))
    prefix, boundary = 2048, 512

    def kv(length):
        cache = KVCache()
        keys = mx.arange(length * 8).reshape(1, 2, length, 4).astype(mx.float32)
        cache.state = keys, keys + 1
        return cache

    recurrent = FixedArraysCache(1)
    recurrent[0] = mx.array([boundary])
    checkpoint = snapshot_untrimmable_cache([kv(boundary), recurrent])
    recurrent[0] = mx.array([prefix])
    trunk, draft = [kv(prefix), recurrent], [kv(prefix - 1)]
    expected_trunk = tuple(x + mx.zeros((), dtype=x.dtype) for x in trunk[0].state)
    expected_draft = tuple(x + mx.zeros((), dtype=x.dtype) for x in draft[0].state)
    mx.eval(expected_trunk, expected_draft)
    runtime = SimpleNamespace(
        model_path=Path("models/native-rewind"), mtp_enabled=True,
        make_cache=lambda: [KVCache(), FixedArraysCache(1)],
        make_mtp_cache=lambda: [KVCache()],
    )
    bank = SessionBank()
    entry = bank.put(
        runtime=runtime, token_ids=list(range(prefix)), cache=trunk,
        logits=None, hidden=None, keep_live_ref=keep_live_ref,
        mtp_history_snapshot=snapshot_cache(draft),
        mtp_history_cache_ref=draft if keep_live_ref else None,
        mtp_history_policy="committed", snapshot_epoch=prefix,
        mtp_snapshot_epoch=prefix, gdn_boundaries=[(boundary, checkpoint, None)],
    )
    assert entry is not None
    if snapshot_kind == "settled":
        entry.cache_snapshot = snapshot_cache(trunk)
        mx.eval(entry.cache_snapshot.states, entry.cache_snapshot.meta_states)
        entry.snapshot_settled_at = 1.0
    compact = snapshot_kind != "eager" and not (keep_live_ref and snapshot_kind == "settled")
    restored = bank.restore_entry_prefix_cache(
        runtime, entry, boundary + 10, mode="reference" if keep_live_ref else "clone"
    )
    assert restored is not None and restored[3] == boundary
    cache, mtp = restored[:2]
    assert cache[1][0].item() == boundary
    assert cache[0].keys.shape[2] == (boundary if compact else prefix)
    assert mtp[0].keys.shape[2] == prefix - 1
    for layer, expected, target in (
        (cache[0], expected_trunk, boundary),
        (mtp[0], expected_draft, boundary - 1),
    ):
        suffix = mx.full((1, 2, 2, 4), -1.0)
        keys, values = layer.update_and_fetch(suffix, suffix)
        mx.eval(keys, values)
        assert mx.array_equal(keys[..., :target, :], expected[0][..., :target, :]).item()
        assert mx.array_equal(values[..., :target, :], expected[1][..., :target, :]).item()
        assert mx.all(keys[..., target:, :] == -1).item()
    for snapshot, expected in ((entry.cache_snapshot, expected_trunk),
                               (entry.mtp_history_snapshot, expected_draft)):
        assert mx.array_equal(snapshot.states[0][0], expected[0]).item()
        assert mx.array_equal(snapshot.states[0][1], expected[1]).item()


def test_native_rewind_keeps_sole_lease_buffer():
    from types import SimpleNamespace
    from mlx_lm.models.cache import KVCache

    layer = KVCache()
    layer.state = (mx.zeros((1, 2, 2048, 4)), mx.ones((1, 2, 2048, 4)))
    runtime = SimpleNamespace(model_path=Path("models/native-lease"), mtp_enabled=False)
    bank = SessionBank(max_bytes=1024, per_session_max_bytes=512)
    entry = bank.put(
        runtime=runtime, token_ids=list(range(2048)), cache=[layer],
        logits=None, hidden=None, keep_live_ref=True, session_id="owner",
        snapshot_epoch=2048, nbytes_override=2048,
    )
    assert entry is not None and entry.live_ref_only
    keys, values = layer.keys, layer.values
    restored = bank.restore_entry_prefix_cache(
        runtime, entry, 512, mode="reference", session_id="owner"
    )
    assert restored is not None and restored[2] == "reference_lease"
    assert layer.offset == 511
    assert layer.keys is keys and layer.values is values
