from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mtplx.cache_state import (
    BlockOwnedKVCache, CacheSnapshot, install_block_owned_attention_kv_cache,
    restore_cache, snapshot_cache, snapshot_cache_lazy_hybrid,
)
from mtplx.graphbank import cache_array_tree
from mtplx.session_bank import SessionBank


def chunk(start, count):
    keys = mx.arange(start, start + count, dtype=mx.float32).reshape(1, 1, count, 1)
    return keys, keys + 1000


def rows(cache):
    keys, values = cache._active_arrays() if isinstance(cache, BlockOwnedKVCache) else cache.state
    return keys.flatten().tolist(), values.flatten().tolist()


@pytest.mark.parametrize("capture", [snapshot_cache, snapshot_cache_lazy_hybrid])
@pytest.mark.parametrize("clone", [False, True])
def test_snapshot_branches_trim_and_append_preserve_donor(capture, clone, monkeypatch):
    source = BlockOwnedKVCache(block_size=4)
    source.update_without_fetch(*chunk(0, 11))
    control = rows(source)
    with monkeypatch.context() as m:
        m.setattr(BlockOwnedKVCache, "_active_arrays", lambda _: pytest.fail("dense snapshot"))
        saved = capture([source])
        branch = BlockOwnedKVCache(block_size=8)
        restore_cache([branch], saved, clone_states=clone, restore_meta_state=False)
        assert branch.block_size == 4
        branch.trim(6)
        assert len(branch.key_blocks) == 2
        branch.update_without_fetch(*chunk(50, 7))
        mx.eval(branch.state, saved.states)
        tree = cache_array_tree([branch])
        assert tree[0][0] is branch.key_blocks
    assert rows(branch) == (list(range(5)) + list(range(50, 57)),
                            list(range(1000, 1005)) + list(range(1050, 1057)))
    assert rows(source) == control
    later = BlockOwnedKVCache(block_size=16)
    restore_cache([later], saved, clone_states=False)
    assert rows(later) == control
    source.trim(2)
    source.update_without_fetch(*chunk(99, 3))
    mx.eval(source.state)
    assert rows(later) == control
    # Verify rollback can cross a block edge, then overwrite rejected rows.
    branch.trim(5)
    branch.update_without_fetch(*chunk(80, 2))
    mx.eval(branch.state)
    assert rows(later) == control


def test_native_and_block_snapshots_restore_both_directions():
    native = KVCache()
    native.update_and_fetch(*chunk(0, 9))
    block = BlockOwnedKVCache(block_size=4)
    restore_cache([block], snapshot_cache_lazy_hybrid([native]), clone_states=False)
    assert rows(block) == rows(native)
    saved = snapshot_cache_lazy_hybrid([block])
    native_again = KVCache()
    restore_cache([native_again], saved, clone_states=False)
    assert rows(native_again) == rows(native)
    native_again.trim(5)
    native_again.update_and_fetch(*chunk(70, 2))
    mx.eval(native_again.state)
    assert rows(block) == rows(native)
    empty = snapshot_cache_lazy_hybrid([BlockOwnedKVCache(block_size=4)])
    restore_cache([native_again], empty, clone_states=False)
    assert native_again.offset == 0 and native_again.keys is None


@pytest.mark.parametrize("bad", [
    {"offset": -1}, {"offset": 12}, {"block_size": 0}, {"block_size": True},
    {"keys": ()}, {"values": (mx.zeros((1, 1, 4, 1)),)},
    {"keys": (mx.zeros((1, 1, 4, 1)), mx.zeros((1, 2, 4, 1)))},
])
def test_malformed_block_state_rejected_without_mutation(bad):
    cache = BlockOwnedKVCache(block_size=4)
    cache.update_without_fetch(*chunk(0, 7))
    before = rows(cache)
    with pytest.raises(ValueError):
        cache.state = {**cache.state, **bad}
    assert cache.offset == 7 and rows(cache) == before


def test_partial_last_block_grows_and_reconfiguration_keeps_geometry():
    cache = BlockOwnedKVCache(block_size=4)
    keys, values = chunk(0, 3)
    cache.state = {"kind": "mtplx_block_kv_v1", "offset": 3,
                   "block_size": 4, "keys": (keys,), "values": (values,)}
    install_block_owned_attention_kv_cache([cache], block_size=8)
    assert cache.block_size == 4
    cache.update_without_fetch(*chunk(3, 3))
    assert rows(cache) == (list(range(6)), list(range(1000, 1006)))


@pytest.mark.parametrize("keep_live_ref", [False, True])
def test_bank_branch_restore_does_not_flatten_or_mutate_donor(monkeypatch, keep_live_ref):
    monkeypatch.setenv("MTPLX_SESSION_LAZY_SNAPSHOT", "1")
    runtime = SimpleNamespace(model_path=Path("models/block-test"), mtp_enabled=False,
                              make_cache=lambda: [BlockOwnedKVCache(block_size=4)])
    bank = SessionBank()
    source = runtime.make_cache()[0]
    source.update_without_fetch(*chunk(0, 11))
    with monkeypatch.context() as m:
        m.setattr(BlockOwnedKVCache, "_active_arrays", lambda _: pytest.fail("dense bank path"))
        entry = bank.put(runtime=runtime, token_ids=list(range(11)), cache=[source],
                         logits=None, hidden=None, keep_live_ref=keep_live_ref,
                         session_id="branch")
        assert entry is not None and entry.lazy_kv
        restored = bank.restore_entry_prefix_cache(runtime, entry, 5, mode="clone")
        assert restored is not None
        restored[0][0].update_without_fetch(*chunk(60, 3))
        mx.eval(restored[0][0].state)
        later = runtime.make_cache()
        restore_cache(later, entry.cache_snapshot, clone_states=False)
    assert rows(later[0]) == (list(range(11)), list(range(1000, 1011)))
