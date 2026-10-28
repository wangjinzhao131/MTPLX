from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mtplx.cache_state import (
    BlockOwnedKVCache, detach_attention_cache_state, restore_cache,
    snapshot_cache_lazy_hybrid,
)
from mtplx.graphbank import (
    CompiledVerifyBank, SpecDecodeGraphBank, promote_kv_cache_offsets,
)
from mtplx.runtime import MTPLXRuntime


def block_cache():
    cache = BlockOwnedKVCache(block_size=4)
    keys = mx.arange(7, dtype=mx.float32).reshape(1, 1, 7, 1)
    cache.update_without_fetch(keys, keys + 100)
    return cache


def forbid_dense(monkeypatch):
    monkeypatch.setattr(
        BlockOwnedKVCache, "_active_arrays",
        lambda _: pytest.fail("block cache was flattened"),
    )


def test_tensor_offset_promotion_declines_blocks_without_dense_read(monkeypatch):
    entry = block_cache()
    cache = [entry]
    forbid_dense(monkeypatch)
    assert promote_kv_cache_offsets(cache, reserve_tokens=2) == (
        0, {"block_kv_cache": 1},
    )
    assert cache[0] is entry and entry.offset == 7


@pytest.mark.parametrize("allow_python", [False, True])
@pytest.mark.parametrize("promote", [False, True])
def test_spec_graphbank_keeps_blocks_eager(monkeypatch, allow_python, promote):
    entry = block_cache()
    runtime = SimpleNamespace(forward_ar=lambda *args, **kwargs: "eager")
    bank = SpecDecodeGraphBank(
        runtime, allow_python_cache_capture=allow_python,
        promote_tensor_offsets=promote,
    )
    forbid_dense(monkeypatch)
    monkeypatch.setattr(mx, "compile", lambda *a, **kw: pytest.fail("block compile"))
    assert bank.forward_ar(mx.array([[1, 2]]), cache=[entry]) == "eager"
    assert bank.stats.fallback_reasons == {"block_kv_cache": 1}
    assert bank.stats.compiled_calls == 0


def test_compiled_verify_keeps_blocks_eager_without_owned_env(monkeypatch):
    monkeypatch.delenv("MTPLX_OWNED_ATTN_KV", raising=False)
    monkeypatch.delenv("MTPLX_OWNED_RECURRENT_STATE", raising=False)
    entry = block_cache()
    runtime = SimpleNamespace(
        forward_ar_capture=lambda *args, **kwargs: ("logits", "hidden", {}),
    )
    bank = CompiledVerifyBank(runtime, parity=True)
    forbid_dense(monkeypatch)
    assert bank.forward_ar_capture(mx.array([[1, 2]]), cache=[entry]) == (
        "logits", "hidden", {},
    )
    assert bank.stats["fallback_reasons"] == {"promotion_failure:block_kv_cache": 1}
    assert bank.stats["compiled_calls"] == 0


@pytest.mark.parametrize("native_first", [False, True])
def test_runtime_compiled_ar_declines_any_block_layer(monkeypatch, native_first):
    monkeypatch.setenv("MTPLX_COMPILE_AR_FORWARD", "1")
    entry = block_cache()
    native = KVCache()
    native.update_and_fetch(mx.zeros((1, 1, 1, 1)), mx.zeros((1, 1, 1, 1)))
    cache = [native, entry] if native_first else [entry]
    forbid_dense(monkeypatch)
    assert MTPLXRuntime._compiled_ar_forward(SimpleNamespace(), cache) is None


@pytest.mark.parametrize("mode", ["eval_only", "contiguous_eval", "metal_copy_leaf"])
def test_detachment_preserves_blocks_offset_and_snapshot(monkeypatch, mode):
    entry = block_cache()
    saved = snapshot_cache_lazy_hybrid([entry])
    block_ids = [id(block) for block in entry.key_blocks + entry.value_blocks]
    key_blocks, value_blocks = entry.key_blocks, entry.value_blocks
    with monkeypatch.context() as m:
        forbid_dense(m)
        stats = detach_attention_cache_state([entry], mode=mode)
        assert stats == {"entries": 1, "arrays": 4, "bytes": 64}
        assert entry.key_blocks is key_blocks and entry.value_blocks is value_blocks
        assert entry.offset == 7 and entry.block_size == 4
        if mode == "eval_only":
            assert block_ids == [id(block) for block in key_blocks + value_blocks]
        entry.trim(2)
        tail = mx.full((1, 1, 2, 1), 99, dtype=mx.float32)
        entry.update_without_fetch(tail, tail + 100)
        later = BlockOwnedKVCache(block_size=4)
        restore_cache([later], saved, clone_states=False)
    assert entry._active_arrays()[0].flatten().tolist() == list(range(5)) + [99, 99]
    assert later._active_arrays()[0].flatten().tolist() == list(range(7))


def test_detachment_empty_block_cache_is_noop(monkeypatch):
    forbid_dense(monkeypatch)
    assert detach_attention_cache_state([BlockOwnedKVCache()], mode="eval_only") == {
        "entries": 0, "arrays": 0, "bytes": 0,
    }
