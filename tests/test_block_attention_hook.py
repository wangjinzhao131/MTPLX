"""The gated Qwen hook must consume physical KV blocks without fetching them."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_next import Qwen3NextAttention

from mtplx.attention_context import (
    attention_phase,
    kv_attention_failure_line,
    kv_attention_request_scope,
)
from mtplx.attention_split import configure_split_full_attention
from mtplx.cache_state import BlockOwnedKVCache


_native_attention = Qwen3NextAttention.__call__


@pytest.fixture
def configure(monkeypatch):
    for name in (
        "MTPLX_SPLIT_FULL_ATTN",
        "MTPLX_SDPA_2PASS",
        "MTPLX_GQA_PACKED_SDPA",
        "MTPLX_VLLM_METAL_PAGED_ATTN",
        "MTPLX_ROUTE_DEBUG",
        "MTPLX_KV_ATTENTION_TRACE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_BLOCKWISE_ATTN", "1")
    monkeypatch.setenv("MTPLX_BLOCKWISE_ATTN_THRESHOLD", "0")

    def make(dtype=mx.float32, *, query_heads=4, kv_heads=2, head_dim=64):
        # Isolate the class-level hook from other tests and loaded models.
        class Attention(Qwen3NextAttention):
            _mtplx_split_full_attention_installed = False
            __call__ = _native_attention

        mx.random.seed(17)
        attn = Attention(SimpleNamespace(
            hidden_size=32, num_attention_heads=query_heads, num_key_value_heads=kv_heads,
            head_dim=head_dim, attention_bias=False, rms_norm_eps=1e-6,
            partial_rotary_factor=1.0, rope_theta=10000.0, rope_scaling=None,
            max_position_embeddings=2048,
        ))
        attn.set_dtype(dtype)
        mx.eval(attn.parameters())
        model = SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
        stats = configure_split_full_attention(model)
        assert stats["enabled"] is True
        assert stats["blockwise_enabled"] is True
        assert stats["split_full_attn_enabled"] is False
        assert attn._mtplx_split_full_attention_enabled is True
        return attn

    return make


def _assert_close(actual, expected, dtype=mx.float32):
    # Block softmax accumulates in fp32; native SDPA uses other reductions.
    tolerance = {mx.float32: 2e-5, mx.float16: 1e-3, mx.bfloat16: 8e-3}[dtype]
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
        rtol=tolerance, atol=tolerance,
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("other_routes_and_diagnostics", [False, True])
def test_cold_warm_decode_verify_never_fetch_blocks(
    configure, monkeypatch, dtype, other_routes_and_diagnostics,
):
    if other_routes_and_diagnostics:
        monkeypatch.setenv("MTPLX_SDPA_2PASS", "1")
        monkeypatch.setenv("MTPLX_SDPA_2PASS_THRESHOLD", "0")
        monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")
        monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA_THRESHOLD", "0")
        monkeypatch.setenv("MTPLX_ROUTE_DEBUG", "1")
        monkeypatch.setenv("MTPLX_KV_ATTENTION_TRACE", "1")
    attn = configure(dtype)
    blocks, dense = BlockOwnedKVCache(block_size=16), KVCache()

    def forbid_fetch(*args, **kwargs):
        raise AssertionError("block attention fetched/concatenated the full KV cache")

    monkeypatch.setattr(blocks, "_active_arrays", forbid_fetch)
    for rows, mask, phase in (
        (64, "causal", "prefill"),
        (64, "causal", "prefill"),
        (1, None, "ar_decode"),
        (2, "causal", "decode_verify"),
    ):
        x = mx.random.normal((1, rows, 32)).astype(dtype)
        with kv_attention_request_scope(), attention_phase(phase):
            expected = _native_attention(attn, x, mask=mask, cache=dense)
            with monkeypatch.context() as no_concat:
                no_concat.setattr(mx, "concatenate", forbid_fetch)
                no_concat.setattr(mx, "repeat", forbid_fetch)
                actual = attn(x, mask=mask, cache=blocks)
                mx.eval(actual)
            assert "route=blockwise " in kv_attention_failure_line()
        assert blocks.offset == dense.offset
        _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("query_heads", [16, 32], ids=["checkpoint", "stress"])
def test_model_head_geometry_and_verify_across_block_boundary(configure, monkeypatch, dtype, query_heads):
    attn = configure(dtype, query_heads=query_heads, kv_heads=2, head_dim=256)
    blocks, dense = BlockOwnedKVCache(block_size=64), KVCache()

    def forbid_materialization(*args, **kwargs):
        raise AssertionError("block attention concatenated or repeated KV heads")

    monkeypatch.setattr(blocks, "_active_arrays", forbid_materialization)
    for rows, phase in ((64, "prefill"), (63, "prefill"), (2, "decode_verify"), (1, "ar_decode")):
        x = mx.random.normal((1, rows, 32)).astype(dtype)
        mask = None if rows == 1 else "causal"
        with kv_attention_request_scope(), attention_phase(phase):
            expected = _native_attention(attn, x, mask=mask, cache=dense)
            with monkeypatch.context() as no_materialization:
                no_materialization.setattr(mx, "concatenate", forbid_materialization)
                no_materialization.setattr(mx, "repeat", forbid_materialization)
                actual = attn(x, mask=mask, cache=blocks)
                mx.eval(actual)
            assert "route=blockwise " in kv_attention_failure_line()
        assert blocks.offset == dense.offset
        _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("batch,query_heads", [(1, 2), (2, 32)])
def test_grouped_head_order_and_distinct_value_width(monkeypatch, batch, query_heads):
    from mtplx.block_attention import blockwise_attention

    cache = BlockOwnedKVCache(block_size=16)
    head_values = mx.arange(1, batch * 2 + 1).reshape(batch, 2, 1, 1)
    values = mx.broadcast_to(head_values, (batch, 2, 17, 8)).astype(mx.float32)
    cache.update_without_fetch(mx.zeros((batch, 2, 17, 16)), values)
    queries = mx.zeros((batch, query_heads, 2, 16))

    def forbid_repeat(*args, **kwargs):
        raise AssertionError("grouped attention repeated physical KV heads")

    monkeypatch.setattr(mx, "repeat", forbid_repeat)
    actual = blockwise_attention(queries=queries, cache=cache, scale=0.25, cached_prefix_len=15)
    expected = np.broadcast_to(
        np.repeat(np.arange(1, batch * 2 + 1).reshape(batch, 2), query_heads // 2, axis=1)
        [:, :, None, None], (batch, query_heads, 2, 8),
    )
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive"])
@pytest.mark.parametrize("rows", [2, 64])
def test_noncausal_masks_use_dense_fallback(configure, monkeypatch, mask_kind, rows):
    attn = configure()
    blocks, dense = BlockOwnedKVCache(block_size=16), KVCache()
    prefix = mx.random.normal((1, 35, 32))
    _native_attention(attn, prefix, mask="causal", cache=dense)
    blocks.update_without_fetch(dense.keys[..., :35, :], dense.values[..., :35, :])
    x = mx.random.normal((1, rows, 32))
    mask = None
    if mask_kind == "bool":
        mask = mx.broadcast_to(mx.arange(35 + rows) % 3 != 0, (rows, 35 + rows))
    elif mask_kind == "additive":
        mask = mx.broadcast_to(mx.linspace(-2, 0, 35 + rows), (rows, 35 + rows))
    fetches = []
    original_fetch = blocks._active_arrays

    def fetch():
        fetches.append(True)
        return original_fetch()

    monkeypatch.setattr(blocks, "_active_arrays", fetch)
    with kv_attention_request_scope():
        expected = _native_attention(attn, x, mask=mask, cache=dense)
        actual = attn(x, mask=mask, cache=blocks)
        assert "route=sdpa " in kv_attention_failure_line()
    assert fetches == [True]
    _assert_close(actual, expected)


def test_blockwise_threshold_still_uses_native_below_threshold(configure, monkeypatch):
    monkeypatch.setenv("MTPLX_BLOCKWISE_ATTN_THRESHOLD", "64")
    attn = configure()
    blocks, dense = BlockOwnedKVCache(block_size=16), KVCache()
    x = mx.random.normal((1, 64, 32))
    with kv_attention_request_scope():
        expected = _native_attention(attn, x, mask="causal", cache=dense)
        actual = attn(x, mask="causal", cache=blocks)
        assert "route=sdpa " in kv_attention_failure_line()
    _assert_close(actual, expected)
