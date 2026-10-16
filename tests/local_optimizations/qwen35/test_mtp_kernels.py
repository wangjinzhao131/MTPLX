"""MTP rows preserve each native expert result, route and reduction order."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.local_optimizations.qwen35.moe_kernel import gate_up
from mtplx.local_optimizations.qwen35.down_kernel import down
from mtplx.local_optimizations.qwen35.reduce_kernel import weighted_down
from mtplx.local_optimizations.qwen35.router_kernel import route
from .test_router_kernel import reference as native_route


@pytest.mark.parametrize('length', [1, 2, 3])
def test_verify_rows_match_native(length):
    mx.random.seed(811 + length)
    packed = mx.quantize(mx.random.normal((8, 1024, 256), dtype=mx.float16) / 16,
                        group_size=64, bits=6)
    down_weights = mx.quantize(mx.random.normal((8, 256, 512), dtype=mx.float16) / 16,
                              group_size=64, bits=6)
    x = mx.random.normal((1, 2 * length, 256), dtype=mx.float16)[:, ::2]
    indices = mx.random.randint(0, 8, (1, length, 8)).astype(mx.uint32)
    logits = mx.random.normal((1, length, 256), dtype=mx.float16)
    for logits in (logits, mx.round(logits), mx.zeros_like(logits)):
        expected_ids, expected_scores = native_route(logits)
        actual_ids, scores = route(logits)
        assert mx.array_equal(actual_ids, expected_ids).item()
        assert mx.array_equal(scores, expected_scores).item()
    native = mx.gather_qmm(x[:, :, None, None, :], *packed, rhs_indices=indices,
                          transpose=True, group_size=64, bits=6)
    g, u = mx.split(native, 2, axis=-1)
    expected = nn.silu(g) * u
    actual = gate_up(x, indices, *packed, pairs_per_simd=4, simdgroups=4)
    assert mx.array_equal(actual, expected).item()
    projected = mx.gather_qmm(expected, *down_weights, rhs_indices=indices,
                             transpose=True, group_size=64, bits=6)
    assert mx.array_equal(down(actual, indices, *down_weights), projected).item()
    expected_sum = (projected.squeeze(-2) * scores[..., None]).sum(-2)
    for operation in (weighted_down, mx.compile(weighted_down)):
        result = operation(actual, indices, scores, *down_weights)
        assert mx.array_equal(result, expected_sum).item()
