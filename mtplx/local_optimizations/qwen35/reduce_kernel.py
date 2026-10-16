"""Affine6/group64 FP16 down projection, route weighting, and expert sum.

Uses MLX v0.32.2's qmv_fast arithmetic through moe_kernel._HEADER, which
carries the upstream Apache-2.0 attribution. Each SIMDgroup computes one
expert's four rows; the threadgroup then reduces the eight FP16 products in
native col_reduce_small slot order. Packed checkpoint arrays stay unchanged.
"""

from functools import lru_cache

import mlx.core as mx

from .moe_kernel import _HEADER


_SOURCE = r"""
    const uint slot = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * 4;
    const uint expert = indices[slot];
    thread float result[4] = {0};
    if (expert < E) {
        constexpr int ROW_BYTES = 512 * 6 / 8;
        constexpr int GROUPS = 512 / 64;
        const size_t first = size_t(expert) * N + row;
        const device uint8_t* ws = (const device uint8_t*)weight +
                                   first * ROW_BYTES + lane * 6;
        const device half* ss = scales + first * GROUPS + lane / 8;
        const device half* bs = biases + first * GROUPS + lane / 8;
        const device half* xs = x + slot * 512 + lane * 8;
        thread float values[8];
        for (int k = 0; k < 512; k += 256) {
            float sum = load_vector6<false>(xs, values);
            for (int r = 0; r < 4; r++) {
                result[r] += qdot6<false>(ws + r * ROW_BYTES, values,
                                         ss[r * GROUPS], bs[r * GROUPS], sum);
            }
            ws += 192;
            ss += 4;
            bs += 4;
            xs += 256;
        }
    }
    threadgroup half partials[8 * 4];
    for (int r = 0; r < 4; r++) {
        float value = simd_sum(result[r]);
        if (lane == 0) {
            half projected = expert < E ? half(value) : half(NAN);
            half product = projected * scores[slot];
            partials[slot * 4 + r] = product + half(0);
        }
    }
    // Every expert reaches this barrier, including invalid expert IDs.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot == 0 && lane < 4) {
        half total = partials[lane];
        for (int j = 1; j < 8; j++) total = partials[j * 4 + lane] + total;
        y[row + lane] = total;
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="affine6_weighted_expert_down",
        input_names=["x", "indices", "scores", "weight", "scales", "biases"],
        output_names=["y"], source=_SOURCE, header=_HEADER,
        compile_options={"math_mode": "safe"},
    )


def weighted_down(x, indices, scores, weight, scales, biases):
    """Return [1, 1, N], preserving projection/product/sum FP16 rounding.

    Invalid expert IDs produce NaNs without reading their weights.
    # shortcut: eight experts, FP16, K512, four-row tiles; profile before extending.
    """
    if x.shape != (1, 1, 8, 1, 512) or x.dtype != mx.float16:
        raise ValueError("weighted down requires [1, 1, 8, 1, 512] float16 activations")
    if indices.shape != (1, 1, 8) or indices.dtype != mx.uint32:
        raise ValueError("weighted down requires [1, 1, 8] uint32 expert indices")
    if scores.shape != (1, 1, 8) or scores.dtype != mx.float16:
        raise ValueError("weighted down requires [1, 1, 8] float16 routing scores")
    if (weight.ndim != 3 or weight.dtype != mx.uint32 or weight.shape[0] < 1
            or weight.shape[-1] != 96 or weight.shape[1] < 8 or weight.shape[1] % 8):
        raise ValueError("weighted down requires [E, N, 96] uint32 weights and N divisible by8")
    e, n, _ = weight.shape
    if (scales.shape != (e, n, 8) or biases.shape != scales.shape
            or scales.dtype != mx.float16 or biases.dtype != mx.float16):
        raise ValueError("weighted down requires matching group64 float16 scales and biases")
    return _kernel()(
        inputs=[x, indices, scores, weight, scales, biases],
        template=[("N", n), ("E", e)], grid=(n // 4 * 256, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(1, 1, n)], output_dtypes=[mx.float16],
    )[0]
