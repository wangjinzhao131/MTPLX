"""Affine6/group64 FP16 expert down GEMV with unchanged packed weights.

Uses MLX v0.32.2's qmv_fast per-row arithmetic through moe_kernel._HEADER;
that module carries the upstream Apache-2.0 attribution. Only the number of
rows per SIMDgroup and SIMDgroups per threadgroup changes. Packed weights,
scales, biases, expert order and the final FP16 rounding remain unchanged.
"""

from functools import lru_cache

import mlx.core as mx
from mlx_lm.models.qwen3_5 import DecoderLayer
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from .moe_kernel import _HEADER


TILES = tuple((rows, groups) for rows in (2, 4, 8, 16) for groups in (2, 4))

_SOURCE = r"""
    const uint slot = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * (ROWS * SIMDGROUPS) +
                     simdgroup_index_in_threadgroup * ROWS;
    const uint expert = indices[slot];
    if (expert >= E) {
        if (lane == 0) {
            for (int r = 0; r < ROWS; r++) y[slot * N + row + r] = half(NAN);
        }
        return;
    }
    constexpr int ROW_BYTES = 512 * 6 / 8;
    constexpr int GROUPS = 512 / 64;
    const size_t first = size_t(expert) * N + row;
    const device uint8_t* ws = (const device uint8_t*)weight +
                               first * ROW_BYTES + lane * 6;
    const device half* ss = scales + first * GROUPS + lane / 8;
    const device half* bs = biases + first * GROUPS + lane / 8;
    const device half* xs = x + slot * 512 + lane * 8;
    thread float values[8];
    thread float result[ROWS] = {0};
    for (int k = 0; k < 512; k += 256) {
        float sum = load_vector6<false>(xs, values);
        for (int r = 0; r < ROWS; r++) {
            result[r] += qdot6<false>(ws + r * ROW_BYTES, values,
                                     ss[r * GROUPS], bs[r * GROUPS], sum);
        }
        ws += 192;
        ss += 4;
        bs += 4;
        xs += 256;
    }
    for (int r = 0; r < ROWS; r++) {
        float value = simd_sum(result[r]);
        if (lane == 0) y[slot * N + row + r] = half(value);
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="affine6_expert_down", input_names=["x", "indices", "weight", "scales", "biases"],
        output_names=["y"], source=_SOURCE, header=_HEADER,
        compile_options={"math_mode": "safe"},
    )


def down(x, indices, weight, scales, biases, *, results_per_simd=4, simdgroups=2):
    """Return [1, S, 8, 1, N] with one independent activation per expert slot.

    The checkpoint has E=256, N=2048; smaller E/N also permit inexpensive
    numerical tests. Invalid expert IDs produce NaNs without reading weights.
    # shortcut: eight experts, FP16, K=512; extend only after real-weight profiling.
    """
    if (results_per_simd, simdgroups) not in TILES or not all(
            type(value) is int for value in (results_per_simd, simdgroups)):
        raise ValueError("down tile requires rows in (2, 4, 8, 16) and SIMDgroups in (2, 4)")
    if (x.ndim != 5 or x.shape[0] != 1 or x.shape[1] not in (1, 2, 3)
            or x.shape[2:] != (8, 1, 512) or x.dtype != mx.float16):
        raise ValueError("down requires [1, S, 8, 1, 512], S=1..3, float16 activations")
    if indices.shape != (*x.shape[:2], 8) or indices.dtype != mx.uint32:
        raise ValueError("down requires [1, S, 8] uint32 expert indices")
    if (weight.ndim != 3 or weight.dtype != mx.uint32 or weight.shape[0] < 1
            or weight.shape[-1] != 96 or weight.shape[1] < 1
            or weight.shape[1] % (results_per_simd * simdgroups)):
        raise ValueError("down requires [E, N, 96] uint32 weights and N divisible by tile rows")
    e, n, _ = weight.shape
    if (scales.shape != (e, n, 8) or biases.shape != scales.shape
            or scales.dtype != mx.float16 or biases.dtype != mx.float16):
        raise ValueError("down requires matching group64 float16 scales and biases")
    return _kernel()(
        inputs=[x, indices, weight, scales, biases],
        template=[("N", n), ("E", e), ("ROWS", results_per_simd), ("SIMDGROUPS", simdgroups)],
        grid=(n // (results_per_simd * simdgroups) * 32 * simdgroups, indices.size, 1),
        threadgroup=(32 * simdgroups, 1, 1),
        output_shapes=[(*indices.shape, 1, n)], output_dtypes=[mx.float16],
    )[0]


class _DownProjection(QuantizedSwitchLinear):
    def __call__(self, x, indices, sorted_indices=False):
        if (not self.training and not sorted_indices
                and x.ndim == 5 and x.shape[0] == 1 and x.shape[1] in (1, 2, 3)
                and x.shape[2:] == (8, 1, 512) and x.dtype == mx.float16
                and indices.shape == (*x.shape[:2], 8) and indices.dtype == mx.uint32):
            return down(x, indices, self.weight, self.scales, self.biases,
                        results_per_simd=4, simdgroups=4)
        return QuantizedSwitchLinear.__call__(self, x, indices, sorted_indices)


def apply(model):
    """Install after projection/MoE fusion and before router/opt compilation.

    Only the original down projection's class changes. Parameters retain their
    names and identities. Undo compilation before undoing this patch, and undo
    this patch before changing weights. The returned undo is idempotent.
    """
    if not model.layers or any(type(layer) is not DecoderLayer
                               or type(layer.mlp) is not Qwen3NextSparseMoeBlock
                               for layer in model.layers):
        raise TypeError("down kernel requires Qwen3.5/3.6 before router/compilation")
    modules = []
    for layer in model.layers:
        if layer.mlp.sharding_group is not None:
            raise ValueError("down kernel requires an unsharded model")
        module = layer.mlp.switch_mlp.down_proj
        if type(module) is not QuantizedSwitchLinear:
            raise TypeError("down kernel requires unpatched quantized down projections")
        if ((module.bits, module.group_size, module.mode) != (6, 64, "affine")
                or "bias" in module or module.get("biases") is None):
            raise ValueError("down kernel requires affine6/group64 without additive bias")
        weight, scales, biases = module.weight, module.scales, module.biases
        if (weight.ndim != 3 or weight.dtype != mx.uint32 or weight.shape[0] < 1
                or weight.shape[-1] != 96 or weight.shape[1] < 1 or weight.shape[1] % 16
                or scales.shape != (*weight.shape[:2], 8) or biases.shape != scales.shape
                or scales.dtype != mx.float16 or biases.dtype != mx.float16):
            raise ValueError("down kernel requires FP16 scales/biases, K512 and N divisible by16")
        modules.append(module)
    for module in modules:
        module.__class__ = _DownProjection

    def undo():
        for module in reversed(modules):
            module.__class__ = QuantizedSwitchLinear
        modules.clear()

    return undo
