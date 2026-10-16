"""Experimental affine6/group64 FP16 expert gate/up + SwiGLU decode kernel.

The packed input is [experts, gate rows followed by up rows, packed K], as in
fusion.py. No codes, scales, biases or quantization metadata are changed.
Each SIMDgroup computes gate/up pairs with MLX's qmv_fast inner loop, then
keeps the projection, sigmoid, SiLU and multiplication FP16 rounding boundaries.
Install apply(model) after fusion.apply and before opt.apply; undo in reverse.

The 6-bit load/dot and sigmoid helpers below are adapted from MLX v0.32.2:
https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/quantized.h
https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/unary_ops.h
Copyright © 2023-2024 Apple Inc. Licensed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0). Distributed on an AS IS basis,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
Modified here to pair gate/up rows and write their SwiGLU instead of both GEMVs.
"""

from functools import lru_cache

import mlx.core as mx
from mlx_lm.models.qwen3_5 import DecoderLayer
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock
from mlx_lm.models.switch_layers import SwitchGLU, SwiGLU

from .fusion import _ExpertMember


_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

template <bool COMPACT>
inline float load_vector6(const device half* x, thread float* values) {
    float sum = 0;
    for (int i = 0; i < 8; i += 4) {
        // Keep the native half additions before accumulating the bias sum.
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        if (COMPACT) {
            for (int j = 0; j < 4; j++) values[i + j] = x[i + j];
        } else {
            values[i] = x[i];
            values[i + 1] = x[i + 1] / 64.0f;
            values[i + 2] = x[i + 2] / 16.0f;
            values[i + 3] = x[i + 3] / 4.0f;
        }
    }
    return sum;
}

template <bool COMPACT>
inline float qdot6(const device uint8_t* w, const thread float* values,
                  float scale, float bias, float sum) {
    float accum = 0;
    for (int i = 0; i < 2; i++) {
        values += 4 * i;
        w += 3 * i;
        if (COMPACT) {
            const uint q0 = w[0] & 63;
            const uint q1 = (w[0] >> 6) | ((w[1] & 15) << 2);
            const uint q2 = (w[1] >> 4) | ((w[2] & 3) << 4);
            const uint q3 = w[2] >> 2;
            accum += q0 * values[0];
            accum += q1 * values[1];
            accum += q2 * values[2];
            accum += q3 * values[3];
        } else {
            accum += (w[0] & 0x3f) * values[0];
            accum += (w[0] & 0xc0) * values[1];
            accum += (w[1] & 0x0f) * (values[1] * 256.0f);
            accum += (w[1] & 0xf0) * values[2];
            accum += (w[2] & 0x03) * (values[2] * 256.0f);
            accum += (w[2] & 0xfc) * values[3];
        }
    }
    return scale * accum + sum * bias;
}

inline half sigmoid_half(half x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
"""

_SOURCE = r"""
    const uint slot = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * (PAIRS * SIMDGROUPS) +
                     simdgroup_index_in_threadgroup * PAIRS;
    const uint expert = indices[slot];
    if (expert >= E) {
        // A bad router index must never read outside the weight buffers.
        if (lane == 0) {
            for (int pair = 0; pair < PAIRS; pair++) h[slot * N + row + pair] = half(NAN);
        }
        return;
    }
    constexpr int ROW_BYTES = K * 6 / 8;
    constexpr int GROUPS = K / 64;
    const size_t first = size_t(expert) * (2 * N) + row;
    const device uint8_t* ws = (const device uint8_t*)weight +
                               first * ROW_BYTES + lane * 6;
    const device half* ss = scales + first * GROUPS + lane / 8;
    const device half* bs = biases + first * GROUPS + lane / 8;
    const device half* xs = x + (slot / TOPK) * K + lane * 8;
    thread float values[8];
    thread float result[2 * PAIRS] = {0};
    for (int k = 0; k < K; k += 256) {
        float sum = load_vector6<COMPACT>(xs, values);
        for (int pair = 0; pair < PAIRS; pair++) {
            result[2 * pair] += qdot6<COMPACT>(ws + pair * ROW_BYTES, values,
                                     ss[pair * GROUPS], bs[pair * GROUPS], sum);
            result[2 * pair + 1] += qdot6<COMPACT>(ws + (N + pair) * ROW_BYTES, values,
                                         ss[(N + pair) * GROUPS],
                                         bs[(N + pair) * GROUPS], sum);
        }
        ws += 192;
        ss += 4;
        bs += 4;
        xs += 256;
    }
    for (int pair = 0; pair < PAIRS; pair++) {
        float gate_sum = simd_sum(result[2 * pair]);
        float up_sum = simd_sum(result[2 * pair + 1]);
        if (lane == 0) {
            half gate = half(gate_sum);
            half up = half(up_sum);
            half sigmoid = sigmoid_half(gate);
            half silu = gate * sigmoid;
            h[slot * N + row + pair] = silu * up;
        }
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="affine6_gate_up_swiglu", input_names=["x", "indices", "weight", "scales", "biases"],
        output_names=["h"], source=_SOURCE, header=_HEADER,
        compile_options={"math_mode": "safe"},
    )


def _layout(weight, scales, biases, k):
    if weight.ndim != 3 or weight.dtype != mx.uint32 or weight.shape[0] < 1:
        raise ValueError("gate_up requires [experts, 2*N, packed K] uint32 weights")
    n = weight.shape[1] // 2
    if k < 256 or k % 256 or n < 4 or n % 4 or weight.shape[1] != 2 * n:
        raise ValueError("gate_up requires K multiple256 and 2*N with N multiple4")
    if weight.shape[-1] != k * 6 // 32:
        raise ValueError("gate_up requires 6-bit packed rows matching input width")
    if (scales.shape != (weight.shape[0], 2 * n, k // 64) or biases.shape != scales.shape
            or scales.dtype != mx.float16 or biases.dtype != mx.float16):
        raise ValueError("gate_up requires matching group64 float16 scales and biases")
    return n


def gate_up(x, indices, weight, scales, biases, *, compact=False, pairs_per_simd=2, simdgroups=2):
    """Return [1, S, selected experts, 1, N] for S=1..3 FP16 tokens.

    Inputs must be affine6/group64 weights with gate rows preceding up rows.
    An out-of-range uint32 expert index produces NaNs without accessing weights.
    compact=True is a benchmark-only integer-unpack variant; its floating-point
    dot order differs from MLX. The default keeps the native dot expression.
    pairs_per_simd and simdgroups expose row/occupancy tuning; model decode uses
    the measured 4-pair/4-SIMDgroup tile when its output width permits it.
    # shortcut: B=1/S<=3, FP16, K multiple256 and N multiple4; extend only after benchmarks.
    """
    if (x.ndim != 3 or x.shape[0] != 1 or x.shape[1] not in (1, 2, 3)
            or x.dtype != mx.float16):
        raise ValueError("gate_up requires [1, S, K], S=1..3, float16 decode token")
    if indices.ndim != 3 or indices.shape[:2] != x.shape[:2] or indices.dtype != mx.uint32 or indices.size < 1:
        raise ValueError("gate_up requires nonempty [1, S, selected experts] uint32 indices")
    k = x.shape[-1]
    n = _layout(weight, scales, biases, k)
    if pairs_per_simd not in (1, 2, 4, 8) or simdgroups not in (2, 4) or n % (pairs_per_simd * simdgroups):
        raise ValueError("gate_up requires pairs_per_simd in 1/2/4/8, simdgroups in 2/4, and N divisible by their product")
    return _kernel()(
        inputs=[x, indices, weight, scales, biases],
        template=[("K", k), ("TOPK", indices.shape[-1]), ("N", n), ("E", weight.shape[0]), ("COMPACT", compact),
                  ("PAIRS", pairs_per_simd), ("SIMDGROUPS", simdgroups)],
        grid=(n // pairs_per_simd * 32, indices.size, 1), threadgroup=(32 * simdgroups, 1, 1),
        output_shapes=[(*indices.shape, 1, n)], output_dtypes=[mx.float16],
    )[0]


class _FusedSwitchGLU(SwitchGLU):
    def _decode_ready(self, x, indices):
        return (not self.training and not self.gate_proj.training and not self.up_proj.training
                and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] in (1, 2, 3) and x.dtype == mx.float16
                and indices.ndim == 3 and indices.shape[:2] == x.shape[:2]
                and self.gate_proj.output_dims % 16 == 0
                and indices.dtype == mx.uint32 and 0 < indices.size < 64)

    def _hidden(self, x, indices):
        hub = self.gate_proj._fusion
        if hub.outputs is not None:
            hub.clear()
        packed = hub.projection
        return gate_up(x, indices, packed.weight, packed.scales, packed.biases,
                       pairs_per_simd=4, simdgroups=4)

    def __call__(self, x, indices):
        if self._decode_ready(x, indices):
            return self.down_proj(self._hidden(x, indices), indices).squeeze(-2)
        return SwitchGLU.__call__(self, x, indices)

    def weighted(self, x, indices, scores):
        # Local imports avoid the down/reduce modules' dependency on our qdot header.
        from .down_kernel import _DownProjection

        down = self.down_proj
        if (self._decode_ready(x, indices) and type(down) is _DownProjection
                and not down.training and self.gate_proj.output_dims == 512
                and indices.shape == (*x.shape[:2], 8) and scores.shape == indices.shape
                and scores.dtype == mx.float16):
            from .reduce_kernel import weighted_down

            return weighted_down(self._hidden(x, indices), indices, scores,
                                 down.weight, down.scales, down.biases)
        return (self(x, indices) * scores[..., None]).sum(axis=-2)


def apply(model):
    """Install native-qdot6 gate/up + SwiGLU after fusion, before compilation.

    Reuse fusion's packed buffers and preserve every parameter name and array.
    Longer inputs, batches and training retain SwitchGLU's original path.
    A one-token initial prompt also uses the kernel. Undo before changing weights
    or undoing projection fusion; the returned undo is idempotent.
    """
    if not model.layers or any(type(layer) is not DecoderLayer
                               or type(layer.mlp) is not Qwen3NextSparseMoeBlock
                               for layer in model.layers):
        raise TypeError("MoE kernel requires Qwen3.5/3.6 before decode compilation")
    modules = []
    for layer in model.layers:
        module = layer.mlp.switch_mlp
        if layer.mlp.sharding_group is not None:
            raise ValueError("MoE kernel requires an unsharded model")
        if type(module) is not SwitchGLU or type(module.activation) is not SwiGLU:
            raise TypeError("MoE kernel requires unpatched SwitchGLU with stock SwiGLU")
        gate, up = module.gate_proj, module.up_proj
        if type(gate) is not _ExpertMember or type(up) is not _ExpertMember:
            raise TypeError("apply projection fusion before the MoE kernel")
        hub = gate._fusion
        if (up._fusion is not hub or hub.parts != 2 or gate._fusion_index != 0
                or up._fusion_index != 1 or hub.boundaries != [gate.output_dims]
                or gate.weight.shape != up.weight.shape):
            raise ValueError("MoE kernel requires one fused gate/up group in that order")
        packed = hub.projection
        for projection in (gate, up, packed):
            if ((projection.bits, projection.group_size, projection.mode) != (6, 64, "affine")
                    or "bias" in projection or projection.get("biases") is None
                    or projection.scales.dtype != mx.float16 or projection.biases.dtype != mx.float16):
                raise ValueError("MoE kernel requires affine6/group64 FP16 without additive bias")
        if _layout(packed.weight, packed.scales, packed.biases, gate.input_dims) != gate.output_dims:
            raise ValueError("MoE fused pack does not match its gate/up projections")
        modules.append(module)
    for module in modules:
        module.gate_proj._fusion.clear()
        module.__class__ = _FusedSwitchGLU

    def undo():
        for module in reversed(modules):
            module.gate_proj._fusion.clear()
            module.__class__ = SwitchGLU
        modules.clear()

    return undo
