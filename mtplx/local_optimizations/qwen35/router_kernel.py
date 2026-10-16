"""Exact-order 256-expert/top8 router candidate for MLX0.32.2 Metal.

Preserves FP32 precise-softmax accumulation, dtype-rounded probabilities,
stable argpartition order, and the original dtype's sequential top8 sum.
"""

import mlx.core as mx
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from .moe_kernel import _FusedSwitchGLU


_SOURCE = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    threadgroup float maxima[32];
    threadgroup float normalizers[32];
    threadgroup T probabilities[256];
    float values[4];
    float maximum = -INFINITY;
    for (uint i = 0; i < 4; ++i) {
        values[i] = float(logits[4 * tid + i]);
        maximum = maximum < values[i] ? values[i] : maximum;
    }
    if (simd == 0) {
        maxima[lane] = -INFINITY;
        normalizers[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = simd_max(maximum);
    if (lane == 0) maxima[simd] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        maximum = simd_max(maxima[lane]);
        if (lane == 0) maxima[0] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float normalizer = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        values[i] = fast::exp(values[i] - maxima[0]);
        normalizer += values[i];
    }
    normalizer = simd_sum(normalizer);
    if (lane == 0) normalizers[simd] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        normalizer = simd_sum(normalizers[lane]);
        if (lane == 0) normalizers[0] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = 1 / normalizers[0];
    for (uint i = 0; i < 4; ++i) {
        probabilities[4 * tid + i] = T(values[i] * normalizer);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0) return;
    // Any NaN/positive infinity (or all negative infinities) makes every probability NaN.
    if (!isfinite(normalizer)) {
        if (lane < 8) {
            indices[lane] = 248 + lane;
            scores[lane] = T(NAN);
        }
        return;
    }
    float remaining[8];
    T selected[8];
    uint selected_indices[8];
    for (uint i = 0; i < 8; ++i) remaining[i] = float(probabilities[8 * lane + i]);
    for (uint rank = 0; rank < 8; ++rank) {
        float best = -INFINITY;
        uint best_index = 0;
        for (uint i = 0; i < 8; ++i) {
            if (remaining[i] >= best) {
                best = remaining[i];
                best_index = 8 * lane + i;
            }
        }
        float winner = simd_max(best);
        uint winner_index = simd_max(best == winner ? best_index : 0);
        selected[7 - rank] = T(winner);
        selected_indices[7 - rank] = winner_index;
        if (winner_index / 8 == lane) remaining[winner_index % 8] = -INFINITY;
    }
    // Native [1,1,8].sum(-1) uses one thread and dtype-rounded sequential additions.
    T total = T(0);
    for (uint i = 0; i < 8; ++i) total = selected[i] + total;
    if (lane < 8) {
        indices[lane] = selected_indices[lane];
        scores[lane] = selected[lane] / total;
    }
"""


_kernel = mx.fast.metal_kernel(
    name="router256_softmax_top8", input_names=["logits"],
    output_names=["indices", "scores"], source=_SOURCE,
)


def route(logits):
    """Return stock-ordered uint32 indices and normalized probabilities."""
    if logits.shape != (1, 1, 256) or logits.dtype not in (mx.float16, mx.float32):
        raise ValueError("router kernel requires FP16/FP32 logits of shape (1,1,256)")
    return _kernel(inputs=[logits], template=[("T", logits.dtype)],
                   grid=(64, 1, 1), threadgroup=(64, 1, 1),
                   output_shapes=[(1, 1, 8), (1, 1, 8)], output_dtypes=[mx.uint32, logits.dtype])


def moe_forward(block, x):
    """Stateless native MoE forward with fused routing for single-token decode."""
    # shortcut: specialized B1/S1/top8/256 router; retain native paths for other shapes.
    if (block.training or block.sharding_group is not None or x.ndim != 3 or x.shape[:2] != (1, 1)
            or block.num_experts != 256 or block.top_k != 8 or not block.norm_topk_prob
            or x.dtype not in (mx.float16, mx.float32)):
        return Qwen3NextSparseMoeBlock.__call__(block, x)
    indices, scores = route(block.gate(x))
    if type(block.switch_mlp) is _FusedSwitchGLU:
        y = block.switch_mlp.weighted(x, indices, scores)
    else:
        y = block.switch_mlp(x, indices)
        y = (y * scores[..., None]).sum(axis=-2)
    shared_y = block.shared_expert(x)
    shared_y = mx.sigmoid(block.shared_expert_gate(x)) * shared_y
    return y + shared_y


class _RoutedMoE(Qwen3NextSparseMoeBlock):
    __call__ = moe_forward


def apply(model):
    """Install after projection/GDN/MoE fusion, before opt.apply; undo in reverse.

    Prefill, batches and training retain native routing. Undo is idempotent.
    """
    modules = [layer.mlp for layer in model.layers]
    if not modules or any(type(module) is not Qwen3NextSparseMoeBlock for module in modules):
        raise TypeError("router kernel requires unpatched MoE modules before decode compilation")
    if any(module.num_experts != 256 or module.top_k != 8 or not module.norm_topk_prob
           or module.sharding_group is not None for module in modules):
        raise ValueError("router kernel requires unsharded 256-expert/top8 normalized routing")
    for module in modules:
        module.__class__ = _RoutedMoE

    def undo():
        for module in reversed(modules):
            module.__class__ = Qwen3NextSparseMoeBlock
        modules.clear()

    return undo
