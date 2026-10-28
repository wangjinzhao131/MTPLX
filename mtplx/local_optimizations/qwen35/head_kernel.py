"""Exact affine6/group64 two/three-token target vocabulary projection.

Shares each weight tile across verification tokens. The native qdot6 helper in
moe_kernel carries the MLX v0.32.2 Apache-2.0 attribution; FP16 dot rounding stays
unchanged. Other shapes, quantizations and training use the original head.
"""
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .moe_kernel import _HEADER

_SOURCE = r"""
const uint lane=thread_index_in_simdgroup;
const uint row=threadgroup_position_in_grid.x*(4*2)+simdgroup_index_in_threadgroup*4;
constexpr int GROUPS=K/64;
constexpr int ROW_BYTES=K*6/8;
const device uint8_t* ws=(const device uint8_t*)weight+row*ROW_BYTES+lane*6;
const device half* ss=scales+row*GROUPS+lane/8;
const device half* bs=biases+row*GROUPS+lane/8;
thread float result[TOKENS*4]={0};
for(int k=0;k<K;k+=256){
    thread float values[TOKENS][8];
    thread float sums[TOKENS];
    for(int t=0;t<TOKENS;t++) sums[t]=load_vector6<false>(x+t*K+k+lane*8,values[t]);
    for(int r=0;r<4;r++){
        for(int t=0;t<TOKENS;t++) result[t*4+r]+=qdot6<false>(ws+r*ROW_BYTES,values[t],ss[r*GROUPS],bs[r*GROUPS],sums[t]);
    }
    ws+=192;ss+=4;bs+=4;
}
for(int r=0;r<4;r++){
    for(int t=0;t<TOKENS;t++){
        float value=simd_sum(result[t*4+r]);
        if(lane==0)y[t*N+row+r]=half(value);
    }
}
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="affine6_multirow_lm_head",
        input_names=["x", "weight", "scales", "biases"], output_names=["y"],
        source=_SOURCE, header=_HEADER, compile_options={"math_mode": "safe"},
    )


def forward(head, x):
    # shortcut: native affine6 FP16 B1/S2-S3 only; extend after matched benchmarks.
    if (type(head) is not nn.QuantizedLinear or head.training
            or (head.bits, head.group_size, head.mode) != (6, 64, "affine")
            or "bias" in head or getattr(head, "sharding_group", None) is not None
            or x.ndim != 3 or x.shape[0] != 1 or x.shape[1] not in (2, 3)
            or x.dtype != mx.float16):
        return head(x)
    weight, scales, biases = head.weight, head.scales, head.get("biases")
    k = x.shape[-1]
    if (weight.ndim != 2 or k < 256 or k % 256
            or weight.shape[0] < 8 or weight.shape[0] % 8
            or weight.shape[1] != k * 6 // 32 or weight.dtype != mx.uint32
            or biases is None or scales.shape != (weight.shape[0], k // 64)
            or biases.shape != scales.shape or scales.dtype != mx.float16
            or biases.dtype != mx.float16):
        return head(x)
    n = weight.shape[0]
    return _kernel()(
        inputs=[x, weight, scales, biases],
        template=[("N", n), ("K", k), ("TOKENS", x.shape[1])],
        grid=(n // 4 * 32, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(1, x.shape[1], n)], output_dtypes=[mx.float16],
    )[0]
