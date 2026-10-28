"""B=1/S=1–3 GDN preprocessing with optional prefix capture; weights stay unchanged.

Arithmetic follows MLX v0.32.2 conv.metal, unary_ops.h and rms_norm.metal:
FP32 convolution accumulation, dtype-rounded convolution/SiLU/RMSNorm/scale,
and four consecutive squared values per lane before the RMS SIMD reduction.
"""

import struct

import mlx.core as mx
from ...vendored_arrays_cache import FixedArraysCache
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_5 import GatedDeltaNet


_HEADER = """
template <typename T> T gdn_sigmoid(T x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
"""

_SOURCE = """
    const uint t = Tokens == 1 ? 0 : thread_position_in_grid.z;
    const uint head = thread_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const bool is_qk = head < 2 * Hk;
    const uint width = is_qk ? Dk : Dv;
    const uint start = is_qk ? head * Dk : 2 * Hk * Dk + (head - 2 * Hk) * Dv;
    constexpr uint channels = 2 * Hk * Dk + Hv * Dv;
    T values[4];
    float squares = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        const uint column = 4 * lane + i;
        values[i] = T(0);
        if (column < width) {
            const uint c = start + column;
            float acc = 0.0f;
            for (uint tap = 0; tap < 3; ++tap) {
                const uint pos = t + tap;
                T input = pos < 3 ? conv_state[pos * channels + c]
                                  : qkv[(pos - 3) * channels + c];
                acc += float(input) * weight[4 * c + tap];
            }
            acc += float(qkv[t * channels + c]) * weight[4 * c + 3];
            T convolved = T(acc);
            T sigmoid = gdn_sigmoid(convolved);
            values[i] = convolved * sigmoid;
            for (uint tap = 0; tap < 3; ++tap) {
                const uint pos = t + tap + 1;
                next_conv[(t * 3 + tap) * channels + c] = pos < 3
                    ? conv_state[pos * channels + c] : qkv[(pos - 3) * channels + c];
            }
        }
        float v = float(values[i]);
        squares += v * v;
    }
    if (is_qk) {
        float sum = simd_sum(squares);
        float inverse = metal::precise::rsqrt(sum / float(Dk) + 1e-6f);
        T scale = head < Hk ? T(as_type<float>(uint(QscaleBits))) : T(as_type<float>(uint(KscaleBits)));
        for (uint i = 0; i < 4; ++i) {
            const uint column = 4 * lane + i;
            if (column < Dk) {
                T normalized = T(float(values[i]) * inverse);
                T scaled = normalized * scale;
                if (head < Hk) {
                    q[(t * Hk + head) * Dk + column] = scaled;
                } else {
                    k[(t * Hk + head - Hk) * Dk + column] = scaled;
                }
            }
        }
    } else {
        for (uint i = 0; i < 4; ++i) {
            const uint column = 4 * lane + i;
            if (column < Dv) {
                v[(t * Hv + head - 2 * Hk) * Dv + column] = values[i];
            }
        }
    }
"""


_kernel = mx.fast.metal_kernel(
    name="gdn_conv4_silu_rms", input_names=["qkv", "conv_state", "weight"],
    output_names=["q", "k", "v", "next_conv"], source=_SOURCE, header=_HEADER,
)


def preprocess(gdn, qkv, conv_state, *, capture=False):
    """Return normalized q/k, activated v and the final or every prefix conv state."""
    hk, hv = gdn.num_k_heads, gdn.num_v_heads
    dk, dv = gdn.head_k_dim, gdn.head_v_dim
    channels = 2 * hk * dk + hv * dv
    # shortcut: one SIMD group per head; support wider heads only when needed.
    if not (0 < dk <= 128 and 0 < dv <= 128 and dk % 4 == dv % 4 == 0):
        raise ValueError("GDN preprocessing needs head dimensions divisible by4 and <=128")
    tokens = qkv.shape[1] if qkv.ndim == 3 else 0
    if (tokens not in ((1, 2, 3) if capture else (1,))
            or qkv.shape != (1, tokens, channels) or conv_state.shape != (1, 3, channels)
            or gdn.conv1d.weight.shape != (channels, 4, 1)):
        raise ValueError("GDN preprocessing needs B1/S1 (S1–3 for capture), four convolution taps and matching state")
    if qkv.dtype not in (mx.float16, mx.float32) or not (
            qkv.dtype == conv_state.dtype == gdn.conv1d.weight.dtype):
        raise ValueError("GDN preprocessing needs matching FP16 or FP32 inputs")
    # Metal template arguments cannot be floats; keep Python-rounded scalar bits.
    q_scale, k_scale = (dk ** -0.5) ** 2, dk ** -0.5
    return _kernel(
        inputs=[qkv, conv_state, gdn.conv1d.weight],
        template=[("T", qkv.dtype), ("Tokens", tokens), ("Hk", hk), ("Hv", hv), ("Dk", dk), ("Dv", dv),
                  ("QscaleBits", struct.unpack("<I", struct.pack("<f", q_scale))[0]),
                  ("KscaleBits", struct.unpack("<I", struct.pack("<f", k_scale))[0])],
        grid=(32, 2 * hk + hv, tokens), threadgroup=(32, 1, 1),
        output_shapes=[(1, tokens, hk, dk), (1, tokens, hk, dk), (1, tokens, hv, dv),
                       (1, tokens, 3, channels) if capture else conv_state.shape],
        output_dtypes=[qkv.dtype] * 4,
    )


class _FusedGDN(GatedDeltaNet):
    def _capture_forward(self, inputs, conv_state, recurrent_state):
        """Called only after the shared capture guard validates shape, cache and flags."""
        from mlx_lm.models.gated_delta import compute_g
        from ...gdn_capture import _linear_gated_delta_capture

        tokens = inputs.shape[1]
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(1, tokens, self.num_v_heads, self.head_v_dim)
        b, a = self.in_proj_b(inputs), self.in_proj_a(inputs)
        q, k, v, conv_states = preprocess(self, qkv, conv_state, capture=True)
        if tokens == 1:
            out, state = gated_delta_update(
                q, k, v, a, b, self.A_log, self.dt_bias, recurrent_state, None, use_kernel=True)
            states = state[:, None]
        else:
            out, states = _linear_gated_delta_capture(
                q, k, v, compute_g(self.A_log, a, self.dt_bias), mx.sigmoid(b), recurrent_state)
        out = self.out_proj(self.norm(out, z).reshape(1, tokens, -1))
        return out, conv_states, states

    def __call__(self, inputs, mask=None, cache=None):
        if (self.training or inputs.shape[:2] != (1, 1) or mask is not None
                or self.sharding_group is not None or type(cache) not in (ArraysCache, FixedArraysCache)
                or cache.lengths is not None or cache.left_padding is not None
                or cache[0] is None or cache[1] is None):
            return GatedDeltaNet.__call__(self, inputs, mask=mask, cache=cache)
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(1, 1, self.num_v_heads, self.head_v_dim)
        b, a = self.in_proj_b(inputs), self.in_proj_a(inputs)
        q, k, v, next_conv = preprocess(self, qkv, cache[0])
        out, state = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, cache[1], None, use_kernel=True)
        cache[0], cache[1] = next_conv, state
        cache.advance(1)
        self._kernel_calls += 1
        return self.out_proj(self.norm(out, z).reshape(1, 1, -1))


def apply(model):
    """Install before opt.apply; undo compilation before undoing this patch."""
    modules = [layer.linear_attn for layer in model.layers if layer.is_linear]
    if not modules or any(type(module) is not GatedDeltaNet for module in modules):
        raise TypeError("GDN kernel requires unpatched GatedDeltaNet modules")
    if any(module.conv_kernel_size != 4 or "bias" in module.conv1d
           or module.conv1d.weight.dtype not in (mx.float16, mx.float32)
           or module.conv1d.stride != 1 or module.conv1d.padding != 0
           or module.conv1d.dilation != 1 or module.conv1d.groups != module.conv_dim
           or not (0 < module.head_k_dim <= 128 and module.head_k_dim % 4 == 0
                   and 0 < module.head_v_dim <= 128 and module.head_v_dim % 4 == 0)
           for module in modules):
        raise ValueError("GDN kernel requires unbiased depthwise conv4 and head dimensions <=128")
    for module in modules:
        module._kernel_calls = 0
        module.__class__ = _FusedGDN

    def undo():
        for module in modules:
            module.__class__ = GatedDeltaNet
            del module._kernel_calls
        modules.clear()

    return undo
