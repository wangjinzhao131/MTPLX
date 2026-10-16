"""Compile attention around native KV updates and actual-length SDPA.

The layer dispatcher specializes S1; MTP islands also reuse the S2/S3 parts.

Install after opt.apply with final weights; undo in reverse before changing them.
The two compiled calls have fixed shapes. The RoPE offset remains a tensor input.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer
from mlx_lm.models.qwen3_next import Qwen3NextAttention

from .opt import _CompiledMoE


def _compile_parts(layer):
    attention = layer.self_attn

    def before(x, offset):
        length = x.shape[1]
        x = layer.input_layernorm(x)
        queries, gate = mx.split(
            attention.q_proj(x).reshape(1, length, attention.num_attention_heads, -1),
            2, axis=-1)
        gate = gate.reshape(1, length, -1)
        keys, values = attention.k_proj(x), attention.v_proj(x)
        queries = attention.q_norm(queries).transpose(0, 2, 1, 3)
        keys = attention.k_norm(keys.reshape(1, length, attention.num_key_value_heads, -1))
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(1, length, attention.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        return attention.rope(queries, offset=offset), attention.rope(keys, offset=offset), values, gate

    def after(x, output, gate):
        output = output.transpose(0, 2, 1, 3).reshape(1, x.shape[1], -1)
        h = x + attention.o_proj(output * mx.sigmoid(gate))
        return h + layer.mlp(layer.post_attention_layernorm(h))

    return mx.compile(before), mx.compile(after)


class _CompiledAttentionLayer(DecoderLayer):
    def __call__(self, x, mask=None, cache=None):
        # shortcut: native populated B1/S1 KV only; other cache kinds keep stock code.
        if (self.training or self.self_attn.training or self.mlp.training
                or self.mlp.sharding_group is not None
                or x.ndim != 3 or x.shape[:2] != (1, 1)
                or x.dtype not in (mx.float16, mx.float32) or mask is not None
                or type(cache) is not KVCache or cache.offset <= 0
                or cache.keys is None or cache.values is None):
            return DecoderLayer.__call__(self, x, mask=mask, cache=cache)
        queries, keys, values, gate = self._attention_before(
            x, mx.array(cache.offset, dtype=mx.int32))
        keys, values = cache.update_and_fetch(keys, values)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.self_attn.scale, mask=None)
        self._attention_calls += 1
        return self._attention_after(x, output, gate)


def apply(model):
    """Install after opt.apply; return idempotent undo without touching parameters.

    ``_attention_calls`` counts specialized layer calls. The nested compiled MLP's
    Python counter advances on tracing rather than on every replay of the post block.
    """
    layers = [layer for layer in model.layers if not layer.is_linear]
    if not layers or any(type(layer) is not DecoderLayer
                         or type(layer.self_attn) is not Qwen3NextAttention
                         or type(layer.mlp) is not _CompiledMoE for layer in layers):
        raise TypeError("attention compilation requires native full-attention layers after opt.apply")
    if any(layer.mlp.sharding_group is not None or type(layer.self_attn.rope) is not nn.RoPE
           for layer in layers):
        raise ValueError("attention compilation requires unsharded attention with native RoPE")
    changed = []

    def undo():
        for layer in reversed(changed):
            layer.__class__ = DecoderLayer
            del layer._attention_before
            del layer._attention_after
            del layer._attention_calls
        changed.clear()

    try:
        for layer in layers:
            layer._attention_before, layer._attention_after = _compile_parts(layer)
            layer._attention_calls = 0
            layer.__class__ = _CompiledAttentionLayer
            changed.append(layer)
    except Exception:
        undo()
        raise
    return undo
