"""Compile fixed-shape decode between native KV updates and actual-length SDPA.

Installed by engine.apply after per-layer compilation; undo it first.
No KV arrays enter compiled functions, and no context is padded or copied.
"""

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from ...vendored_arrays_cache import FixedArraysCache
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.qwen3_5 import Model
from mlx_lm.models.qwen3_5_moe import Model as MoEModel

from .attention_compile import _CompiledAttentionLayer
from .opt import _CompiledGDNLayer


def _ready(model, cache):
    if cache is None or len(cache) != len(model.layers):
        return False
    offset = None
    for layer, entry in zip(model.layers, cache):
        attention = layer.linear_attn if layer.is_linear else layer.self_attn
        if layer.training or layer.mlp.training or attention.training:
            return False
        if layer.is_linear:
            if (type(entry) not in (ArraysCache, FixedArraysCache) or len(entry.cache) != 2
                    or entry.lengths is not None or entry.left_padding is not None
                    or any(value is None or value.shape[0] != 1 for value in entry.cache)):
                return False
        else:
            if (type(entry) is not KVCache or type(entry.offset) is not int or entry.offset <= 0
                    or entry.keys is None or entry.values is None
                    or entry.keys.shape[0] != 1 or entry.values.shape[0] != 1
                    or min(entry.keys.shape[2], entry.values.shape[2]) < entry.offset):
                return False
            if offset is not None and entry.offset != offset:
                return False
            offset = entry.offset
    return True


def _compile_island(model, start, stop, index, traces):
    layers = model.layers

    def forward(x, offset, *arrays):
        traces[index] += 1
        if start == 0:
            x = model.language_model.model.embed_tokens(x)
            state = arrays
        else:
            output, gate, *state = arrays
            x = layers[start - 1]._attention_after(x, output, gate)
        updated = []
        for i, layer in enumerate(layers[start:stop]):
            x, conv, recurrent = layer._decode(x, state[2 * i], state[2 * i + 1])
            updated.extend((conv, recurrent))
        queries, keys, values, gate = layers[stop]._attention_before(x, offset)
        return x, queries, keys, values, gate, *updated

    return mx.compile(forward)


class _IslandModel(Model):
    def __call__(self, inputs, cache=None, input_embeddings=None, return_hidden=False):
        if return_hidden:
            raise ValueError("qwen35-ar does not provide MTP hidden states")
        # shortcut: populated B1/S1 native caches only; other inputs keep the existing engine.
        if (self.training or self.language_model.training or self.language_model.model.training
                or input_embeddings is not None or inputs.shape != (1, 1)
                or not _ready(self, cache)):
            return Model.__call__(self, inputs, cache=cache, input_embeddings=input_embeddings)
        for start, stop, forward in self._island_blocks:
            entry = cache[stop]
            offset = mx.array(entry.offset, dtype=mx.int32)
            arguments = [inputs, offset] if start == 0 else [x, offset, output, gate]
            arguments.extend(value for item in cache[start:stop] for value in item.cache)
            x, queries, keys, values, gate, *updated = forward(*arguments)
            for i, item in enumerate(cache[start:stop]):
                item[0], item[1] = updated[2 * i:2 * i + 2]
            keys, values = entry.update_and_fetch(keys, values)
            output = scaled_dot_product_attention(
                queries, keys, values, cache=entry, scale=self.layers[stop].self_attn.scale,
                mask=None)
        logits = self._island_tail(x, output, gate)
        self._island_calls += 1
        return logits


class _IslandMoEModel(_IslandModel, MoEModel):
    pass


def apply(model):
    """Return idempotent undo. ``_island_calls`` counts complete decode calls.

    ``_island_traces`` has one counter per compiled island, including the tail;
    changing native KV lengths and scalar RoPE offsets must not retrace them.
    """
    original_class = type(model)
    if original_class not in (Model, MoEModel) or not model.layers or model.layers[-1].is_linear:
        raise TypeError("island compilation requires an engine-patched Qwen3.5/3.6 Model ending in attention")
    if any(type(layer) is not (_CompiledGDNLayer if layer.is_linear else _CompiledAttentionLayer)
           for layer in model.layers):
        raise TypeError("install island compilation after all engine compilation patches")
    if any(layer.mlp.sharding_group is not None or
           (layer.is_linear and layer.linear_attn.sharding_group is not None)
           for layer in model.layers):
        raise ValueError("island compilation requires an unsharded model")
    stops = [i for i, layer in enumerate(model.layers) if not layer.is_linear]
    traces = [0] * (len(stops) + 1)
    blocks = []
    start = 0
    for index, stop in enumerate(stops):
        blocks.append((start, stop, _compile_island(model, start, stop, index, traces)))
        start = stop + 1

    def tail(x, output, gate):
        traces[-1] += 1
        x = model.layers[-1]._attention_after(x, output, gate)
        text = model.language_model
        x = text.model.norm(x)
        return text.model.embed_tokens.as_linear(x) if text.args.tie_word_embeddings else text.lm_head(x)

    compiled_tail = mx.compile(tail)
    model._island_blocks = blocks
    model._island_tail = compiled_tail
    model._island_calls = 0
    model._island_traces = traces
    model.__class__ = _IslandModel if original_class is Model else _IslandMoEModel
    active = True

    def undo():
        nonlocal active
        if active:
            model.__class__ = original_class
            for name in ("_island_blocks", "_island_tail", "_island_calls", "_island_traces"):
                delattr(model, name)
            active = False

    return undo
