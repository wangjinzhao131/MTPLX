"""Compile single-token Qwen3.5/3.6 MoE decode without changing weights."""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import DecoderLayer
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from .router_kernel import _RoutedMoE


def _single_token(x):
    return x.ndim == 3 and x.shape[:2] == (1, 1)


class _CompiledMoE(Qwen3NextSparseMoeBlock):
    def __call__(self, x):
        if not self.training and self.sharding_group is None and _single_token(x):
            self._decode_calls += 1
            return self._decode(x)
        return Qwen3NextSparseMoeBlock.__call__(self, x)


class _CompiledGDNLayer(DecoderLayer):
    def __call__(self, x, mask=None, cache=None):
        if (
            not self.training
            and _single_token(x)
            and mask is None
            and isinstance(cache, ArraysCache)
            and cache.lengths is None
            and cache.left_padding is None
            and cache[0] is not None
            and cache[1] is not None
            and self.linear_attn.sharding_group is None
            and self.mlp.sharding_group is None
        ):
            out, conv_state, recurrent_state = self._decode(x, cache[0], cache[1])
            cache[0], cache[1] = conv_state, recurrent_state
            self._decode_calls += 1
            return out
        return DecoderLayer.__call__(self, x, mask=mask, cache=cache)


def _compile_gdn(layer):
    def forward(x, conv_state, recurrent_state):
        # Scratch state must never retain or replace a caller's cache object.
        cache = ArraysCache(2)
        cache[0], cache[1] = conv_state, recurrent_state
        out = DecoderLayer.__call__(layer, x, cache=cache)
        return out, cache[0], cache[1]

    return mx.compile(forward)


def _compile_moe(block):
    call = type(block).__call__
    return mx.compile(lambda x: call(block, x))


def apply(model, mode="compile"):
    """Install decode compilation on a loaded model; return an idempotent undo.

    Call after loading/fusing weights. Undo before changing model parameters.
    The modified layer or MLP's ``_decode_calls`` counts compiled invocations.
    # shortcut: compilation is B=1/S=1 only; add batched cache support when needed.
    """
    if mode != "compile":
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    layers = model.layers
    if not layers or any(
        type(layer) is not DecoderLayer
        or type(layer.mlp) not in (Qwen3NextSparseMoeBlock, _RoutedMoE)
        for layer in layers
    ):
        raise TypeError("compile requires native Qwen3.5/3.6 layers and native or fused-router MoE")

    changed = []

    def undo():
        for module, original_class in reversed(changed):
            module.__class__ = original_class
            del module._decode
            del module._decode_calls
        changed.clear()

    try:
        for layer in layers:
            module = layer if layer.is_linear else layer.mlp
            compiled = _compile_gdn(layer) if layer.is_linear else _compile_moe(module)
            original_class = type(module)
            module._decode = compiled
            module._decode_calls = 0
            module.__class__ = _CompiledGDNLayer if layer.is_linear else _CompiledMoE
            changed.append((module, original_class))
    except Exception:
        undo()
        raise
    return undo
