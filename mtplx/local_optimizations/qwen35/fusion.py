"""Single-token projection fusion, preserving the loaded quantization and names.

Output rows are packed once at installation; original parameters become views.
Apply before decode compilation, and undo before loading/changing parameters.
"""

from itertools import accumulate

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import DecoderLayer
from mlx_lm.models.switch_layers import QuantizedSwitchLinear


class _ProjectionGroup:
    def __init__(self, modules):
        first = modules[0]
        projection = type(first).__new__(type(first))
        nn.Module.__init__(projection)
        projection.group_size = first.group_size
        projection.bits = first.bits
        projection.mode = first.mode
        for name in ("weight", "scales", "biases"):
            if first.get(name) is not None:
                setattr(projection, name, mx.concatenate([m[name] for m in modules], axis=-2))
        mx.eval(projection.parameters())
        self.projection = projection
        self.boundaries = list(accumulate(m.weight.shape[-2] for m in modules))[:-1]
        self.parts = len(modules)
        self.calls = 0
        self.clear()

    def clear(self):
        self.input = None
        self.indices = None
        self.outputs = None
        self.served = 0

    def part(self, x, index, indices=None):
        if self.outputs is None or self.input is not x or self.indices is not indices:
            self.clear()
            output = self.projection(x) if indices is None else self.projection(x, indices)
            # Strided split outputs can change downstream reduction kernels.
            self.outputs = tuple(mx.contiguous(y) for y in mx.split(output, self.boundaries, axis=-1))
            self.input, self.indices = x, indices
            self.calls += 1
        output = self.outputs[index]
        self.served |= 1 << index
        if self.served == (1 << self.parts) - 1:
            self.clear()
        return output


class _DenseMember(nn.QuantizedLinear):
    def __call__(self, x):
        if not self.training and x.ndim == 3 and x.shape[:2] == (1, 1):
            return self._fusion.part(x, self._fusion_index)
        return nn.QuantizedLinear.__call__(self, x)


class _ExpertMember(QuantizedSwitchLinear):
    def __call__(self, x, indices, sorted_indices=False):
        if (
            not self.training
            and x.ndim == 5
            and x.shape[:-1] == (1, 1, 1, 1)
            and indices.ndim == 3
            and indices.shape[:2] == (1, 1)
            and not sorted_indices
        ):
            return self._fusion.part(x, self._fusion_index, indices)
        return QuantizedSwitchLinear.__call__(self, x, indices, sorted_indices)


def _validate(modules, expected):
    first = modules[0]
    if any(type(module) is not expected for module in modules):
        raise TypeError("fusion requires unpatched quantized projections")
    signature = first.group_size, first.bits, first.mode
    for module in modules:
        if (module.group_size, module.bits, module.mode) != signature:
            raise ValueError("projection quantization metadata differs within a fusion group")
        if module.mode != "affine" or "bias" in module:
            raise ValueError("fusion requires affine quantization without additive bias")
        for name in ("weight", "scales", "biases"):
            a, b = first.get(name), module.get(name)
            if (a is None) != (b is None):
                raise ValueError("projection quantization arrays differ within a fusion group")
            if a is not None and (
                a.dtype != b.dtype or a.shape[:-2] + a.shape[-1:] != b.shape[:-2] + b.shape[-1:]
            ):
                raise ValueError("projection quantization array layouts differ within a fusion group")


def apply(model):
    """Fuse expert/shared gate-up, GDN inputs and attention QKV; return undo.

    All multi-token/batched calls use the stock projection implementations.
    A one-token initial prompt also takes the single-token path. Parameters keep
    their original names, shapes, and exact packed values, with no full duplicate.
    # shortcut: B=1/S=1 inference only; validate more shapes before extending it.
    """
    if not model.layers or any(not isinstance(layer, DecoderLayer) for layer in model.layers):
        raise TypeError("fusion requires a Qwen3.5/3.6 model")
    groups = []
    for layer in model.layers:
        if layer.mlp.sharding_group is not None:
            raise ValueError("fusion requires an unsharded model")
        groups.append(((layer.mlp.switch_mlp.gate_proj, layer.mlp.switch_mlp.up_proj), QuantizedSwitchLinear))
        groups.append(((layer.mlp.shared_expert.gate_proj, layer.mlp.shared_expert.up_proj), nn.QuantizedLinear))
        if layer.is_linear:
            owner = layer.linear_attn
            if owner.sharding_group is not None:
                raise ValueError("fusion requires an unsharded model")
            names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        else:
            owner = layer.self_attn
            names = ("q_proj", "k_proj", "v_proj")
        groups.append((tuple(getattr(owner, name) for name in names), nn.QuantizedLinear))
    for modules, expected in groups:
        _validate(modules, expected)

    changed = []

    def undo():
        for module, original_class in reversed(changed):
            module.__class__ = original_class
            module._fusion.clear()
            del module._fusion
            del module._fusion_index
        changed.clear()

    try:
        for modules, original_class in groups:
            hub = _ProjectionGroup(modules)
            views = {
                name: mx.split(hub.projection[name], hub.boundaries, axis=-2)
                for name in ("weight", "scales", "biases")
                if hub.projection.get(name) is not None
            }
            for i, module in enumerate(modules):
                for name, values in views.items():
                    setattr(module, name, values[i])
                module._fusion = hub
                module._fusion_index = i
                module.__class__ = _ExpertMember if original_class is QuantizedSwitchLinear else _DenseMember
                changed.append((module, original_class))
            # Releasing replaced expert buffers bounds installation's memory peak.
            mx.clear_cache()
    except Exception:
        undo()
        raise
    return undo
