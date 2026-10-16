"""Experimental B1/M1-3 dense projection packing for the unchanged target.

Install after linear_kernels and prefill_gemm, before compilation. Only output
rows are concatenated: no dequantization or requantization occurs. Original
parameters become packed views, so their names and values survive undo but
their Python identities may change. Undo in reverse order before changing
weights. The measured decode path is installed by engine.py.
"""

from itertools import accumulate

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import DecoderLayer, Model, Qwen3_5TextModel

from .linear_kernels import _VerifyLinear
from .prefill_gemm import _PrefillLinear
from .verify_kernel import qmv


class _ProjectionGroup:
    def __init__(self, modules):
        first = modules[0]
        projection = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
        nn.Module.__init__(projection)
        projection.group_size, projection.bits, projection.mode = (
            first.group_size, first.bits, first.mode)
        for name in ("weight", "scales", "biases"):
            setattr(projection, name, mx.concatenate([module[name] for module in modules], axis=0))
        projection.eval()
        mx.eval(projection.parameters())
        self.projection = projection
        self.widths = tuple(module.weight.shape[0] for module in modules)
        self.boundaries = list(accumulate(self.widths))[:-1]
        self.parts = len(modules)
        self.partitioned_outputs = False
        self.calls = 0
        self.clear()

    def clear(self):
        self.input = None
        self.outputs = None
        self.served = 0

    def part(self, x, index):
        if (self.outputs is None or self.input is not x
                or self.served & (1 << index)):
            self.clear()
            p = self.projection
            if (self.partitioned_outputs and x.shape[1] == 2
                    and (p.bits, p.group_size) == (4, 32) and self.parts in (3, 4)):
                self.outputs = qmv(x, p.weight, p.scales, p.biases,
                                   bits=p.bits, group_size=p.group_size,
                                   variant="paired4", output_splits=self.widths)
            else:
                if x.shape[1] in (2, 3):
                    output = qmv(x, p.weight, p.scales, p.biases,
                                 bits=p.bits, group_size=p.group_size,
                                 variant=("paired4" if x.shape[1] == 2 else "triplet4")
                                 if p.bits == 4 else "shared")
                else:
                    output = nn.QuantizedLinear.__call__(p, x)
                # M2-3 split views have a larger row stride; keep native downstream
                # reduction layouts by materializing contiguous projection outputs.
                self.outputs = tuple(mx.contiguous(y) for y in mx.split(output, self.boundaries, axis=-1))
            self.input = x
            self.calls += 1
        output = self.outputs[index]
        self.served |= 1 << index
        if self.served == (1 << self.parts) - 1:
            self.clear()
        return output


class _FusedLinear(nn.QuantizedLinear):
    def __call__(self, x):
        if (not self.training and x.ndim == 3 and x.shape[0] == 1
                and x.shape[1] in (1, 2, 3) and x.dtype == mx.float16
                and x.shape[2] == self.weight.shape[1] * 32 // self.bits
                and mx.default_device() == mx.gpu):
            return self._fusion.part(x, self._fusion_index)
        self._fusion.clear()
        return self._fusion_base_call(self, x)


def _eligible(modules):
    if any(type(module) not in (nn.QuantizedLinear, _VerifyLinear, _PrefillLinear)
           for module in modules):
        raise TypeError("decode fusion expects native or known decode/prefill projections")
    first = modules[0]
    signature = first.bits, first.group_size, first.mode
    if any((module.bits, module.group_size, module.mode) != signature for module in modules):
        raise ValueError("decode fusion requires matching quantization within each group")
    if signature not in ((4, 32, "affine"), (8, 64, "affine")):
        return False
    ready = True
    first_k = first.weight.shape[-1] * 32 // first.bits
    for module in modules:
        weight, scales, biases = module.weight, module.scales, module.get("biases")
        if (weight.ndim != 2 or scales.ndim != 2 or biases is None or biases.ndim != 2):
            raise ValueError("decode fusion requires 2D affine projection arrays")
        n, packed_k = weight.shape
        k = packed_k * 32 // module.bits
        if (k != first_k or k % module.group_size
                or scales.shape != (n, k // module.group_size) or biases.shape != scales.shape):
            raise ValueError("decode fusion projection shapes do not match")
        # Each original projection must use native qmv_fast too. Packing small
        # output widths into a fast tile would otherwise change its dot order.
        ready &= ("bias" not in module and weight.dtype == mx.uint32
                  and scales.dtype == biases.dtype == mx.float16
                  and n >= 8 and n % 8 == 0
                  and k >= 2048 // module.bits and k % (2048 // module.bits) == 0)
    return ready


def apply(model):
    """Pack target GDN inputs, attention QKV and dense MLP gate/up; return undo.

    Unsupported shapes/formats keep their existing methods. The MTP head and
    target LM head are not packed. M>=4, batch, training and FP32 calls retain
    each member's captured method, including prior decode/prefill installers.
    """
    if (not isinstance(model, Model)
            or type(model.language_model.model) is not Qwen3_5TextModel):
        raise TypeError("decode fusion requires the native dense Qwen text trunk")
    body = model.language_model.model
    if (model.language_model.args.num_experts != 0 or not body.layers
            or any(type(layer) is not DecoderLayer for layer in body.layers)):
        raise TypeError("decode fusion requires uncompiled dense Qwen decoder layers")
    if any(getattr(module, "sharding_group", None) is not None for module in body.modules()):
        raise ValueError("decode fusion requires an unsharded target")
    groups = []
    for layer in body.layers:
        groups.append((layer.mlp.gate_proj, layer.mlp.up_proj))
        if layer.is_linear:
            owner, names = layer.linear_attn, ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        else:
            owner, names = layer.self_attn, ("q_proj", "k_proj", "v_proj")
        groups.append(tuple(getattr(owner, name) for name in names))
    # Finish validation before replacing parameters or classes in any group.
    groups = [modules for modules in groups if _eligible(modules)]
    changed = []

    def undo():
        for module in reversed(changed):
            module._fusion.clear()
            module.__class__ = module._fusion_base_class
            del module._fusion
            del module._fusion_index
            del module._fusion_base_class
            del module._fusion_base_call
        changed.clear()

    try:
        for modules in groups:
            hub = _ProjectionGroup(modules)
            views = {name: mx.split(hub.projection[name], hub.boundaries, axis=0)
                     for name in ("weight", "scales", "biases")}
            for index, module in enumerate(modules):
                for name, values in views.items():
                    setattr(module, name, values[index])
                module._fusion_base_class = type(module)
                module._fusion_base_call = type(module).__call__
                module._fusion = hub
                module._fusion_index = index
                module.__class__ = _FusedLinear
                changed.append(module)
            # Old allocations are no longer needed once members own packed views.
            mx.clear_cache()
    except Exception:
        undo()
        raise
    return undo
