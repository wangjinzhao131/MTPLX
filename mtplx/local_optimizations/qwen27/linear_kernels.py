"""Opt-in exact small-batch GEMV selection for the dense Qwen target only."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import DecoderLayer, Model, Qwen3_5TextModel

from .verify_kernel import qmv


class _VerifyLinear(nn.QuantizedLinear):
    def __call__(self, x):
        if (not self.training and x.ndim >= 2 and x.dtype == mx.float16
                and all(size == 1 for size in x.shape[:-2]) and x.shape[-1] > 0
                and self.mode == "affine"):
            m = x.size // x.shape[-1]
            variant = None
            if (self.bits, self.group_size) == (4, 32) and m in (2, 3):
                variant = "paired4" if m == 2 else "triplet4"
            elif (self.bits, self.group_size) == (8, 64) and m in (2, 3, 4):
                variant = "shared"
            if variant is not None:
                out = qmv(x, self.weight, self.scales, self.biases,
                          bits=self.bits, group_size=self.group_size, variant=variant)
                if "bias" in self:
                    out = out + self.bias
                return out
        return nn.QuantizedLinear.__call__(self, x)


def apply(model):
    """Patch eligible target projections without changing parameters or MTP.

    Native M=1 decode, prefill, training and unsupported inputs stay native.
    Install before prefill wrappers/compilation; undo those first, then call
    the returned idempotent undo before replacing weights. MTPLX's outer Model
    subclass is accepted, while target body and projection classes are exact.
    """
    if not isinstance(model, Model) or type(model.language_model.model) is not Qwen3_5TextModel:
        raise TypeError("verify kernels require the native dense Qwen3.5 target body")
    body = model.language_model.model
    if (model.language_model.args.num_experts != 0 or not body.layers
            or any(type(layer) is not DecoderLayer for layer in body.layers)):
        raise TypeError("verify kernels require uncompiled dense Qwen decoder layers")
    modules = list(body.named_modules())
    if not model.language_model.args.tie_word_embeddings:
        modules.append(("lm_head", model.language_model.lm_head))
    changed = []
    for name, module in modules:
        if getattr(module, "sharding_group", None) is not None:
            raise ValueError("verify kernels require an unsharded target")
        if not isinstance(module, nn.QuantizedLinear):
            continue
        if type(module) is not nn.QuantizedLinear:
            raise TypeError(f"verify kernels require unpatched QuantizedLinear: {name}")
        if (module.bits, module.group_size, module.mode) not in ((4, 32, "affine"), (8, 64, "affine")):
            continue
        weight, scales, biases = module.weight, module.scales, module.get("biases")
        if weight.ndim != 2 or scales.ndim != 2 or biases is None or biases.ndim != 2:
            raise ValueError(f"malformed affine projection: {name}")
        n, packed_k = weight.shape
        k = packed_k * 32 // module.bits
        if scales.shape != (n, k // module.group_size) or biases.shape != scales.shape:
            raise ValueError(f"inconsistent affine projection shapes: {name}")
        if (weight.dtype != mx.uint32 or scales.dtype != mx.float16 or biases.dtype != mx.float16
                or n < 8 or n % 8 or k < 2048 // module.bits or k % (2048 // module.bits)):
            continue
        changed.append(module)
    # Validation finishes before any class is replaced.
    for module in changed:
        module.__class__ = _VerifyLinear

    def undo():
        for module in reversed(changed):
            module.__class__ = nn.QuantizedLinear
        changed.clear()

    return undo
