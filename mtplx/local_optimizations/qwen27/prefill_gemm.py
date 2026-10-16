"""Opt-in B1 prefill QMM tile installer; native decode and split-K fallback.

Install after linear_kernels and undo in reverse order. Parameter objects and
cache handling are unchanged. Undo before replacing weights or compiling a
different model configuration.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import Model, Qwen3_5TextModel

from . import gemm_kernel
from .linear_kernels import _VerifyLinear


def _ready(module, x):
    if (module.training or x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 32
            or x.dtype != mx.float16 or mx.default_device() != mx.gpu
            or module.mode != "affine"
            or (module.bits, module.group_size) not in ((4, 32), (8, 64))):
        return False
    weight, scales, biases = module.weight, module.scales, module.biases
    if (weight.ndim != 2 or scales.ndim != 2 or biases is None or biases.ndim != 2
            or weight.dtype != mx.uint32 or scales.dtype != mx.float16
            or biases.dtype != mx.float16):
        return False
    m, k, n = x.shape[1], x.shape[2], weight.shape[0]
    return (n >= 32 and n % 32 == 0 and k >= module.group_size
            and k % module.group_size == 0
            and weight.shape[1] == k * module.bits // 32
            and scales.shape == biases.shape == (n, k // module.group_size)
            and ((m + 31) // 32) * ((n + 31) // 32) >= 512)


class _PrefillLinear(nn.QuantizedLinear):
    def __call__(self, x):
        if not _ready(self, x):
            return self._prefill_base_call(self, x)
        output = gemm_kernel.qmm(
            x, self.weight, self.scales, self.biases,
            group_size=self.group_size, bits=self.bits, tile=(64, 32, 32, 2, 2))
        if "bias" in self:
            output = output + self.bias
        return output


def apply(model):
    """Patch native or known decode QLs in the Qwen trunk and LM head only."""
    if (not isinstance(model, Model)
            or type(model.language_model.model) is not Qwen3_5TextModel):
        raise TypeError("prefill GEMM requires a Qwen3.5 text trunk")
    modules = model.language_model.model.modules()
    if "lm_head" in model.language_model:
        modules.append(model.language_model.lm_head)
    targets = [module for module in modules if isinstance(module, nn.QuantizedLinear)]
    # Validate all classes before touching any, so installation is atomic.
    if not targets or any(type(module) not in (nn.QuantizedLinear, _VerifyLinear) for module in targets):
        raise TypeError("prefill GEMM expects native or known decode QuantizedLinear modules")
    for module in targets:
        module._prefill_base_class = type(module)
        module._prefill_base_call = type(module).__call__
        module.__class__ = _PrefillLinear
    active = True

    def undo():
        nonlocal active
        if active:
            for module in reversed(targets):
                module.__class__ = module._prefill_base_class
                del module._prefill_base_class
                del module._prefill_base_call
            active = False

    return undo
