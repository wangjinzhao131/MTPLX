"""Experimental B1/M1-3 FP16 compilation of the target's pure dense MLPs.

Install after all projection patches. No attention or recurrent cache enters
these graphs. Undo before changing weights, projection classes or installers.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import DecoderLayer, Model, Qwen3_5TextModel
from mlx_lm.models.qwen3_next import Qwen3NextMLP

from .decode_fusion import _FusedLinear
from .linear_kernels import _VerifyLinear
from .prefill_gemm import _PrefillLinear


class _CompiledMLP(Qwen3NextMLP):
    def __call__(self, x):
        if (not self.training and x.ndim == 3 and x.shape[0] == 1
                and x.shape[1] in (1, 2, 3) and x.dtype == mx.float16
                and not any(projection.training for projection in
                            (self.gate_proj, self.up_proj, self.down_proj))
                and mx.default_device() == mx.gpu):
            self._compiled_calls += 1
            return self._compiled_mlp(x)
        return Qwen3NextMLP.__call__(self, x)


def _compile_mlp(block):
    call = Qwen3NextMLP.__call__

    def forward(x):
        block._compiled_traces += 1
        return call(block, x)

    return mx.compile(forward)


def apply(model):
    """Compile native dense target MLPs; return an idempotent undo.

    ``_compiled_calls`` counts eligible forwards; ``_compiled_traces`` counts
    Python traces. M>=4, batch, training and FP32 retain the native MLP call
    with its existing projection wrappers. The MTP branch is excluded.
    """
    if (not isinstance(model, Model)
            or type(model.language_model.model) is not Qwen3_5TextModel):
        raise TypeError("MLP compile requires the native dense Qwen text trunk")
    body = model.language_model.model
    if (model.language_model.args.num_experts != 0 or not body.layers
            or any(type(layer) is not DecoderLayer or type(layer.mlp) is not Qwen3NextMLP
                   for layer in body.layers)):
        raise TypeError("MLP compile requires native dense Qwen decoder layers and MLPs")
    for layer in body.layers:
        for projection in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
            if type(projection) not in (nn.QuantizedLinear, _VerifyLinear, _PrefillLinear, _FusedLinear):
                raise TypeError("MLP compile requires native or known quantized projections")
    if any(getattr(module, "sharding_group", None) is not None for module in body.modules()):
        raise ValueError("MLP compile requires an unsharded target")
    changed = []

    def undo():
        for block in reversed(changed):
            block.__class__ = Qwen3NextMLP
            del block._compiled_mlp
            del block._compiled_calls
            del block._compiled_traces
        changed.clear()

    try:
        for layer in body.layers:
            block = layer.mlp
            compiled = _compile_mlp(block)
            block._compiled_mlp = compiled
            block._compiled_calls = block._compiled_traces = 0
            block.__class__ = _CompiledMLP
            changed.append(block)
    except Exception:
        undo()
        raise
    return undo
