"""Compile in-tree MTPLX GDN capture with all prefix states intact.

Install after projection patches. Only this model's B1/M1-3 capture forwards
are compiled; native model forward and unrelated models retain their calls.
Both input cache leaves and every output prefix state are explicit graph IO.
M2/M3 use normalized linear_gdn; M1 retains stock recurrence. Commit and
attention stay intact, including the FP32 recurrent state and partial prefixes.
Undo before changing projections or weights. Hooks belong to each GDN instance.
"""

import mlx.nn as nn
from mlx_lm.models.qwen3_5 import DecoderLayer, GatedDeltaNet, Model, Qwen3_5TextModel
from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated

from .decode_fusion import _FusedLinear
from .linear_kernels import _VerifyLinear
from .prefill_gemm import _PrefillLinear

from ..capture import install

_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")

def apply(model):
    """Install the opt-in prefix capture hook; return an idempotent undo.

    Native forward/prefill is unchanged. Only validated instances receive hooks.
    ``_capture_calls`` counts uses and ``_capture_traces`` counts Python traces.
    """
    if (not isinstance(model, Model)
            or type(model.language_model.model) is not Qwen3_5TextModel):
        raise TypeError("GDN compilation requires the native dense Qwen text trunk")
    body = model.language_model.model
    if (model.language_model.args.num_experts != 0 or not body.layers
            or any(type(layer) is not DecoderLayer for layer in body.layers)):
        raise TypeError("GDN compilation requires native dense Qwen decoder layers")
    gdns = [layer.linear_attn for layer in body.layers if layer.is_linear]
    for gdn in gdns:
        if (type(gdn) is not GatedDeltaNet or type(gdn.conv1d) is not nn.Conv1d
                or type(gdn.norm) is not Qwen3NextRMSNormGated
                or any(type(getattr(gdn, name)) not in
                       (nn.QuantizedLinear, _VerifyLinear, _PrefillLinear, _FusedLinear)
                       for name in _PROJECTIONS)):
            raise TypeError("GDN compilation requires native GDN and known quantized projections")
    if not gdns or any(getattr(module, "sharding_group", None) is not None for module in body.modules()):
        raise ValueError("GDN compilation requires an unsharded target with GDN layers")
    return install(gdns)
