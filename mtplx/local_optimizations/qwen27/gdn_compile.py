"""Compile in-tree MTPLX GDN capture with all prefix states intact.

Install after projection patches. Only this model's B1/M1-3 capture forwards
are compiled; native model forward and unrelated models retain their calls.
Both input cache leaves and every output prefix state are explicit graph IO.
M2/M3 use normalized linear_gdn; M1 retains stock recurrence. Commit and
attention stay intact, including the FP32 recurrent state and partial prefixes.
Undo before changing projections or weights. Hooks belong to each GDN instance.
"""

import importlib
import os

import mlx.core as mx
import mlx.nn as nn
from ...vendored_arrays_cache import FixedArraysCache
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import DecoderLayer, GatedDeltaNet, Model, Qwen3_5TextModel
from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated

from .decode_fusion import _FusedLinear
from .linear_kernels import _VerifyLinear
from .prefill_gemm import _PrefillLinear


_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
# These branches occur inside the traced helper. An enabled alternative
# always falls back, including when its value changes after a graph was traced.
_ENV_BRANCHES = (
    "MTPLX_FUSE_GDN_PROJECTIONS", "MTPLX_LINEAR_CONV1D_CAPTURE",
    "MTPLX_LINEAR_GDN_FROM_CONV",
    "MTPLX_NATIVE_GDN_TAIL", "MTPLX_FUSE_GDN_NORM_GATE", "MTPLX_GDN_OUT_QMV8",
)
_OFF = {"", "0", "false", "no", "off"}


def _capture_module():
    return importlib.import_module("mtplx.gdn_capture")


def _ready(gdn, inputs, mask, cache, capture_backend, module):
    if (gdn.training or inputs.ndim != 3 or inputs.shape[0] != 1
            or inputs.shape[1] not in (1, 2, 3) or inputs.dtype != mx.float16
            or mask is not None or type(cache) not in (ArraysCache, FixedArraysCache)
            or cache.lengths is not None or cache.left_padding is not None
            or cache[0] is None or cache[1] is None
            or cache[0].dtype != mx.float16 or cache[1].dtype != mx.float32
            or (inputs.shape[1] in (2, 3) and
                (gdn.head_k_dim % 32 != 0 or module._linear_gated_delta_kernel is None))
            or gdn.sharding_group is not None
            or any(getattr(gdn, name).training for name in _PROJECTIONS)
            or gdn.conv1d.training or gdn.norm.training
            or mx.default_device() != mx.gpu):
        return False
    return (module.resolve_gdn_capture_backend(capture_backend) == "stock"
            and all(os.environ.get(name, "").lower() in _OFF for name in _ENV_BRANCHES))


def _compile_capture(gdn, original):
    def forward(inputs, conv_state, recurrent_state):
        gdn._capture_traces += 1
        scratch = ArraysCache(2)
        scratch[0], scratch[1] = conv_state, recurrent_state
        backend = "linear_gdn" if inputs.shape[1] in (2, 3) else "stock"
        out, capture = original(gdn, inputs, cache=scratch, capture_backend=backend)
        return out, capture["conv_states"], capture["states"]

    return mx.compile(forward)


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
    module = _capture_module()
    original = module._gdn_forward_with_capture
    if any(hasattr(gdn, "_local_capture") for gdn in gdns):
        raise RuntimeError("GDN capture optimization already installed")
    changed = []

    def dispatch(gdn, inputs, mask=None, cache=None, *, capture_backend=None):
        if not _ready(gdn, inputs, mask, cache, capture_backend, module):
            return original(gdn, inputs, mask=mask, cache=cache, capture_backend=capture_backend)
        out, conv_states, states = gdn._compiled_capture(inputs, cache[0], cache[1])
        # Match stock's cache ownership policy on every call, outside the graph.
        cache[0] = mx.contiguous(conv_states[:, -1, :, :])
        cache[1] = module._maybe_contiguous_authoritative_gdn_leaf(states[:, -1, :, :, :])
        cache.advance(inputs.shape[1])
        gdn._capture_calls += 1
        return out, {"conv_states": conv_states, "states": states}

    def undo():
        for gdn in reversed(changed):
            del gdn._local_capture
            del gdn._compiled_capture
            del gdn._capture_calls
            del gdn._capture_traces
        changed.clear()

    try:
        for gdn in gdns:
            compiled = _compile_capture(gdn, original)
            gdn._compiled_capture = compiled
            gdn._capture_calls = gdn._capture_traces = 0
            gdn._local_capture = dispatch
            changed.append(gdn)
    except Exception:
        undo()
        raise
    return undo
