"""Shared instance-local GDN capture compilation for the Qwen profiles."""

import importlib
import os

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from ..vendored_arrays_cache import FixedArraysCache

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


def install(gdns):
    """Install on validated GDN instances; return idempotent undo."""
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
