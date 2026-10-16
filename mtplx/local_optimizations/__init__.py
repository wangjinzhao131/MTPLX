"""Explicit alternatives to the upstream Qwen fusion stack."""

import os
from contextlib import ExitStack

PROFILES = ("qwen35-ar", "qwen27-mtp")


def resolve(config, *, mtp_adapter=None, proj_quant=None, proj_requant=None):
    name = os.environ.get("MTPLX_LOCAL_OPT", "").strip()
    if not name:
        return None
    if name not in PROFILES:
        raise ValueError(f"MTPLX_LOCAL_OPT must be one of {PROFILES}; got {name!r}")
    text = config.get("text_config", config)
    expected_experts = 256 if name == "qwen35-ar" else 0
    if (not str(config.get("model_type", "")).startswith("qwen3_5")
            or text.get("num_experts", 0) != expected_experts):
        raise ValueError(f"{name} requires a Qwen3.5-family model with {expected_experts} experts")
    if (mtp_adapter or proj_quant or proj_requant
            or os.environ.get("MTPLX_PROJ_QUANT") or os.environ.get("MTPLX_PROJ_REQUANT")):
        raise ValueError("Local optimization profiles require the original weights without adapters or requantization")
    return name


def apply(model, name):
    """Install before MTP injection; all mutations belong to this model."""
    with ExitStack() as stack:
        if name == "qwen35-ar":
            from .qwen35.engine import apply as install
            stack.callback(install(model))
        elif name == "qwen27-mtp":
            from .qwen27.engine import apply as install
            from .qwen27.gdn_compile import apply as capture
            stack.callback(install(model))
            stack.callback(capture(model))
        else:
            raise ValueError(f"Unknown local optimization profile: {name!r}")
        return stack.pop_all().close
