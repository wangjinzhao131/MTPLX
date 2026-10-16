"""Reversible instance-local kernels for the unchanged Qwen3.8-27B weights."""

from contextlib import ExitStack


def apply(model):
    """Install measured kernels, rolling back if a later installer rejects the model."""
    from . import linear_kernels
    from . import prefill_gemm
    from . import decode_fusion
    from . import mlp_compile

    with ExitStack() as stack:
        for install in (linear_kernels.apply, prefill_gemm.apply, decode_fusion.apply, mlp_compile.apply):
            stack.callback(install(model))
        return stack.pop_all().close
