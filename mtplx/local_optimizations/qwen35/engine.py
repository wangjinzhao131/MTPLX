"""Install the Qwen3.6 35B-A3B affine6/FP16 decode optimizations in order."""

from contextlib import ExitStack


def apply(model):
    """Return idempotent undo; roll back earlier patches if installation fails."""
    from .fusion import apply as fusion
    from .gdn_kernel import apply as gdn
    from .moe_kernel import apply as moe
    from .down_kernel import apply as down
    from .router_kernel import apply as router
    from .opt import apply as compile_decode
    from .attention_compile import apply as attention
    from .island_compile import apply as islands

    with ExitStack() as installed:
        for installer in (fusion, gdn, moe, down, router, compile_decode, attention, islands):
            installed.callback(installer(model))
        return installed.pop_all().close
