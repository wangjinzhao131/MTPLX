"""Multirow target head keeps native arithmetic and unsupported-head behavior."""
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import pytest

from mtplx.local_optimizations.qwen35 import head_kernel


def make_head(k=256, n=64, **kwargs):
    head = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=6, **kwargs)
    head.set_dtype(mx.float16)
    head.eval()
    return head


@pytest.mark.parametrize('length', [2, 3])
@pytest.mark.parametrize('width', [256, 2048])
def test_exact_native_arithmetic_and_parameters(length, width):
    mx.random.seed(391 + length + width)
    head = make_head(width, 64)
    # Materialize random fixture quantization before tracing, like loaded weights.
    mx.eval(head.parameters())
    original = dict(tree_flatten(head.parameters()))
    compiled = mx.compile(lambda x: head_kernel.forward(head, x))
    for magnitude in (0, .001, .1, 1, 10):
        x = mx.random.normal((1, 2 * length, width), dtype=mx.float16)[:, ::2] * magnitude
        expected = head(x)
        for actual in (head_kernel.forward(head, x), compiled(x)):
            assert mx.array_equal(actual, expected).item()
    assert type(head) is nn.QuantizedLinear
    assert all(original[k] is v for k, v in tree_flatten(head.parameters()))


@pytest.mark.parametrize('case', ['single', 'prefill', 'batch', 'fp32', 'width',
                                  'rows', 'bits', 'group', 'bias', 'training', 'subclass'])
def test_native_fallback(case):
    shape = (1, 2, 256)
    head = make_head()
    if case == 'single': shape = (1, 1, 256)
    elif case == 'prefill': shape = (1, 4, 256)
    elif case == 'batch': shape = (2, 2, 256)
    elif case == 'width':
        shape = (1, 2, 64)
        head = make_head(64)
    elif case == 'rows': head = make_head(n=9)
    elif case in ('bits', 'group', 'bias'):
        head = nn.QuantizedLinear(256, 64, bits=4 if case == 'bits' else 6,
                                 group_size=32 if case == 'group' else 64, bias=case == 'bias')
        head.set_dtype(mx.float16)
        head.eval()
    elif case == 'training': head.train()
    elif case == 'subclass':
        class OffsetHead(nn.QuantizedLinear):
            def __call__(self, x): return super().__call__(x) + 1
        head.__class__ = OffsetHead
    x = mx.random.normal(shape, dtype=mx.float32 if case == 'fp32' else mx.float16)
    with patch.object(head_kernel, '_kernel', side_effect=AssertionError('must fall back')):
        assert mx.array_equal(head_kernel.forward(head, x), head(x)).item()
