"""Run with ./python -m unittest discover -s tests -p test_gdn_kernel.py."""

from types import SimpleNamespace
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache

from mtplx.local_optimizations.qwen35.gdn_kernel import apply, preprocess
from mtplx.local_optimizations.qwen35.opt import apply as compile_model
from .test_opt import make_model


def reference(gdn, qkv, conv_state):
    combined = mx.concatenate([conv_state, qkv], axis=1)
    conv = nn.silu(gdn.conv1d(combined))
    kd = gdn.num_k_heads * gdn.head_k_dim
    q, k, v = mx.split(conv, [kd, 2 * kd], axis=-1)
    q = q.reshape(1, 1, gdn.num_k_heads, gdn.head_k_dim)
    k = k.reshape(q.shape)
    v = v.reshape(1, 1, gdn.num_v_heads, gdn.head_v_dim)
    scale = gdn.head_k_dim ** -0.5
    return ((scale ** 2) * mx.fast.rms_norm(q, None, 1e-6),
            scale * mx.fast.rms_norm(k, None, 1e-6), v,
            mx.contiguous(combined[:, -3:, :]))


class GDNKernelTests(unittest.TestCase):
    def check_arrays(self, actual, expected, tolerance=0):
        for value, target in zip(actual, expected):
            self.assertEqual(value.shape, target.shape)
            self.assertEqual(value.dtype, target.dtype)
            error = mx.max(mx.abs(value.astype(mx.float32) - target.astype(mx.float32))).item()
            self.assertTrue(mx.allclose(value, target, atol=tolerance, rtol=tolerance).item(),
                            f"max absolute error {error}, allowed {tolerance}")

    def test_preprocessing_real_dimensions_and_roll(self):
        mx.random.seed(117)
        for dtype in (mx.float16, mx.float32):
            with self.subTest(dtype=dtype):
                gdn = SimpleNamespace(num_k_heads=16, num_v_heads=32,
                                      head_k_dim=128, head_v_dim=128)
                gdn.conv1d = nn.Conv1d(8192, 8192, 4, groups=8192, bias=False)
                gdn.conv1d.weight = mx.random.normal((8192, 4, 1)).astype(dtype)
                state = mx.random.normal((1, 3, 8192)).astype(dtype)
                for scale in (0.0, 0.01, 1.0, 10.0):
                    x = (scale * mx.random.normal((1, 1, 8192))).astype(dtype)
                    expected = reference(gdn, x, state)
                    actual = preprocess(gdn, x, state)
                    self.check_arrays(actual, expected, tolerance=0 if dtype == mx.float16 else 1e-6)
                    self.check_arrays(actual[-1:], [mx.concatenate([state[:, 1:], x], axis=1)])
                    state = expected[-1]

    def test_model_states_interleaving_compilation_and_undo(self):
        mx.random.seed(119)
        stock, candidate = make_model(True), make_model(True)
        stock.set_dtype(mx.float16)
        candidate.load_weights(tree_flatten(stock.parameters()))
        original_names = [name for name, _ in tree_flatten(candidate.parameters())]
        undo = apply(candidate)
        undo_compile = compile_model(candidate)
        sessions = [(stock.make_cache(), candidate.make_cache()) for _ in range(2)]
        for index, (ref_cache, opt_cache) in enumerate(sessions):
            token = mx.array([[1, 2, 3 + index]])
            self.check_arrays([candidate(token, opt_cache)], [stock(token, ref_cache)], 1e-4)
        for step in range(8):
            for index, (ref_cache, opt_cache) in enumerate(sessions):
                token = mx.array([[5 + step + index]])
                expected, actual = stock(token, ref_cache), candidate(token, opt_cache)
                self.check_arrays([actual], [expected], 0.002)
                self.check_arrays([mx.argmax(actual, -1)], [mx.argmax(expected, -1)])
                for ref_entry, opt_entry in zip(ref_cache, opt_cache):
                    self.check_arrays(opt_entry.state, ref_entry.state, 0.002)
        self.assertTrue(all(layer.linear_attn._kernel_calls > 0 for layer in candidate.layers
                            if layer.is_linear))
        undo_compile()
        undo()
        undo()
        self.assertEqual(original_names, [name for name, _ in tree_flatten(candidate.parameters())])
        ref_cache, opt_cache = sessions[0]
        token = mx.array([[19]])
        self.check_arrays([candidate(token, opt_cache)], [stock(token, ref_cache)], 0.002)

    def test_fallbacks_and_invalid_shapes(self):
        model = make_model()
        gdn = model.layers[0].linear_attn
        original = type(gdn)
        undo = apply(model)
        for shape, populated, padding, mask in (
                ((1, 1, 64), False, False, False), ((1, 3, 64), True, False, False),
                ((2, 1, 64), True, False, False), ((1, 1, 64), True, True, False),
                ((1, 1, 64), True, False, True)):
            cache = ArraysCache(2)
            if populated:
                mx.eval(original.__call__(gdn, mx.zeros((shape[0], 3, 64)), cache=cache))
            if padding:
                cache.left_padding = mx.array([0])
            actual_mask = mx.ones(shape[:2], dtype=mx.bool_) if mask else None
            mx.eval(gdn(mx.zeros(shape), mask=actual_mask, cache=cache), cache.state)
            self.assertEqual(gdn._kernel_calls, 0)
        with self.assertRaises(TypeError):
            apply(model)
        with self.assertRaises(ValueError):
            preprocess(gdn, mx.zeros((2, 1, gdn.conv_dim)), mx.zeros((1, 3, gdn.conv_dim)))
        undo()


if __name__ == "__main__":
    unittest.main()
