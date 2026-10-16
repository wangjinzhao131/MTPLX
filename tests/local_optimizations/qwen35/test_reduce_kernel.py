"""GPU checks: ./python -m unittest discover -s tests -p test_reduce_kernel.py."""

import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5_moe import Model, ModelArgs
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from mtplx.local_optimizations.qwen35 import engine
from mtplx.local_optimizations.qwen35.down_kernel import down
from mtplx.local_optimizations.qwen35.moe_kernel import _FusedSwitchGLU
from mtplx.local_optimizations.qwen35.reduce_kernel import weighted_down
from .test_opt import make_model


def baseline(x, indices, scores, weight, scales, biases):
    y = down(x, indices, weight, scales, biases, results_per_simd=4, simdgroups=4)
    return (y.squeeze(-2) * scores[..., None]).sum(axis=-2)


class ReduceKernelTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(307)
        self.arrays = mx.quantize(mx.random.normal((8, 64, 512), dtype=mx.float16) / 16,
                                  group_size=64, bits=6)
        mx.eval(*self.arrays)
        self.indices = mx.array([[[7, 0, 3, 3, 1, 5, 2, 6]]], dtype=mx.uint32)
        self.scores = mx.array([[[.01, .02, .04, .07, .11, .17, .23, .35]]], dtype=mx.float16)
        mx.eval(self.indices, self.scores)

    def assert_exact(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, mx.float16)
        maximum = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
        self.assertTrue(mx.array_equal(actual, expected).item(), f"max error={maximum}")

    def test_native_projection_weighting_and_sum_order(self):
        compiled = mx.compile(lambda x, i, s: weighted_down(x, i, s, *self.arrays))
        compiled_baseline = mx.compile(lambda x, i, s: baseline(x, i, s, *self.arrays))
        for magnitude in (0.0, 0.25, 1.0, 4.0, 32.0):
            x = mx.random.normal((1, 1, 8, 1, 512), dtype=mx.float16) * magnitude
            mx.eval(x)
            native = mx.gather_qmm(x, *self.arrays, rhs_indices=self.indices,
                                   transpose=True, group_size=64, bits=6, sorted_indices=False)
            expected = (native.squeeze(-2) * self.scores[..., None]).sum(axis=-2)
            with self.subTest(magnitude=magnitude):
                self.assert_exact(baseline(x, self.indices, self.scores, *self.arrays), expected)
                self.assert_exact(compiled_baseline(x, self.indices, self.scores), expected)
                self.assert_exact(weighted_down(x, self.indices, self.scores, *self.arrays), expected)
                self.assert_exact(compiled(x, self.indices, self.scores), expected)

    def test_zero_one_hot_and_signed_scores(self):
        x = mx.random.normal((1, 1, 8, 1, 512), dtype=mx.float16)
        for values in ([0] * 8, [1] + [0] * 7, [0] * 7 + [1],
                       [0.125] * 8, [16, -.125, 0, 2, -.5, .25, -8, .03125]):
            scores = mx.array([[values]], dtype=mx.float16)
            self.assert_exact(weighted_down(x, self.indices, scores, *self.arrays),
                              baseline(x, self.indices, scores, *self.arrays))

    def test_noncontiguous_inputs_and_invalid_experts(self):
        x = mx.random.normal((1, 1, 8, 1, 1024), dtype=mx.float16)[..., ::2]
        indices = mx.repeat(self.indices, 2, axis=-1)[..., ::2]
        scores = mx.repeat(self.scores, 2, axis=-1)[..., ::2]
        compiled = mx.compile(lambda a, i, s: weighted_down(a, i, s, *self.arrays))
        expected = baseline(x, indices, scores, *self.arrays)
        self.assert_exact(weighted_down(x, indices, scores, *self.arrays), expected)
        self.assert_exact(compiled(x, indices, scores), expected)
        for slot, value in ((0, 8), (7, 4294967295)):
            invalid = mx.array([[[value if i == slot else int(self.indices[0, 0, i].item())
                                   for i in range(8)]]], dtype=mx.uint32)
            actual = compiled(x, invalid, scores)
            self.assertTrue(mx.all(mx.isnan(actual)).item())

    def test_validation(self):
        x = mx.zeros((1, 1, 8, 1, 512), dtype=mx.float16)
        weight, scales, biases = self.arrays
        args = [x, self.indices, self.scores, weight, scales, biases]
        for position, bad in ((0, x[..., :256]), (0, x.astype(mx.float32)),
                              (1, self.indices.astype(mx.int32)), (1, self.indices[..., :7]),
                              (2, self.scores.astype(mx.float32)), (2, self.scores[..., :7]),
                              (3, weight[..., :95]), (3, weight[:, :63]), (3, weight[:, :12]),
                              (4, scales.astype(mx.float32)), (5, biases[..., :7])):
            changed = args.copy()
            changed[position] = bad
            with self.assertRaises(ValueError):
                weighted_down(*changed)


class _UnweightedSwitch(_FusedSwitchGLU):
    """Keep the previous down-then-reduce route through the exact-class guard."""


class ReduceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(313)
        args = make_model().args
        config = {**args.text_config, "hidden_size": 256, "head_dim": 256,
                  "num_hidden_layers": 2, "num_experts": 256, "num_experts_per_tok": 8,
                  "moe_intermediate_size": 512}

        def create_model():
            model = Model(ModelArgs(args.model_type, config))
            model.set_dtype(mx.float16)
            nn.quantize(model, group_size=64, bits=6,
                        class_predicate=lambda path, module: hasattr(module, "to_quantized")
                        and model.quant_predicate(path, module))
            model.eval()
            return model

        cls.reference, cls.candidate = create_model(), create_model()
        cls.candidate.load_weights(tree_flatten(cls.reference.parameters()))
        mx.eval(cls.reference.parameters(), cls.candidate.parameters())

    def setUp(self):
        self.reference.eval()
        self.candidate.eval()
        self.undo_ref = engine.apply(self.reference)
        self.addCleanup(self.undo_ref)
        self.undo_opt = engine.apply(self.candidate)
        self.addCleanup(self.undo_opt)
        # Change only route reduction, before either compiled model is traced.
        for layer in self.reference.layers:
            layer.mlp.switch_mlp.__class__ = _UnweightedSwitch

    def assert_exact(self, actual, expected, label=""):
        self.assertEqual(actual.shape, expected.shape, label)
        self.assertEqual(actual.dtype, expected.dtype, label)
        if not mx.array_equal(actual, expected).item():
            error = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
            self.fail(f"{label}: max error={error}")

    def assert_caches(self, actual, expected):
        for layer, (left, right) in enumerate(zip(actual, expected)):
            self.assertIs(type(left), type(right))
            if type(left) is KVCache:
                self.assertEqual(left.offset, right.offset)
            for slot, (a, b) in enumerate(zip(left.state, right.state)):
                self.assert_exact(a, b, f"layer={layer} cache={slot}")

    def test_full_stack_independent_caches_and_undo(self):
        names = [name for name, _ in tree_flatten(self.candidate.parameters())]
        classes = [(type(layer.mlp.switch_mlp), type(layer.mlp.switch_mlp.down_proj))
                   for layer in self.candidate.layers]
        sessions = [(self.reference.make_cache(), self.candidate.make_cache()) for _ in range(2)]
        with patch("mtplx.local_optimizations.qwen35.reduce_kernel.weighted_down", wraps=weighted_down) as kernel:
            for session, (ref_cache, opt_cache) in enumerate(sessions):
                prompt = mx.array([[1, 2, 3] + ([4] if session else [])])
                self.assert_exact(self.candidate(prompt, opt_cache), self.reference(prompt, ref_cache), "prefill")
                self.assert_caches(opt_cache, ref_cache)
            self.assertEqual(kernel.call_count, len(self.candidate.layers))
            tokens = [mx.array([[5]], dtype=mx.uint32), mx.array([[6]], dtype=mx.uint32)]
            for step in range(6):
                for session, (ref_cache, opt_cache) in enumerate(sessions):
                    expected = self.reference(tokens[session], ref_cache)
                    actual = self.candidate(tokens[session], opt_cache)
                    self.assert_exact(actual, expected, f"decode={step} session={session}")
                    self.assert_exact(mx.argmax(actual, -1), mx.argmax(expected, -1), "greedy")
                    self.assert_caches(opt_cache, ref_cache)
                    tokens[session] = mx.argmax(expected[:, -1, :], -1)[:, None]
            self.assertEqual(kernel.call_count, 2 * len(self.candidate.layers))
        self.assertEqual(self.candidate._island_calls, 12)
        self.assertEqual(self.candidate._island_traces, [1, 1])
        self.assertEqual(names, [name for name, _ in tree_flatten(self.candidate.parameters())])
        self.assertEqual(classes, [(type(layer.mlp.switch_mlp), type(layer.mlp.switch_mlp.down_proj))
                                   for layer in self.candidate.layers])
        self.undo_opt()
        self.undo_opt()
        self.undo_ref()
        self.undo_ref()
        self.assertIs(type(self.candidate), Model)
        self.assertTrue(all(type(layer.mlp.switch_mlp.down_proj) is QuantizedSwitchLinear
                            for layer in self.candidate.layers))
        ref_cache, opt_cache = sessions[0]
        self.assert_exact(self.candidate(mx.array([[11]]), opt_cache),
                          self.reference(mx.array([[11]]), ref_cache), "after undo")
        self.assert_caches(opt_cache, ref_cache)

    def test_weighted_fallbacks(self):
        switch = self.candidate.layers[0].mlp.switch_mlp
        cases = [((1, 1, 8), mx.float16, None), ((1, 3, 8), mx.float16, None),
                 ((2, 1, 8), mx.float16, None), ((1, 1, 2), mx.float16, None),
                 ((1, 1, 8), mx.float32, None), ((1, 1, 8), mx.float16, "all"),
                 ((1, 1, 8), mx.float16, "gate"), ((1, 1, 8), mx.float16, "down")]
        for position, (shape, dtype, training) in enumerate(cases):
            with self.subTest(shape=shape, dtype=dtype, training=training):
                switch.eval()
                if training == "all":
                    switch.train()
                elif training == "gate":
                    switch.gate_proj.train()
                elif training == "down":
                    switch.down_proj.train()
                x = mx.random.normal((*shape[:2], 256)).astype(dtype)
                indices = mx.broadcast_to(mx.arange(shape[-1], dtype=mx.uint32), shape)
                scores = mx.full(shape, 1 / shape[-1], dtype=dtype)
                expected = (switch(x, indices) * scores[..., None]).sum(axis=-2)
                with patch("mtplx.local_optimizations.qwen35.reduce_kernel.weighted_down", wraps=weighted_down) as kernel:
                    self.assert_exact(switch.weighted(x, indices, scores), expected, "weighted fallback")
                    self.assertEqual(kernel.call_count, int(position in (0, 1)))
        switch.eval()
        # A native down projection is a supported fallback, including after undo.
        down_projection = switch.down_proj
        previous = type(down_projection)
        try:
            down_projection.__class__ = QuantizedSwitchLinear
            x = mx.zeros((1, 1, 256), dtype=mx.float16)
            indices = mx.arange(8, dtype=mx.uint32).reshape(1, 1, 8)
            scores = mx.full((1, 1, 8), .125, dtype=mx.float16)
            expected = (switch(x, indices) * scores[..., None]).sum(axis=-2)
            with patch("mtplx.local_optimizations.qwen35.reduce_kernel.weighted_down", wraps=weighted_down) as kernel:
                self.assert_exact(switch.weighted(x, indices, scores), expected)
                self.assertEqual(kernel.call_count, 0)
        finally:
            down_projection.__class__ = previous


if __name__ == "__main__":
    unittest.main()
