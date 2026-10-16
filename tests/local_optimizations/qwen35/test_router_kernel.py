"""Run with ./python -m unittest discover -s tests -p test_router_kernel.py."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import Model, ModelArgs, TextModelArgs
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from mtplx.local_optimizations.qwen35.fusion import apply as fuse_projections
from mtplx.local_optimizations.qwen35.gdn_kernel import apply as fuse_gdn
from mtplx.local_optimizations.qwen35.moe_kernel import apply as fuse_moe
from mtplx.local_optimizations.qwen35.opt import apply as compile_decode
from mtplx.local_optimizations.qwen35.router_kernel import _RoutedMoE, apply, moe_forward, route
from .test_opt import make_model


def reference(logits):
    probabilities = mx.softmax(logits, axis=-1, precise=True)
    indices = mx.argpartition(probabilities, kth=-8, axis=-1)[..., -8:]
    scores = mx.take_along_axis(probabilities, indices, axis=-1)
    return indices, scores / scores.sum(axis=-1, keepdims=True)


class RouterKernelTests(unittest.TestCase):
    def assert_exact(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        error = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
        self.assertTrue(mx.array_equal(actual, expected).item(), f"max absolute error {error}")

    def check_route(self, logits):
        expected = reference(logits)
        actual = route(logits)
        compiled = mx.compile(route)(logits)
        for result in (actual, compiled):
            self.assertEqual(result[0].dtype, expected[0].dtype)
            self.assertEqual(result[1].dtype, expected[1].dtype)
            self.assertTrue(mx.array_equal(result[0], expected[0]).item(), "expert order differs")
            error = mx.max(mx.abs(result[1].astype(mx.float32) - expected[1].astype(mx.float32))).item()
            self.assertTrue(mx.array_equal(result[1], expected[1], equal_nan=True).item(),
                            f"normalized scores differ, max absolute error {error}")

    def test_random_logits_and_rounding_ties(self):
        mx.random.seed(217)
        for dtype in (mx.float16, mx.float32):
            for scale in (0.00001, 0.01, 1.0, 10.0, 100.0):
                with self.subTest(dtype=dtype, scale=scale):
                    for _ in range(8):
                        self.check_route((mx.random.normal((1, 1, 256)) * scale).astype(dtype))

    def test_ties_extremes_and_nonfinite(self):
        for dtype in (mx.float16, mx.float32):
            for constant in (0, -65504, 65504, float("inf"), float("-inf"), float("nan")):
                with self.subTest(dtype=dtype, constant=constant):
                    logits = mx.full((1, 1, 256), constant, dtype=dtype)
                    self.check_route(logits)
                    self.assertEqual(route(logits)[0].reshape(-1).tolist(), list(range(248, 256)))
            for special in (float("inf"), float("-inf"), float("nan")):
                self.check_route(mx.array([0.0] * 255 + [special], dtype=dtype).reshape(1, 1, 256))
            self.check_route(mx.array([float(i % 5) for i in range(256)], dtype=dtype).reshape(1, 1, 256))

    def test_moe_forward_and_fallback(self):
        args = TextModelArgs.from_dict({"model_type": "qwen3_5_moe_text", "hidden_size": 32,
                                       "num_experts": 256, "num_experts_per_tok": 8,
                                       "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                                       "norm_topk_prob": True})
        block = Qwen3NextSparseMoeBlock(args)
        block.set_dtype(mx.float16)
        block.eval()
        for shape in ((1, 1, 32), (1, 3, 32), (2, 1, 32), (1, 1, 1, 32)):
            x = mx.random.normal(shape).astype(mx.float16)
            self.assertTrue(mx.array_equal(moe_forward(block, x), block(x)).item())
        block.train()
        x = mx.zeros((1, 1, 32), dtype=mx.float16)
        self.assertTrue(mx.array_equal(moe_forward(block, x), block(x)).item())
        for value in (mx.zeros((256,)), mx.zeros((2, 1, 256)),
                      mx.zeros((1, 1, 256), dtype=mx.bfloat16)):
            with self.assertRaises(ValueError):
                route(value)

    def test_full_stack_independent_caches_compile_and_undo(self):
        mx.random.seed(239)
        args = make_model().args

        def create_model():
            model = Model(ModelArgs(args.model_type, {**args.text_config, "hidden_size": 256,
                                                     "num_experts": 256, "num_experts_per_tok": 8}))
            model.set_dtype(mx.float16)
            nn.quantize(model, group_size=64, bits=6,
                        class_predicate=lambda path, module: hasattr(module, "to_quantized")
                        and model.quant_predicate(path, module))
            model.eval()
            # Keep random initialization and weight quantization outside decode compilation.
            mx.eval(model.parameters())
            return model

        baseline, candidate = create_model(), create_model()
        candidate.load_weights(tree_flatten(baseline.parameters()))
        parameter_names = [name for name, _ in tree_flatten(candidate.parameters())]
        undo, undo_baseline = [], []
        try:
            # Tiny fused N196 switches MLX's qmv reduction; isolate the router increment.
            for installer in (fuse_projections, fuse_gdn, fuse_moe, compile_decode):
                undo_baseline.append(installer(baseline))
            for installer in (fuse_projections, fuse_gdn, fuse_moe, apply, compile_decode):
                undo.append(installer(candidate))
            self.assertEqual(parameter_names, [name for name, _ in tree_flatten(candidate.parameters())])
            sessions = [(baseline.make_cache(), candidate.make_cache()) for _ in range(2)]
            with patch("mtplx.local_optimizations.qwen35.router_kernel.route", wraps=route) as routed:
                for session, (ref_cache, opt_cache) in enumerate(sessions):
                    prompt = mx.array([[1, 2, 3] + ([4] if session else [])])
                    self.assert_exact(candidate(prompt, opt_cache), baseline(prompt, ref_cache))
                self.assertEqual(routed.call_count, len(candidate.layers))
                tokens = [mx.array([[5]]), mx.array([[6]])]
                for _ in range(6):
                    for session, (ref_cache, opt_cache) in enumerate(sessions):
                        actual = candidate(tokens[session], opt_cache)
                        expected = baseline(tokens[session], ref_cache)
                        self.assert_exact(actual, expected)
                        self.assert_exact(mx.argmax(actual, -1), mx.argmax(expected, -1))
                        for ref_entry, opt_entry in zip(ref_cache, opt_cache):
                            for ref_state, opt_state in zip(ref_entry.state, opt_entry.state):
                                self.assert_exact(opt_state, ref_state)
                        tokens[session] = mx.argmax(expected[:, -1:, :], -1)
                self.assertEqual(routed.call_count, 2 * len(candidate.layers))
            for layer in candidate.layers:
                self.assertEqual((layer if layer.is_linear else layer.mlp)._decode_calls,
                                 12 if layer.is_linear else 13)
            undo.pop()()
            self.assertTrue(all(type(layer.mlp) is _RoutedMoE for layer in candidate.layers))
            ref_cache, opt_cache = sessions[0]
            self.assert_exact(candidate(mx.array([[7, 8]]), opt_cache), baseline(mx.array([[7, 8]]), ref_cache))
        finally:
            for restore in reversed(undo_baseline + undo):
                restore()
                restore()
        self.assertTrue(all(type(layer.mlp) is Qwen3NextSparseMoeBlock for layer in candidate.layers))
        self.assertEqual(parameter_names, [name for name, _ in tree_flatten(candidate.parameters())])
        self.assert_exact(candidate(mx.array([[9]]), opt_cache), baseline(mx.array([[9]]), ref_cache))

    def test_install_validation_and_undo(self):
        args = TextModelArgs(hidden_size=32, num_experts=256, num_experts_per_tok=8,
                             moe_intermediate_size=32, shared_expert_intermediate_size=32)
        blocks = [Qwen3NextSparseMoeBlock(args) for _ in range(2)]
        model = SimpleNamespace(layers=[SimpleNamespace(mlp=block) for block in blocks])
        blocks[-1].top_k = 4
        with self.assertRaises(ValueError):
            apply(model)
        self.assertTrue(all(type(block) is Qwen3NextSparseMoeBlock for block in blocks))
        blocks[-1].top_k = 8
        undo = apply(model)
        self.addCleanup(undo)
        with self.assertRaises(TypeError):
            apply(model)
        self.assertTrue(all(type(block) is _RoutedMoE for block in blocks))
        undo()
        undo()
        self.assertTrue(all(type(block) is Qwen3NextSparseMoeBlock for block in blocks))


if __name__ == "__main__":
    unittest.main()
