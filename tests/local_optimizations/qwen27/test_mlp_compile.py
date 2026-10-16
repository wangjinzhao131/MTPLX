"""Strict MLP compilation, fusion composition, cache and undo GPU checks."""

import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_next import Qwen3NextMLP

from mtplx.local_optimizations.qwen27 import engine
from mtplx.local_optimizations.qwen27 import mlp_compile as candidate
from .test_decode_fusion import make_model


def apply_without_mlp(model):
    # Keep the reference stack independent when engine includes this candidate.
    with patch.object(candidate, "apply", return_value=lambda: None):
        return engine.apply(model)


class MLPCompileTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(491)

    def assert_exact(self, actual, expected):
        mx.eval(actual, expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(mx.array_equal(actual, expected).item(), {
            "max_abs_error": float(mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()),
            "unequal_elements": int(mx.sum(actual != expected).item()),
        })

    def assert_caches(self, actual, expected):
        for cache, reference in zip(actual, expected):
            self.assertEqual(getattr(cache, "offset", None), getattr(reference, "offset", None))
            for state, ref_state in zip(cache.state, reference.state):
                self.assert_exact(state, ref_state)

    def test_matched_engine_two_caches_and_undo(self):
        reference, optimized = make_model(), make_model()
        optimized.load_weights(tree_flatten(reference.parameters()))
        mx.eval(optimized.parameters())
        undo_reference, undo_engine = apply_without_mlp(reference), apply_without_mlp(optimized)
        before = tree_flatten(optimized.parameters())
        undo = candidate.apply(optimized)
        blocks = [layer.mlp for layer in optimized.layers]
        sessions = [(reference.make_cache(), optimized.make_cache()) for _ in range(2)]
        next_ids = []
        try:
            with self.assertRaises(TypeError):
                candidate.apply(optimized)
            for session, (left, right) in enumerate(sessions):
                ids = (mx.arange(33 + 2 * session, dtype=mx.uint32)[None, :] * 7) % 512
                expected, actual = reference(ids, left), optimized(ids, right)
                self.assert_exact(actual, expected)
                self.assert_caches(right, left)
                next_ids.append(mx.argmax(expected[:, -1], -1).astype(mx.uint32)[:, None])
            self.assertTrue(all(block._compiled_calls == 0 for block in blocks))
            for length in (1, 2, 3, 4, 1, 2, 3):
                for session, (left, right) in enumerate(sessions):
                    ids = mx.repeat(next_ids[session], length, axis=1)
                    expected, actual = reference(ids, left), optimized(ids, right)
                    self.assert_exact(actual, expected)
                    self.assert_caches(right, left)
                    self.assert_exact(mx.argmax(actual[:, -1], -1), mx.argmax(expected[:, -1], -1))
                    next_ids[session] = mx.argmax(expected[:, -1], -1).astype(mx.uint32)[:, None]
            for block in blocks:
                self.assertEqual(block._compiled_calls, 12)
                self.assertEqual(block._compiled_traces, 3)
                self.assertEqual(block.gate_proj._fusion.calls, 3)
                self.assertIsNone(block.gate_proj._fusion.outputs)
            undo()
            undo()
            for block in blocks:
                self.assertIs(type(block), Qwen3NextMLP)
                self.assertFalse(hasattr(block, "_compiled_mlp"))
                self.assertFalse(hasattr(block, "_compiled_calls"))
                self.assertFalse(hasattr(block, "_compiled_traces"))
            after = tree_flatten(optimized.parameters())
            self.assertEqual([name for name, _ in before], [name for name, _ in after])
            for (_, old), (_, current) in zip(before, after):
                self.assertIs(current, old)
            self.assert_exact(optimized(next_ids[0], sessions[0][1]), reference(next_ids[0], sessions[0][0]))
            self.assert_caches(sessions[0][1], sessions[0][0])
        finally:
            undo()
            undo_engine()
            undo_reference()

    def test_native_and_fused_fallbacks_after_tracing(self):
        for install_engine in (False, True):
            model = make_model()
            undo_engine = apply_without_mlp(model) if install_engine else lambda: None
            undo = candidate.apply(model)
            try:
                for layer in model.layers:
                    block = layer.mlp
                    for length in (1, 2, 3, 1, 2, 3):
                        x = mx.random.normal((1, length, 512)).astype(mx.float16)
                        mx.eval(x)
                        self.assert_exact(block(x), Qwen3NextMLP.__call__(block, x))
                    self.assertEqual(block._compiled_calls, 6)
                    self.assertEqual(block._compiled_traces, 3)
                    for shape, dtype in (((1, 4, 512), mx.float16),
                                         ((1, 512, 512), mx.float16),
                                         ((2, 3, 512), mx.float16),
                                         ((1, 512), mx.float16),
                                         ((1, 3, 512), mx.float32)):
                        x = mx.ones(shape, dtype=dtype)
                        self.assert_exact(block(x), Qwen3NextMLP.__call__(block, x))
                        self.assertEqual(block._compiled_calls, 6)
                    x = mx.ones((1, 3, 512), dtype=mx.float16)
                    for training_module in (block, block.gate_proj, block.up_proj, block.down_proj):
                        training_module.train()
                        self.assert_exact(block(x), Qwen3NextMLP.__call__(block, x))
                        self.assertEqual(block._compiled_calls, 6)
                        training_module.eval()
                    self.assert_exact(block(x), Qwen3NextMLP.__call__(block, x))
                    self.assertEqual(block._compiled_calls, 7)
            finally:
                undo()
                undo_engine()

    def test_atomic_validation_and_mtp_exclusion(self):
        for corruption in ("mlp", "projection", "sharding"):
            model = make_model()
            block = model.layers[-1].mlp
            if corruption == "mlp":
                class UnknownMLP(Qwen3NextMLP):
                    pass
                block.__class__ = UnknownMLP
            elif corruption == "projection":
                class UnknownProjection(nn.QuantizedLinear):
                    pass
                block.down_proj.__class__ = UnknownProjection
            else:
                block.sharding_group = object()
            before = [(module, type(module)) for module in model.modules()]
            with self.subTest(corruption=corruption), self.assertRaises((TypeError, ValueError)):
                candidate.apply(model)
            for module, previous_class in before:
                self.assertIs(type(module), previous_class)
                self.assertFalse(hasattr(module, "_compiled_mlp"))
        model = make_model()
        model.language_model.mtp = Qwen3NextMLP(512, 1024)
        undo = candidate.apply(model)
        self.assertIs(type(model.language_model.mtp), Qwen3NextMLP)
        undo()

    def test_compile_setup_failure_rolls_back(self):
        model = make_model()
        original = candidate._compile_mlp
        calls = 0

        def compile_once(block):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("test interrupted compile setup")
            return original(block)

        with patch.object(candidate, "_compile_mlp", side_effect=compile_once):
            with self.assertRaisesRegex(RuntimeError, "test interrupted compile setup"):
                candidate.apply(model)
        for layer in model.layers:
            self.assertIs(type(layer.mlp), Qwen3NextMLP)
            self.assertFalse(hasattr(layer.mlp, "_compiled_mlp"))
            self.assertFalse(hasattr(layer.mlp, "_compiled_calls"))
            self.assertFalse(hasattr(layer.mlp, "_compiled_traces"))


if __name__ == "__main__":
    unittest.main()
