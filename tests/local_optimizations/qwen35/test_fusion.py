"""Run with ./python -m unittest discover -s tests -p test_fusion.py on Metal."""

import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from mtplx.local_optimizations.qwen35.fusion import apply
from .test_opt import make_model


class ProjectionFusionTests(unittest.TestCase):
    def assert_close(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(mx.allclose(actual, expected, rtol=1e-4, atol=1e-5).item())

    def test_six_bit_logits_cache_parameters_and_undo(self):
        mx.random.seed(23)
        reference, candidate = make_model(True), make_model(True)
        # Match the production model's mixed precision router configuration.
        for model in (reference, candidate):
            for layer in model.layers:
                layer.mlp.gate = nn.QuantizedLinear(64, 4, bias=False, group_size=64, bits=8)
                layer.mlp.shared_expert_gate = nn.QuantizedLinear(64, 1, bias=False, group_size=64, bits=8)
            model.eval()
        candidate.load_weights(tree_flatten(reference.parameters()))
        original = dict(tree_flatten(candidate.parameters()))
        classes = [(module, type(module)) for _, module in candidate.named_modules()]
        undo = apply(candidate)
        groups = {module._fusion for _, module in candidate.named_modules() if hasattr(module, "_fusion")}
        self.assertEqual(len(groups), 12)
        actual = dict(tree_flatten(candidate.parameters()))
        self.assertEqual(original.keys(), actual.keys())
        for name, value in original.items():
            self.assertTrue(mx.array_equal(value, actual[name]).item(), name)
        for layer in candidate.layers:
            self.assertEqual(layer.mlp.gate.bits, 8)
            self.assertEqual(layer.mlp.shared_expert_gate.bits, 8)
        sessions = [(reference.make_cache(), candidate.make_cache()) for _ in range(2)]
        for index, (ref_cache, opt_cache) in enumerate(sessions):
            tokens = mx.array([[1, 2, 3 + index]])
            self.assert_close(candidate(tokens, opt_cache), reference(tokens, ref_cache))
        for group in groups:
            self.assertEqual(group.calls, 0)
        for ref_cache, opt_cache in sessions:
            token = mx.array([[5]])
            for step in range(6):
                expected = reference(token, ref_cache)
                actual = candidate(token, opt_cache)
                self.assert_close(actual, expected)
                self.assertTrue(mx.array_equal(mx.argmax(actual, -1), mx.argmax(expected, -1)).item())
                for ref_entry, opt_entry in zip(ref_cache, opt_cache):
                    for ref_state, opt_state in zip(ref_entry.state, opt_entry.state):
                        self.assert_close(opt_state, ref_state)
                token = mx.array([[6 + step]]) if step < 2 else mx.argmax(expected[:, -1:, :], -1)
        for group in groups:
            self.assertEqual(group.calls, 12)
            self.assertIsNone(group.outputs)
        undo()
        undo()
        for module, original_class in classes:
            self.assertIs(type(module), original_class)
        self.assertEqual(original.keys(), dict(tree_flatten(candidate.parameters())).keys())
        ref_cache, opt_cache = sessions[0]
        self.assert_close(candidate(mx.array([[9]]), opt_cache), reference(mx.array([[9]]), ref_cache))

    def test_batched_and_training_fallbacks(self):
        model = make_model(True)
        dense = model.layers[0].mlp.shared_expert.gate_proj
        expert = model.layers[0].mlp.switch_mlp.gate_proj
        undo = apply(model)
        for shape in ((1, 3, 64), (2, 1, 64)):
            x = mx.ones(shape)
            self.assert_close(dense(x), nn.QuantizedLinear.__call__(dense, x))
        dense.train()
        x = mx.ones((1, 1, 64))
        self.assert_close(dense(x), nn.QuantizedLinear.__call__(dense, x))
        indices = mx.array([[[1, 2]], [[0, 1]]])
        x = mx.ones((2, 1, 1, 1, 64))
        self.assert_close(expert(x, indices), QuantizedSwitchLinear.__call__(expert, x, indices))
        self.assertEqual(dense._fusion.calls, 0)
        self.assertEqual(expert._fusion.calls, 0)
        undo()

    def test_invalid_metadata_and_double_install(self):
        model = make_model(True)
        member = model.layers[-1].mlp.switch_mlp.up_proj
        member.bits = 8
        with self.assertRaises(ValueError):
            apply(model)
        self.assertFalse(hasattr(model.layers[0].mlp.switch_mlp.gate_proj, "_fusion"))
        member.bits = 6
        undo = apply(model)
        with self.assertRaises(TypeError):
            apply(model)
        undo()


if __name__ == "__main__":
    unittest.main()
