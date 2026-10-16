"""Small real-model checks; run with the installed MLX runtime on the GPU."""

import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from mtplx.local_optimizations.qwen35.opt import apply


def make_model(quantized=False):
    model = Model(ModelArgs(model_type="qwen3_5_moe", text_config={
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "vocab_size": 128,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "norm_topk_prob": True,
    }))
    if quantized:
        nn.quantize(model, group_size=64, bits=6)
    model.eval()
    return model


class CompileDecodeTests(unittest.TestCase):
    def assert_array_close(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(mx.allclose(actual, expected, rtol=1e-4, atol=1e-5).item())

    def check_decode(self, quantized):
        mx.random.seed(17)
        reference = make_model(quantized)
        candidate = make_model(quantized)
        candidate.load_weights(tree_flatten(reference.parameters()))
        original_classes = [(type(layer), type(layer.mlp)) for layer in candidate.layers]
        parameter_names = [name for name, _ in tree_flatten(candidate.parameters())]
        undo = apply(candidate)
        self.assertEqual(parameter_names, [name for name, _ in tree_flatten(candidate.parameters())])
        sessions = [(reference.make_cache(), candidate.make_cache()) for _ in range(2)]
        for i, (ref_cache, opt_cache) in enumerate(sessions):
            prompt = mx.array([[1, 2 + i, 3 + i]])
            self.assert_array_close(candidate(prompt, opt_cache), reference(prompt, ref_cache))
        for token in (5, 6, 7):
            for i, (ref_cache, opt_cache) in enumerate(sessions):
                inputs = mx.array([[token + i]])
                self.assert_array_close(candidate(inputs, opt_cache), reference(inputs, ref_cache))
                for ref_entry, opt_entry in zip(ref_cache, opt_cache):
                    for ref_state, opt_state in zip(ref_entry.state, opt_entry.state):
                        self.assert_array_close(opt_state, ref_state)
        for layer in candidate.layers:
            self.assertEqual((layer if layer.is_linear else layer.mlp)._decode_calls, 6)
        undo()
        undo()
        self.assertEqual(original_classes, [(type(layer), type(layer.mlp)) for layer in candidate.layers])
        self.assertEqual(parameter_names, [name for name, _ in tree_flatten(candidate.parameters())])
        ref_cache, opt_cache = sessions[0]
        self.assert_array_close(candidate(mx.array([[8]]), opt_cache), reference(mx.array([[8]]), ref_cache))

    def test_decode_and_independent_states(self):
        self.check_decode(False)

    def test_six_bit_decode_and_independent_states(self):
        self.check_decode(True)

    def test_gdn_fallbacks(self):
        model = make_model()
        layer = model.layers[0]
        original_call = type(layer).__call__
        undo = apply(model)
        for shape, populated, padded, masked in (
            ((1, 1, 64), False, False, False),
            ((1, 2, 64), True, False, False),
            ((2, 1, 64), True, False, False),
            ((1, 1, 64), True, True, False),
            ((1, 1, 64), True, False, True),
        ):
            with self.subTest(shape=shape, padded=padded, masked=masked):
                cache = ArraysCache(2)
                if populated:
                    original_call(layer, mx.zeros((shape[0], 2, 64)), cache=cache)
                if padded:
                    cache.left_padding = mx.array([0])
                reference_cache = ArraysCache(2)
                reference_cache.state = list(cache.state)
                reference_cache.left_padding = cache.left_padding
                mask = mx.ones(shape[:2], dtype=mx.bool_) if masked else None
                x = mx.ones(shape)
                self.assert_array_close(
                    layer(x, mask=mask, cache=cache),
                    original_call(layer, x, mask=mask, cache=reference_cache),
                )
                for actual, expected in zip(cache.state, reference_cache.state):
                    self.assert_array_close(actual, expected)
        self.assertEqual(layer._decode_calls, 0)
        undo()

    def test_invalid_mode_and_double_install(self):
        model = make_model()
        with self.assertRaises(ValueError):
            apply(model, "unknown")
        undo = apply(model)
        with self.assertRaises(TypeError):
            apply(model)
        undo()

    def test_unknown_moe_subclass_rejected(self):
        class UnknownMoE(Qwen3NextSparseMoeBlock):
            pass

        model = make_model()
        model.layers[-1].mlp.__class__ = UnknownMoE
        with self.assertRaises(TypeError):
            apply(model)
        self.assertFalse(any(hasattr(layer, "_decode") for layer in model.layers))


if __name__ == "__main__":
    unittest.main()
