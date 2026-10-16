"""Strict installer, cache, fallback and decode-composition GPU checks."""

import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import Model, ModelArgs

from mtplx.local_optimizations.qwen27 import linear_kernels
from mtplx.local_optimizations.qwen27 import prefill_gemm as candidate


def make_model():
    model = Model(ModelArgs(model_type="qwen3_5", text_config={
        "model_type": "qwen3_5_text", "hidden_size": 128,
        "intermediate_size": 1024, "num_hidden_layers": 2,
        "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 128,
        "vocab_size": 128, "linear_num_value_heads": 3,
        "linear_num_key_heads": 1, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
    }))
    model.set_dtype(mx.float16)

    def quantization(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if path.startswith("language_model.model.layers.1.mlp."):
            return {"bits": 8, "group_size": 64}
        return True

    nn.quantize(model, group_size=32, bits=4, class_predicate=quantization)
    model.eval()
    mx.eval(model.parameters())
    return model


class PrefillGemmTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(467)

    def assert_exact(self, actual, expected):
        mx.eval(actual, expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(mx.array_equal(actual, expected).item(), {
            "max_abs_error": float(mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()),
            "unequal_elements": int(mx.sum(actual != expected).item()),
        })

    def test_models_caches_decode_composition_and_undo(self):
        stock, optimized = make_model(), make_model()
        optimized.load_weights(tree_flatten(stock.parameters()))
        mx.eval(optimized.parameters())
        before = tree_flatten(optimized.parameters())
        undo_decode = linear_kernels.apply(optimized)
        previous = [(module, type(module)) for module in optimized.language_model.model.modules()
                    if isinstance(module, nn.QuantizedLinear)]
        undo = candidate.apply(optimized)
        sessions = [(stock.make_cache(), optimized.make_cache()) for _ in range(2)]
        next_ids = []
        try:
            with patch.object(candidate.gemm_kernel, "qmm", wraps=candidate.gemm_kernel.qmm) as called, \
                    patch.object(linear_kernels, "qmv", wraps=linear_kernels.qmv) as decode_called:
                for session, (left, right) in enumerate(sessions):
                    ids = (mx.arange(512 + session, dtype=mx.uint32)[None, :] * 7) % 128
                    expected, actual = stock(ids, left), optimized(ids, right)
                    self.assert_exact(actual, expected)
                    for reference, cache in zip(left, right):
                        for ref_state, state in zip(reference.state, cache.state):
                            self.assert_exact(state, ref_state)
                        if hasattr(reference, "offset"):
                            self.assertEqual(cache.offset, reference.offset)
                    next_ids.append(mx.argmax(expected[:, -1], -1).astype(mx.uint32)[:, None])
                # Each prefill engages two MLP gate/up projections per layer;
                # other small output shapes retain native split-K.
                self.assertEqual(called.call_count, 8)
                for length in (1, 2, 3):
                    for session, (left, right) in enumerate(sessions):
                        ids = mx.repeat(next_ids[session], length, axis=1)
                        expected, actual = stock(ids, left), optimized(ids, right)
                        self.assert_exact(actual, expected)
                        self.assert_exact(mx.argmax(actual[:, -1], -1), mx.argmax(expected[:, -1], -1))
                        next_ids[session] = mx.argmax(expected[:, -1], -1).astype(mx.uint32)[:, None]
                        for reference, cache in zip(left, right):
                            for ref_state, state in zip(reference.state, cache.state):
                                self.assert_exact(state, ref_state)
                            if hasattr(reference, "offset"):
                                self.assertEqual(cache.offset, reference.offset)
                self.assertEqual(called.call_count, 8)
                # Both MLP down projections retain the installed decode path
                # for M=2/3 in each independent session.
                self.assertEqual(decode_called.call_count, 8)
            undo()
            undo()
            for module, previous_class in previous:
                self.assertIs(type(module), previous_class)
                self.assertFalse(hasattr(module, "_prefill_base_call"))
            self.assert_exact(optimized(next_ids[0], sessions[0][1]), stock(next_ids[0], sessions[0][0]))
        finally:
            undo()
            undo_decode()
        after = tree_flatten(optimized.parameters())
        self.assertEqual([name for name, _ in before], [name for name, _ in after])
        for (_, expected), (_, actual) in zip(before, after):
            self.assertIs(actual, expected)
        self.assertTrue(all(type(module) is nn.QuantizedLinear for module, _ in previous))

    def test_native_install_fallbacks_and_bias(self):
        model = make_model()
        gate = model.layers[0].mlp.gate_proj
        gate.bias = mx.random.normal((1024,)).astype(mx.float16)
        mx.eval(gate.bias)
        undo = candidate.apply(model)
        try:
            cases = [(mx.zeros((1, 512, 128), dtype=mx.float16), False, True),
                     (mx.zeros((1, 511, 128), dtype=mx.float16), False, True),
                     (mx.zeros((1, 481, 128), dtype=mx.float16), False, True),
                     (mx.zeros((1, 480, 128), dtype=mx.float16), False, False),
                     (mx.zeros((1, 31, 128), dtype=mx.float16), False, False),
                     (mx.zeros((1, 1, 128), dtype=mx.float16), False, False),
                     (mx.zeros((2, 512, 128), dtype=mx.float16), False, False),
                     (mx.zeros((512, 128), dtype=mx.float16), False, False),
                     (mx.zeros((1, 512, 128), dtype=mx.float32), False, False),
                     (mx.zeros((1, 512, 128), dtype=mx.float16), True, False)]
            for x, training, eligible in cases:
                gate.train(training)
                with self.subTest(shape=x.shape, dtype=str(x.dtype), training=training):
                    with patch.object(candidate.gemm_kernel, "qmm", wraps=candidate.gemm_kernel.qmm) as called:
                        self.assert_exact(gate(x), nn.QuantizedLinear.__call__(gate, x))
                    self.assertEqual(called.call_count, int(eligible))
            gate.eval()
        finally:
            undo()

    def test_unknown_classes_rejected_before_any_changes_and_mtp_excluded(self):
        model = make_model()
        mtp = nn.Module()
        mtp.projection = nn.QuantizedLinear(128, 1024, group_size=32, bits=4)
        model.language_model.mtp = mtp
        model.mtp = mtp
        gate = model.layers[-1].mlp.gate_proj

        class Unknown(nn.QuantizedLinear):
            pass

        gate.__class__ = Unknown
        before = [(module, type(module)) for module in model.modules()]
        with self.assertRaises(TypeError):
            candidate.apply(model)
        for module, old_class in before:
            self.assertIs(type(module), old_class)
        gate.__class__ = nn.QuantizedLinear
        undo = candidate.apply(model)
        try:
            self.assertIs(type(mtp.projection), nn.QuantizedLinear)
            with self.assertRaises(TypeError):
                candidate.apply(model)
        finally:
            undo()


if __name__ == "__main__":
    unittest.main()
