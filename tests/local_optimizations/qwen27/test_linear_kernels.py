"""Exact target-only GEMV installation, fallback, cache and undo checks."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import Model, ModelArgs

from mtplx.local_optimizations.qwen27 import linear_kernels as candidate


def make_model():
    model = Model(ModelArgs(model_type="qwen3_5", text_config={
        "model_type": "qwen3_5_text", "hidden_size": 512,
        "intermediate_size": 512, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 128,
        "vocab_size": 512, "linear_num_value_heads": 4,
        "linear_num_key_heads": 1, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
    }))
    model.set_dtype(mx.float16)

    def predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if path.endswith("linear_attn.out_proj") or path.endswith("lm_head"):
            return {"bits": 8, "group_size": 64}
        return {"bits": 4, "group_size": 32}

    nn.quantize(model, class_predicate=predicate)
    model.eval()
    mx.eval(model.parameters())
    return model


class LinearKernelTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(731)

    def assert_exact(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        mx.eval(actual, expected)
        self.assertTrue(mx.array_equal(actual, expected).item(), {
            "unequal_values": mx.sum(actual != expected).item(),
            "max_abs_error": mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item(),
        })

    def test_dispatch_bias_noncontiguous_leading_dims_and_native_fallbacks(self):
        model = make_model()
        q4 = model.layers[0].mlp.gate_proj
        q8 = model.language_model.lm_head
        q4.bias = mx.random.normal((512,)).astype(mx.float16)
        mx.eval(q4.bias)
        undo = candidate.apply(model)
        for module in (q4, q8):
            for shape in ((2, 512), (1, 2, 512), (1, 1, 2, 512), (1, 3, 512),
                          (1, 4, 512), (1, 1, 512), (1, 5, 512), (2, 2, 512), (512,)):
                x = mx.random.normal((*shape[:-1], 1024)).astype(mx.float16)[..., ::2]
                mx.eval(x)
                expected = nn.QuantizedLinear.__call__(module, x)
                m = x.size // 512
                use_kernel = (len(shape) >= 2 and all(d == 1 for d in shape[:-2])
                              and m in ((2, 3) if module.bits == 4 else (2, 3, 4)))
                with self.subTest(bits=module.bits, shape=shape), patch.object(
                        candidate, "qmv", wraps=candidate.qmv) as called:
                    self.assert_exact(module(x), expected)
                    self.assertEqual(called.call_count, int(use_kernel))
                    if use_kernel:
                        expected_variant = {(4, 2): "paired4", (4, 3): "triplet4"}.get(
                            (module.bits, m), "shared")
                        self.assertEqual(called.call_args.kwargs["variant"],
                                         expected_variant)
            for dtype, training in ((mx.float32, False), (mx.float16, True)):
                module.train(training)
                x = mx.ones((1, 2, 512), dtype=dtype)
                with patch.object(candidate, "qmv", wraps=candidate.qmv) as called:
                    self.assert_exact(module(x), nn.QuantizedLinear.__call__(module, x))
                    called.assert_not_called()
            module.eval()
        undo()

    def test_full_model_two_caches_parameters_and_undo(self):
        stock, optimized = make_model(), make_model()
        optimized.load_weights(tree_flatten(stock.parameters()))
        mx.eval(optimized.parameters())
        original = tree_flatten(optimized.parameters())
        undo = candidate.apply(optimized)
        with self.assertRaises(TypeError):
            candidate.apply(optimized)
        current = tree_flatten(optimized.parameters())
        self.assertEqual([p for p, _ in original], [p for p, _ in current])
        for (_, a), (_, b) in zip(original, current):
            self.assertIs(a, b)
        sessions = [(stock.make_cache(), optimized.make_cache()) for _ in range(2)]
        with patch.object(candidate, "qmv", wraps=candidate.qmv) as called:
            for length in (6, 2, 3, 4, 1):
                for session, (left, right) in enumerate(sessions):
                    ids = (mx.arange(length, dtype=mx.uint32)[None, :] + 11 + session) % 512
                    self.assert_exact(optimized(ids, right), stock(ids, left))
                    for native, patched in zip(left, right):
                        self.assertEqual(getattr(native, "offset", None), getattr(patched, "offset", None))
                        for a, b in zip(native.state, patched.state):
                            self.assert_exact(b, a)
            self.assertGreater(called.call_count, 0)
        undo()
        undo()
        self.assertTrue(all(type(module) is nn.QuantizedLinear for _, module in optimized.named_modules()
                            if isinstance(module, nn.QuantizedLinear)))
        for (_, a), (_, b) in zip(original, tree_flatten(optimized.parameters())):
            self.assertIs(a, b)
        ids = mx.array([[31]], dtype=mx.uint32)
        self.assert_exact(optimized(ids, sessions[0][1]), stock(ids, sessions[0][0]))

    def test_mtp_exclusion_unsupported_format_and_outer_subclass(self):
        class InjectedOuter(Model):
            pass

        model = make_model()
        model.__class__ = InjectedOuter
        model.language_model.mtp = nn.QuantizedLinear(512, 512, False, group_size=64, bits=8)
        model.language_model.mtp.set_dtype(mx.float16)
        unsupported = nn.QuantizedLinear(512, 512, False, group_size=64, bits=4)
        unsupported.set_dtype(mx.float16)
        model.layers[0].mlp.up_proj = unsupported
        model.eval()
        mx.eval(model.parameters())
        undo = candidate.apply(model)
        self.assertIs(type(model.language_model.mtp), nn.QuantizedLinear)
        self.assertIs(type(unsupported), nn.QuantizedLinear)
        self.assertIs(type(model.language_model.lm_head), candidate._VerifyLinear)
        undo()

    def test_validation_is_atomic_and_unknown_classes_rejected(self):
        class UnknownLinear(nn.QuantizedLinear):
            pass

        for corruption in ("class", "shape", "sharding"):
            model = make_model()
            first = model.layers[0].mlp.gate_proj
            last = model.language_model.lm_head
            if corruption == "class":
                last.__class__ = UnknownLinear
            elif corruption == "shape":
                last.scales = last.scales[:, :-1]
            else:
                model.layers[0].linear_attn.sharding_group = object()
            with self.subTest(corruption=corruption), self.assertRaises((ValueError, TypeError)):
                candidate.apply(model)
            self.assertIs(type(first), nn.QuantizedLinear)
        with self.assertRaises(TypeError):
            candidate.apply(SimpleNamespace(layers=[]))


if __name__ == "__main__":
    unittest.main()
