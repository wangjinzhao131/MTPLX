"""GPU checks: ./python -m unittest discover -s tests -p test_down_kernel.py."""

import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from mtplx.local_optimizations.qwen35.down_kernel import TILES, apply, down
from mtplx.local_optimizations.qwen35.fusion import apply as fuse_projections
from mtplx.local_optimizations.qwen35.gdn_kernel import apply as fuse_gdn
from mtplx.local_optimizations.qwen35.moe_kernel import apply as fuse_moe
from mtplx.local_optimizations.qwen35.opt import apply as compile_decode
from .test_opt import make_model


def native(x, indices, weight, scales, biases):
    return mx.gather_qmm(x, weight, scales, biases, rhs_indices=indices,
                         transpose=True, group_size=64, bits=6, sorted_indices=False)


def half_model():
    args = make_model().args
    model = Model(ModelArgs(args.model_type, {**args.text_config, "hidden_size": 256,
                                             "num_experts": 8, "num_experts_per_tok": 8,
                                             "moe_intermediate_size": 512}))
    model.set_dtype(mx.float16)
    nn.quantize(model, group_size=64, bits=6,
                class_predicate=lambda path, module: hasattr(module, "to_quantized")
                and model.quant_predicate(path, module))
    model.eval()
    mx.eval(model.parameters())
    return model


class DownKernelTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(73)
        self.arrays = mx.quantize(mx.random.normal((8, 64, 512), dtype=mx.float16) / 16,
                                  group_size=64, bits=6)
        # Compile only the projection, not the lazy random/quantization fixture.
        mx.eval(*self.arrays)
        self.indices = mx.array([[[7, 0, 3, 3, 1, 5, 2, 6]]], dtype=mx.uint32)

    def test_tiles_keep_native_rounding_and_slot_order(self):
        for magnitude in (0.0, 0.25, 1.0, 4.0):
            x = mx.random.normal((1, 1, 8, 1, 512), dtype=mx.float16) * magnitude
            expected = native(x, self.indices, *self.arrays)
            for rows, groups in TILES:
                with self.subTest(magnitude=magnitude, rows=rows, groups=groups):
                    actual = down(x, self.indices, *self.arrays,
                                  results_per_simd=rows, simdgroups=groups)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, mx.float16)
                    error = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
                    self.assertTrue(mx.array_equal(actual, expected).item(), f"max error={error}")

    def test_noncontiguous_inputs_and_invalid_expert(self):
        x = mx.random.normal((1, 1, 8, 1, 1024), dtype=mx.float16)[..., ::2]
        indices = mx.repeat(self.indices, 2, axis=-1)[..., ::2]
        expected = native(x, indices, *self.arrays)
        compiled = mx.compile(lambda a, i: down(a, i, *self.arrays, results_per_simd=16, simdgroups=4))
        contiguous_x, contiguous_indices = mx.contiguous(x), mx.contiguous(indices)
        outputs = {
            "native_contiguous": native(contiguous_x, contiguous_indices, *self.arrays),
            "metal_strided": down(x, indices, *self.arrays, results_per_simd=16, simdgroups=4),
            "compiled_strided": compiled(x, indices),
            "metal_contiguous": down(contiguous_x, contiguous_indices, *self.arrays,
                                      results_per_simd=16, simdgroups=4),
            "compiled_contiguous": compiled(contiguous_x, contiguous_indices),
        }
        mx.eval(expected, *outputs.values())
        checks = {name: {"unequal": mx.sum(value != expected).item(),
                         "max_error": mx.max(mx.abs(value.astype(mx.float32) - expected.astype(mx.float32))).item()}
                  for name, value in outputs.items()}
        self.assertTrue(all(check["unequal"] == 0 for check in checks.values()), str(checks))
        invalid = mx.array([[[8, 0, 3, 3, 1, 5, 2, 4294967295]]], dtype=mx.uint32)
        actual = down(x, invalid, *self.arrays)
        self.assertTrue(mx.all(mx.isnan(actual[:, :, 0])).item())
        self.assertTrue(mx.all(mx.isnan(actual[:, :, 7])).item())
        self.assertTrue(mx.array_equal(actual[:, :, 1:7], expected[:, :, 1:7]).item())

    def test_validation(self):
        x = mx.zeros((1, 1, 8, 1, 512), dtype=mx.float16)
        weight, scales, biases = self.arrays
        args = [x, self.indices, weight, scales, biases]
        for position, bad in ((0, x[..., :256]), (0, x.astype(mx.float32)),
                              (1, self.indices.astype(mx.int32)), (1, self.indices[..., :7]),
                              (2, weight[..., :95]), (2, weight[:, :63]),
                              (3, scales.astype(mx.float32)), (4, biases[..., :7])):
            changed = args.copy()
            changed[position] = bad
            with self.assertRaises(ValueError):
                down(*changed)
        for rows, groups in ((1, 2), (4, 1), (4, 8), (4.0, 2)):
            with self.assertRaises(ValueError):
                down(*args, results_per_simd=rows, simdgroups=groups)


class DownModelTests(unittest.TestCase):
    def assert_exact(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        error = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
        self.assertTrue(mx.array_equal(actual, expected).item(), f"max error={error}")

    def test_composition_caches_compilation_and_undo(self):
        mx.random.seed(79)
        reference, candidate = half_model(), half_model()
        candidate.load_weights(tree_flatten(reference.parameters()))
        for model in (reference, candidate):
            for install in (fuse_projections, fuse_gdn, fuse_moe):
                self.addCleanup(install(model))
            mx.eval(model.parameters())
        self.assertEqual({layer.is_linear for layer in candidate.layers}, {True, False})
        modules = [layer.mlp.switch_mlp.down_proj for layer in candidate.layers]
        before = dict(tree_flatten(candidate.parameters()))
        undo = apply(candidate)
        self.addCleanup(undo)
        after = dict(tree_flatten(candidate.parameters()))
        self.assertEqual(before.keys(), after.keys())
        for name, value in before.items():
            self.assertIs(after[name], value)
        for layer, module in zip(candidate.layers, modules):
            self.assertIs(layer.mlp.switch_mlp.down_proj, module)
        for compiled in (False, True):
            with self.subTest(compiled=compiled):
                undo_ref = compile_decode(reference) if compiled else lambda: None
                undo_opt = compile_decode(candidate) if compiled else lambda: None
                try:
                    sessions = [(reference.make_cache(), candidate.make_cache()) for _ in range(2)]
                    for session, (ref_cache, opt_cache) in enumerate(sessions):
                        prompt = mx.array([[1, 2, 3 + session]])
                        self.assert_exact(candidate(prompt, opt_cache), reference(prompt, ref_cache))
                    for step in range(4):
                        for session, (ref_cache, opt_cache) in enumerate(sessions):
                            token = mx.array([[5 + step + session]])
                            expected, actual = reference(token, ref_cache), candidate(token, opt_cache)
                            self.assert_exact(actual, expected)
                            for ref_entry, opt_entry in zip(ref_cache, opt_cache):
                                self.assertEqual(len(opt_entry.state), len(ref_entry.state))
                                for ref_state, opt_state in zip(ref_entry.state, opt_entry.state):
                                    self.assert_exact(opt_state, ref_state)
                finally:
                    undo_opt()
                    undo_ref()
        undo()
        undo()
        for module in modules:
            self.assertIs(type(module), QuantizedSwitchLinear)
        for name, value in dict(tree_flatten(candidate.parameters())).items():
            self.assertIs(value, before[name])
        ref_cache, opt_cache = sessions[0]
        token = mx.array([[17]])
        self.assert_exact(candidate(token, opt_cache), reference(token, ref_cache))

    def test_projection_fallbacks(self):
        model = half_model()
        module = model.layers[0].mlp.switch_mlp.down_proj
        self.addCleanup(apply(model))
        cases = [((1, 1, 8), mx.float16, False, False),
                 ((1, 3, 8), mx.float16, False, False),
                 ((2, 1, 8), mx.float16, False, False),
                 ((1, 1, 2), mx.float16, False, False),
                 ((1, 1, 8), mx.float32, False, False),
                 ((1, 1, 8), mx.float16, True, False),
                 ((1, 1, 8), mx.float16, False, True)]
        for i, (shape, dtype, training, sorted_indices) in enumerate(cases):
            with self.subTest(shape=shape, dtype=dtype, training=training, sorted=sorted_indices):
                module.train(training)
                x = mx.random.normal((*shape, 1, 512)).astype(dtype)
                indices = mx.broadcast_to(mx.arange(shape[-1], dtype=mx.uint32), shape)
                expected = QuantizedSwitchLinear.__call__(module, x, indices, sorted_indices)
                with patch("mtplx.local_optimizations.qwen35.down_kernel.down", wraps=down) as kernel:
                    actual = module(x, indices, sorted_indices)
                    self.assert_exact(actual, expected)
                    self.assertEqual(kernel.call_count, int(i == 0))

    def test_install_validation_is_atomic(self):
        model = half_model()
        first, last = (model.layers[i].mlp.switch_mlp.down_proj for i in (0, -1))
        last.bits = 8
        with self.assertRaises(ValueError):
            apply(model)
        self.assertIs(type(first), QuantizedSwitchLinear)
        last.bits = 6
        last.bias = mx.zeros((8, 256), dtype=mx.float16)
        with self.assertRaises(ValueError):
            apply(model)
        self.assertIs(type(first), QuantizedSwitchLinear)
        del last.bias
        undo = apply(model)
        self.addCleanup(undo)
        with self.assertRaises(TypeError):
            apply(model)
        undo()
        undo()
        self.assertIs(type(first), QuantizedSwitchLinear)
        self.assertIs(type(last), QuantizedSwitchLinear)


if __name__ == "__main__":
    unittest.main()
