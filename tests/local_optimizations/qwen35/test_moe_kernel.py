"""GPU checks: ./python -m unittest discover -s tests -p test_moe_kernel.py."""

import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.activations import swiglu
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from mlx_lm.models.switch_layers import SwitchGLU

from mtplx.local_optimizations.qwen35.fusion import apply as fuse_projections
from mtplx.local_optimizations.qwen35.moe_kernel import apply, gate_up
from mtplx.local_optimizations.qwen35.opt import apply as compile_decode
from .test_opt import make_model


def native(x, indices, weight, scales, biases):
    both = mx.gather_qmm(x[..., None, None, :], weight, scales, biases,
                         rhs_indices=indices, transpose=True, group_size=64, bits=6)
    gate, up = (mx.contiguous(y) for y in mx.split(both, 2, axis=-1))
    return swiglu(gate, up)


def half_model():
    args = make_model().args
    model = Model(ModelArgs(args.model_type, {**args.text_config, "hidden_size": 256}))
    model.set_dtype(mx.float16)
    nn.quantize(model, group_size=64, bits=6,
                class_predicate=lambda path, module: hasattr(module, "to_quantized")
                and model.quant_predicate(path, module))
    model.eval()
    return model


class MoEKernelTests(unittest.TestCase):
    def test_native_inner_loop_and_half_activation(self):
        mx.random.seed(31)
        for k, n in ((256, 4), (512, 16), (2048, 64)):
            with self.subTest(k=k, n=n):
                w = mx.random.normal((8, 2 * n, k), dtype=mx.float16) / 32
                weight, scales, biases = mx.quantize(w, group_size=64, bits=6)
                for magnitude in (0.0, 0.25, 1.0, 4.0):
                    x = mx.random.normal((1, 1, k), dtype=mx.float16) * magnitude
                    # Unsorted duplicates and boundary experts must keep slot order.
                    indices = mx.array([[[7, 0, 3, 3, 1, 5, 2, 6]]], dtype=mx.uint32)
                    expected = native(x, indices, weight, scales, biases)
                    actual = gate_up(x, indices, weight, scales, biases)
                    mx.eval(expected, actual)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, mx.float16)
                    error = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
                    self.assertTrue(mx.array_equal(actual, expected).item(), f"max error={error}")
                    compact = gate_up(x, indices, weight, scales, biases, compact=True)
                    self.assertTrue(mx.allclose(compact, expected, atol=0.003, rtol=0.003).item())

    def test_single_expert_and_noncontiguous_input(self):
        mx.random.seed(7)
        weight, scales, biases = mx.quantize(
            mx.random.normal((2, 8, 256), dtype=mx.float16) / 16, group_size=64, bits=6)
        x = mx.random.normal((1, 1, 512), dtype=mx.float16)[..., ::2]
        indices = mx.array([[[1]]], dtype=mx.uint32)
        self.assertTrue(mx.array_equal(gate_up(x, indices, weight, scales, biases),
                                       native(x, indices, weight, scales, biases)).item())
        invalid = mx.array([[[2]]], dtype=mx.uint32)
        self.assertTrue(mx.all(mx.isnan(gate_up(x, invalid, weight, scales, biases))).item())

    def test_native_tiles_preserve_dot_order(self):
        mx.random.seed(59)
        weight, scales, biases = mx.quantize(
            mx.random.normal((4, 64, 512), dtype=mx.float16) / 16, group_size=64, bits=6)
        x = mx.random.normal((1, 1, 512), dtype=mx.float16)
        indices = mx.array([[[3, 0, 2, 2]]], dtype=mx.uint32)
        expected = native(x, indices, weight, scales, biases)
        for pairs in (1, 2, 4, 8):
            for simdgroups in (2, 4):
                with self.subTest(pairs=pairs, simdgroups=simdgroups):
                    actual = gate_up(x, indices, weight, scales, biases,
                                     pairs_per_simd=pairs, simdgroups=simdgroups)
                    self.assertTrue(mx.array_equal(actual, expected).item())

    def test_validation(self):
        x = mx.zeros((1, 1, 256), dtype=mx.float16)
        indices = mx.array([[[0]]], dtype=mx.uint32)
        weight = mx.zeros((2, 8, 48), dtype=mx.uint32)
        scales = mx.ones((2, 8, 4), dtype=mx.float16)
        biases = mx.zeros_like(scales)
        args = [x, indices, weight, scales, biases]
        for position, bad in ((0, mx.zeros((1, 2, 256), dtype=mx.float16)),
                              (0, x.astype(mx.float32)), (1, indices.astype(mx.int32)),
                              (2, weight[..., :-1]), (3, scales.astype(mx.float32))):
            changed = args.copy()
            changed[position] = bad
            with self.assertRaises(ValueError):
                gate_up(*changed)

    def test_sorted_prefill_reuses_projection_and_clears_hub(self):
        mx.random.seed(43)
        model = half_model()
        self.addCleanup(fuse_projections(model))
        self.addCleanup(apply(model))
        module = model.layers[0].mlp.switch_mlp
        hub = module.gate_proj._fusion
        for width in (32, 64, 128, 512):
            with self.subTest(width=width):
                x = mx.random.normal((1, width, 512), dtype=mx.float16)[..., ::2]
                indices = mx.array([[[3, 0], [1, 1], [0, 2], [2, 3]]] * (width // 4), dtype=mx.uint32)
                indices = indices.reshape(1, width, 2)
                expected = SwitchGLU.__call__(module, x, indices)
                module.gate_proj(x[:, :1, None, None, :], indices[:, :1])
                self.assertIsNotNone(hub.outputs)
                with patch.object(hub, "projection", wraps=hub.projection) as projection:
                    actual = module(x, indices)
                    projection.assert_called_once()
                    self.assertTrue(projection.call_args.kwargs["sorted_indices"])
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(mx.array_equal(actual, expected).item())
                self.assertIsNone(hub.outputs)
                self.assertIsNone(hub.input)
                self.assertIsNone(hub.indices)
                self.assertEqual(hub.served, 0)

    def test_sorted_prefill_fallback_dispatch(self):
        model = half_model()
        self.addCleanup(fuse_projections(model))
        self.addCleanup(apply(model))
        module = model.layers[0].mlp.switch_mlp
        for shape, indices_shape, dtype, index_dtype, training in (
            ((1, 31, 256), (1, 31, 2), mx.float16, mx.uint32, None),
            ((2, 32, 256), (2, 32, 2), mx.float16, mx.uint32, None),
            ((32, 256), (32, 2), mx.float16, mx.uint32, None),
            ((1, 32, 256), (1, 32, 2), mx.float32, mx.uint32, None),
            ((1, 32, 256), (1, 32, 2), mx.float16, mx.int32, None),
            ((1, 32, 256), (32, 2), mx.float16, mx.uint32, None),
            ((1, 32, 256), (1, 64, 2), mx.float16, mx.uint32, None),
            ((1, 3, 256), (1, 3, 64), mx.float16, mx.uint32, None),
            ((1, 32, 256), (1, 32, 2), mx.float16, mx.uint32, "self"),
            ((1, 32, 256), (1, 32, 2), mx.float16, mx.uint32, "gate_proj"),
            ((1, 32, 256), (1, 32, 2), mx.float16, mx.uint32, "up_proj"),
        ):
            with self.subTest(shape=shape, dtype=dtype, index_dtype=index_dtype, training=training):
                module.eval()
                if training is not None:
                    (module if training == "self" else getattr(module, training)).train()
                x, indices = mx.zeros(shape, dtype=dtype), mx.zeros(indices_shape, dtype=index_dtype)
                with patch.object(SwitchGLU, "__call__", return_value=object()) as native_call:
                    self.assertIs(module(x, indices), native_call.return_value)
                    native_call.assert_called_once_with(module, x, indices)

    def test_model_prefill_decode_cache_compile_and_undo(self):
        mx.random.seed(41)
        reference, candidate = half_model(), half_model()
        candidate.load_weights(tree_flatten(reference.parameters()))
        self.addCleanup(fuse_projections(reference))
        self.addCleanup(fuse_projections(candidate))
        before = dict(tree_flatten(candidate.parameters()))
        members = [(module, type(module)) for layer in candidate.layers
                   for module in (layer.mlp.switch_mlp.gate_proj, layer.mlp.switch_mlp.up_proj,
                                  layer.mlp.switch_mlp.down_proj)]
        undo = apply(candidate)
        self.addCleanup(undo)
        after = dict(tree_flatten(candidate.parameters()))
        self.assertEqual(before.keys(), after.keys())
        for name, value in before.items():
            self.assertIs(after[name], value)
        for module, original_class in members:
            self.assertIs(type(module), original_class)
        hubs = [layer.mlp.switch_mlp.gate_proj._fusion for layer in candidate.layers]
        for compiled in (False, True):
            with self.subTest(compiled=compiled):
                undo_ref = compile_decode(reference) if compiled else lambda: None
                undo_opt = compile_decode(candidate) if compiled else lambda: None
                try:
                    widths = (3, 64, 512)
                    sessions = [(reference.make_cache(), candidate.make_cache()) for _ in widths]
                    for width, (ref_cache, opt_cache) in zip(widths, sessions):
                        prompt = (mx.arange(width) % reference.args.text_config["vocab_size"])[None, :]
                        self.assertTrue(mx.array_equal(reference(prompt, ref_cache), candidate(prompt, opt_cache)).item())
                        for ref_entry, opt_entry in zip(ref_cache, opt_cache):
                            self.assertEqual(opt_entry.meta_state, ref_entry.meta_state)
                            for ref_state, opt_state in zip(ref_entry.state, opt_entry.state):
                                self.assertTrue(mx.array_equal(opt_state, ref_state).item())
                    for ref_cache, opt_cache in sessions:
                        token = mx.array([[5]])
                        for step in range(5):
                            expected, actual = reference(token, ref_cache), candidate(token, opt_cache)
                            self.assertTrue(mx.array_equal(actual, expected).item())
                            self.assertTrue(mx.array_equal(mx.argmax(actual, -1), mx.argmax(expected, -1)).item())
                            for ref_entry, opt_entry in zip(ref_cache, opt_cache):
                                for ref_state, opt_state in zip(ref_entry.state, opt_entry.state):
                                    self.assertTrue(mx.array_equal(opt_state, ref_state).item())
                            token = mx.array([[6]]) if step == 0 else mx.argmax(expected[:, -1:, :], -1)
                    for hub in hubs:
                        self.assertEqual(hub.calls, 0)
                        self.assertIsNone(hub.outputs)
                finally:
                    undo_opt()
                    undo_ref()
        undo()
        undo()
        for layer in candidate.layers:
            self.assertIs(type(layer.mlp.switch_mlp), SwitchGLU)
        for name, value in dict(tree_flatten(candidate.parameters())).items():
            self.assertIs(value, before[name])
        self.assertTrue(mx.array_equal(reference(mx.array([[9]]), ref_cache),
                                       candidate(mx.array([[9]]), opt_cache)).item())

    def test_model_fallbacks_and_empty_hub(self):
        model = half_model()
        self.addCleanup(fuse_projections(model))
        module = model.layers[0].mlp.switch_mlp
        hub = module.gate_proj._fusion
        indices = mx.array([[[0, 1]]], dtype=mx.uint32)
        # Installing the bypass must release an unfinished external projection call.
        module.gate_proj(mx.ones((1, 1, 1, 1, 256), dtype=mx.float16), indices)
        self.assertIsNotNone(hub.outputs)
        undo = apply(model)
        self.addCleanup(undo)
        self.assertIsNone(hub.outputs)
        for shape, training, selected in (((1, 3, 256), False, 2),
                                           ((2, 1, 256), False, 2),
                                           ((1, 1, 256), True, 2),
                                           ((1, 1, 256), False, 64)):
            module.train(training)
            x = mx.random.normal(shape, dtype=mx.float16)
            idx = mx.zeros((*shape[:2], selected), dtype=mx.uint32)
            self.assertTrue(mx.array_equal(module(x, idx), SwitchGLU.__call__(module, x, idx)).item())
            self.assertIsNone(hub.outputs)

    def test_install_validation_is_atomic(self):
        model = half_model()
        with self.assertRaises(TypeError):
            apply(model)
        self.addCleanup(fuse_projections(model))
        last = model.layers[-1].mlp.switch_mlp
        last.up_proj.bits = 8
        with self.assertRaises(ValueError):
            apply(model)
        self.assertIs(type(model.layers[0].mlp.switch_mlp), SwitchGLU)
        last.up_proj.bits = 6
        activation = last.activation
        last.activation = nn.ReLU()
        with self.assertRaises(TypeError):
            apply(model)
        last.activation = activation
        undo = apply(model)
        self.addCleanup(undo)
        with self.assertRaises(TypeError):
            apply(model)


if __name__ == "__main__":
    unittest.main()
