"""Strict projection packing, installer composition, cache and undo GPU checks."""

import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import Model, ModelArgs

from mtplx.local_optimizations.qwen27 import decode_fusion as candidate
from mtplx.local_optimizations.qwen27 import linear_kernels
from mtplx.local_optimizations.qwen27 import prefill_gemm


def make_model(value_heads=8):
    model = Model(ModelArgs(model_type="qwen3_5", text_config={
        "model_type": "qwen3_5_text", "hidden_size": 512,
        "intermediate_size": 1024, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 128,
        "vocab_size": 512, "linear_num_value_heads": value_heads,
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
        return {"bits": 4, "group_size": 32}

    nn.quantize(model, class_predicate=quantization)
    model.eval()
    mx.eval(model.parameters())
    return model


class DecodeFusionTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(487)

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

    def test_full_model_packing_composition_caches_and_undo(self):
        stock, optimized = make_model(), make_model()
        optimized.load_weights(tree_flatten(stock.parameters()))
        mx.eval(optimized.parameters())
        before = dict(tree_flatten(optimized.parameters()))
        undo_decode = linear_kernels.apply(optimized)
        undo_prefill = prefill_gemm.apply(optimized)
        previous = [(module, type(module)) for module in optimized.modules()]
        undo = candidate.apply(optimized)
        groups = {module._fusion for module in optimized.modules() if hasattr(module, "_fusion")}
        self.assertEqual(len(groups), 4)
        self.assertTrue(all(not group.projection.training for group in groups))
        self.assertTrue(all(group.partitioned_outputs is False for group in groups))
        for group in groups:
            group.partitioned_outputs = True
        after = dict(tree_flatten(optimized.parameters()))
        self.assertEqual(after.keys(), before.keys())
        for name, value in before.items():
            self.assert_exact(after[name], value)
        sessions = [(stock.make_cache(), optimized.make_cache()) for _ in range(2)]
        next_ids = []
        try:
            with patch.object(candidate, "qmv", wraps=candidate.qmv) as fused_qmv, \
                    patch.object(prefill_gemm.gemm_kernel, "qmm", wraps=prefill_gemm.gemm_kernel.qmm) as prefill_qmm:
                for session, (left, right) in enumerate(sessions):
                    ids = (mx.arange(511 + 2 * session, dtype=mx.uint32)[None, :] * 7) % 512
                    expected, actual = stock(ids, left), optimized(ids, right)
                    self.assert_exact(actual, expected)
                    self.assert_caches(right, left)
                    next_ids.append(mx.argmax(expected[:, -1], -1).astype(mx.uint32)[:, None])
                # Four MLP projections, GDN qkv/z and attention q_proj per
                # prefill satisfy the conservative native-unsplit guard.
                self.assertEqual(prefill_qmm.call_count, 14)
                self.assertTrue(all(group.calls == 0 for group in groups))
                for length in (1, 2, 3, 4, 1, 2, 3):
                    for session, (left, right) in enumerate(sessions):
                        ids = mx.repeat(next_ids[session], length, axis=1)
                        expected, actual = stock(ids, left), optimized(ids, right)
                        self.assert_exact(actual, expected)
                        self.assert_caches(right, left)
                        self.assert_exact(mx.argmax(actual[:, -1], -1), mx.argmax(expected[:, -1], -1))
                        next_ids[session] = mx.argmax(expected[:, -1], -1).astype(mx.uint32)[:, None]
                self.assertEqual(fused_qmv.call_count, 32)
                self.assertEqual({call.kwargs["bits"] for call in fused_qmv.call_args_list}, {4, 8})
                partitioned_calls = 0
                for call in fused_qmv.call_args_list:
                    group = next(group for group in groups if call.args[1] is group.projection.weight)
                    partitioned = (call.args[0].shape[1] == 2 and group.projection.bits == 4
                                   and group.parts in (3, 4))
                    self.assertEqual(call.kwargs.get("output_splits"), group.widths if partitioned else None)
                    partitioned_calls += partitioned
                    self.assertEqual(call.kwargs["variant"],
                                     {(4, 2): "paired4", (4, 3): "triplet4"}.get(
                                         (call.kwargs["bits"], call.args[0].shape[1]), "shared"))
                self.assertEqual(partitioned_calls, 8)
                self.assertEqual(prefill_qmm.call_count, 14)
                for group in groups:
                    self.assertEqual(group.calls, 12)
                    self.assertIsNone(group.input)
                    self.assertIsNone(group.outputs)
            undo()
            undo()
            for module, old_class in previous:
                self.assertIs(type(module), old_class)
                self.assertFalse(hasattr(module, "_fusion"))
            self.assert_exact(optimized(next_ids[0], sessions[0][1]), stock(next_ids[0], sessions[0][0]))
            self.assert_caches(sessions[0][1], sessions[0][0])
        finally:
            undo()
            undo_prefill()
            undo_decode()
        after = dict(tree_flatten(optimized.parameters()))
        self.assertEqual(after.keys(), before.keys())
        for name, value in before.items():
            self.assert_exact(after[name], value)

    def test_direct_strides_compilation_and_original_fallback(self):
        model = make_model()
        undo_decode = linear_kernels.apply(model)
        undo_prefill = prefill_gemm.apply(model)
        undo = candidate.apply(model)
        try:
            module_groups = [(layer.mlp.gate_proj, layer.mlp.up_proj) for layer in model.layers]
            gdn, attention = model.layers[0].linear_attn, model.layers[1].self_attn
            module_groups.extend([
                (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a),
                (attention.q_proj, attention.k_proj, attention.v_proj),
            ])
            for modules in module_groups:
                hub = modules[0]._fusion
                self.assertEqual(hub.widths, tuple(module.weight.shape[0] for module in modules))
                for partitioned_outputs in (False, True):
                    hub.partitioned_outputs = partitioned_outputs
                    for length in (1, 2, 3):
                        x = mx.random.normal((1, length, 1024)).astype(mx.float16)[..., ::2]
                        mx.eval(x)
                        expected = tuple(nn.QuantizedLinear.__call__(module, x) for module in modules)
                        with patch.object(candidate, "qmv", wraps=candidate.qmv) as fused_qmv:
                            actual = tuple(module(x) for module in modules)
                        self.assertEqual(fused_qmv.call_count, int(length in (2, 3)))
                        if fused_qmv.call_count:
                            partitioned = (partitioned_outputs and length == 2
                                           and hub.projection.bits == 4 and len(modules) in (3, 4))
                            self.assertEqual(fused_qmv.call_args.kwargs.get("output_splits"),
                                             hub.widths if partitioned else None)
                        for a, b in zip(actual, expected):
                            self.assert_exact(a, b)
                        compiled = mx.compile(lambda value: tuple(module(value) for module in modules))
                        for a, b in zip(compiled(x), expected):
                            self.assert_exact(a, b)
                        changed_x = x * mx.array(0.5, dtype=mx.float16)
                        for a, module in zip(compiled(changed_x), modules):
                            self.assert_exact(a, nn.QuantizedLinear.__call__(module, changed_x))
                        self.assertIsNone(hub.outputs)
                cases = [((1, 4, 512), mx.float16, False),
                         ((2, 3, 512), mx.float16, False),
                         ((1, 512), mx.float16, False),
                         ((1, 3, 512), mx.float32, False),
                         ((1, 3, 512), mx.float16, True)]
                for shape, dtype, training in cases:
                    x = mx.ones(shape, dtype=dtype)
                    start = hub.calls
                    for module in modules:
                        module.train(training)
                        self.assert_exact(module(x), module._fusion_base_call(module, x))
                        module.eval()
                    self.assertEqual(hub.calls, start)
                    self.assertIsNone(hub.outputs)
                # A repeated member cannot reuse a partially served result;
                # fallback and undo also release any abandoned partial call.
                x = mx.ones((1, 2, 512), dtype=mx.float16)
                start = hub.calls
                modules[0](x)
                modules[0](x)
                self.assertEqual(hub.calls, start + 2)
                for module in modules[1:]:
                    module(x)
                self.assertEqual(hub.calls, start + 2)
                self.assertIsNone(hub.input)
                self.assertIsNone(hub.outputs)
                self.assertEqual(hub.served, 0)
                modules[0](x)
                modules[1](mx.ones((1, 4, 512), dtype=mx.float16))
                self.assertIsNone(hub.outputs)
            # Three-part q8 groups must stay packed even with the opt-in enabled.
            q8 = model.layers[1].mlp
            modules = (q8.gate_proj, q8.up_proj, q8.gate_proj)
            hub = candidate._ProjectionGroup(modules)
            hub.partitioned_outputs = True
            with patch.object(candidate, "qmv", wraps=candidate.qmv) as fused_qmv:
                actual = tuple(hub.part(x, index) for index in range(len(modules)))
            self.assertEqual(fused_qmv.call_count, 1)
            self.assertIsNone(fused_qmv.call_args.kwargs.get("output_splits"))
            for result, module in zip(actual, modules):
                self.assert_exact(result, nn.QuantizedLinear.__call__(module, x))
            self.assertIsNone(hub.outputs)
        finally:
            undo()
            undo_prefill()
            undo_decode()

    def test_native_install_guards_mtp_exclusion_and_atomic_validation(self):
        model = make_model(value_heads=4)
        model.language_model.mtp = nn.QuantizedLinear(512, 512, False, group_size=32, bits=4)
        undo = candidate.apply(model)
        try:
            self.assertFalse(hasattr(model.layers[0].linear_attn.in_proj_qkv, "_fusion"))
            self.assertTrue(hasattr(model.layers[0].mlp.gate_proj, "_fusion"))
            self.assertIs(type(model.language_model.mtp), nn.QuantizedLinear)
            self.assertIs(type(model.language_model.lm_head), nn.QuantizedLinear)
            x = mx.ones((1, 3, 512), dtype=mx.float16)
            for module in (model.layers[0].mlp.gate_proj, model.layers[0].mlp.up_proj):
                self.assert_exact(module(x), nn.QuantizedLinear.__call__(module, x))
            with self.assertRaises(TypeError):
                candidate.apply(model)
        finally:
            undo()
        # The input width must match native qmv_fast for every original group.
        model = make_model()
        gate, up = model.layers[0].mlp.gate_proj, model.layers[0].mlp.up_proj
        for module in (gate, up):
            module.weight = module.weight[:, :32]
            module.scales = module.scales[:, :8]
            module.biases = module.biases[:, :8]
        undo = candidate.apply(model)
        self.assertIs(type(gate), nn.QuantizedLinear)
        undo()
        for corruption in ("class", "shape", "metadata", "sharding"):
            model = make_model()
            last = model.layers[-1].self_attn.v_proj
            if corruption == "class":
                class Unknown(nn.QuantizedLinear):
                    pass
                last.__class__ = Unknown
            elif corruption == "shape":
                last.scales = last.scales[:, :-1]
            elif corruption == "metadata":
                last.bits = 8
            else:
                model.layers[0].linear_attn.sharding_group = object()
            before = [(module, type(module), module.get("weight")) for module in model.modules()]
            with self.subTest(corruption=corruption), self.assertRaises((TypeError, ValueError)):
                candidate.apply(model)
            for module, old_class, weight in before:
                self.assertIs(type(module), old_class)
                self.assertIs(module.get("weight"), weight)
                self.assertFalse(hasattr(module, "_fusion"))

    def test_rollback_after_a_packed_group(self):
        model = make_model()
        before = dict(tree_flatten(model.parameters()))
        original = candidate._ProjectionGroup
        groups = []

        def create(modules):
            if groups:
                raise RuntimeError("test interrupted packing")
            groups.append(original(modules))
            return groups[-1]

        with patch.object(candidate, "_ProjectionGroup", side_effect=create):
            with self.assertRaisesRegex(RuntimeError, "test interrupted packing"):
                candidate.apply(model)
        self.assertIsNone(groups[0].outputs)
        for module in model.modules():
            self.assertFalse(hasattr(module, "_fusion"))
            if isinstance(module, nn.QuantizedLinear):
                self.assertIs(type(module), nn.QuantizedLinear)
        after = dict(tree_flatten(model.parameters()))
        self.assertEqual(after.keys(), before.keys())
        for name, value in before.items():
            self.assert_exact(after[name], value)


if __name__ == "__main__":
    unittest.main()
