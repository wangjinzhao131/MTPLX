"""GPU checks: ./python -m unittest discover -s tests -p test_attention_compile.py."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.qwen3_5 import DecoderLayer, Model, ModelArgs

from mtplx.local_optimizations.qwen35 import attention_compile
from mtplx.local_optimizations.qwen35.attention_compile import _CompiledAttentionLayer, apply
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ArraysCache
from mtplx.local_optimizations.qwen35.fusion import apply as fuse_projections
from mtplx.local_optimizations.qwen35.gdn_kernel import apply as fuse_gdn
from mtplx.local_optimizations.qwen35.moe_kernel import apply as fuse_moe
from mtplx.local_optimizations.qwen35.opt import apply as compile_decode
from mtplx.local_optimizations.qwen35.router_kernel import apply as fuse_router
from .test_opt import make_model


def prefill(layer, prompt):
    cache = ArraysCache(2) if layer.is_linear else KVCache()
    mask = None if layer.is_linear else create_attention_mask(prompt, cache)
    output = layer(prompt, mask=mask, cache=cache)
    mx.eval(output, cache.state)
    return cache


def make_half_model(router=False):
    args = make_model().args
    config = {**args.text_config, "hidden_size": 256, "head_dim": 256,
              "num_hidden_layers": 2, "num_experts": 256 if router else 4,
              "num_experts_per_tok": 8 if router else 2}
    model = Model(ModelArgs(args.model_type, config))
    model.set_dtype(mx.float16)
    nn.quantize(model, group_size=64, bits=6,
                class_predicate=lambda path, module: hasattr(module, "to_quantized")
                and model.quant_predicate(path, module))
    model.eval()
    return model


class AttentionCompileTests(unittest.TestCase):
    def assert_exact(self, actual, expected, label=""):
        self.assertEqual(actual.shape, expected.shape, label)
        self.assertEqual(actual.dtype, expected.dtype, label)
        if not mx.array_equal(actual, expected).item():
            errors = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)).reshape(-1)
            index = int(mx.argmax(errors).item())
            self.fail(f"{label}: shape={actual.shape} dtype={actual.dtype} "
                      f"unequal={int(mx.sum(actual != expected).item())} "
                      f"max_abs={errors[index].item()} flat_index={index} "
                      f"actual={actual.reshape(-1)[index].item()} "
                      f"expected={expected.reshape(-1)[index].item()}")

    def assert_model_cache(self, actual, expected, label):
        for layer, (left, right) in enumerate(zip(actual, expected)):
            for slot, (actual_state, expected_state) in enumerate(zip(left.state, right.state)):
                self.assert_exact(actual_state, expected_state,
                                  f"{label} layer={layer} cache={type(left).__name__} state={slot}")

    def assert_cache(self, actual, expected):
        self.assertEqual(actual.offset, expected.offset)
        self.assertEqual(actual.keys.shape, expected.keys.shape)
        self.assertEqual(actual.values.shape, expected.values.shape)
        for left, right in zip(actual.state, expected.state):
            self.assert_exact(left, right)

    def test_growth_sdpa_boundary_tensor_offset_and_trim(self):
        mx.random.seed(269)
        stock = make_half_model().layers[-1]
        candidate = make_half_model().layers[-1]
        candidate.load_weights(tree_flatten(stock.parameters()))
        model = SimpleNamespace(layers=[candidate])
        undo = [fuse_projections(model), fuse_moe(model), compile_decode(model)]
        parameter_ids = [(name, id(value)) for name, value in tree_flatten(candidate.parameters())]
        undo.append(apply(model))
        self.addCleanup(lambda: [restore() for restore in reversed(undo)])
        self.assertEqual(parameter_ids, [(name, id(value)) for name, value in tree_flatten(candidate.parameters())])
        sessions = []
        for length in (253, 1021):
            prompt = mx.random.normal((1, length, 256)).astype(mx.float16)
            sessions.append((prefill(stock, prompt), prefill(candidate, prompt)))
            self.assert_cache(sessions[-1][1], sessions[-1][0])
        self.assertEqual(candidate._attention_calls, 0)
        hub = candidate.self_attn.q_proj._fusion
        seen_lengths = []
        native_sdpa = attention_compile.scaled_dot_product_attention

        def record(queries, keys, values, **kwargs):
            seen_lengths.append(keys.shape[2])
            self.assertEqual(keys.shape[2], kwargs["cache"].offset)
            self.assertIsNone(kwargs["mask"])
            return native_sdpa(queries, keys, values, **kwargs)

        with patch("mtplx.local_optimizations.qwen35.attention_compile.scaled_dot_product_attention", side_effect=record):
            for _ in range(6):
                for ref_cache, opt_cache in sessions:
                    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
                    self.assert_exact(candidate(x, cache=opt_cache), stock(x, cache=ref_cache))
                    self.assert_cache(opt_cache, ref_cache)
            for ref_cache, opt_cache in sessions:
                self.assertEqual(opt_cache.trim(3), ref_cache.trim(3))
                x = mx.random.normal((1, 1, 256)).astype(mx.float16)
                self.assert_exact(candidate(x, cache=opt_cache), stock(x, cache=ref_cache))
                self.assert_cache(opt_cache, ref_cache)
        self.assertEqual(candidate._attention_calls, 14)
        self.assertEqual(hub.calls, 1, "changing tensor offsets or KV lengths retraced the pre block")
        self.assertEqual(candidate.mlp._decode_calls, 1, "post block should reuse its nested compiled MLP")
        self.assertTrue({255, 256, 257, 1023, 1024, 1025}.issubset(seen_lengths))
        undo.pop()()
        self.assertIs(type(candidate), DecoderLayer)
        ref_cache, opt_cache = sessions[0]
        x = mx.random.normal((1, 1, 256)).astype(mx.float16)
        self.assert_exact(candidate(x, cache=opt_cache), stock(x, cache=ref_cache))

    def test_attention_increment_over_matched_full_stack(self):
        mx.random.seed(277)
        stock, candidate = make_half_model(True), make_half_model(True)
        candidate.load_weights(tree_flatten(stock.parameters()))
        original_classes = [(type(layer), type(layer.mlp)) for layer in candidate.layers]
        undo = []
        try:
            common = (fuse_projections, fuse_gdn, fuse_moe, fuse_router, compile_decode)
            # Tiny GDN fusion changes QMV dispatch; isolate attention against the same existing stack.
            for installer in common:
                undo.append(installer(stock))
            for installer in (*common, apply):
                undo.append(installer(candidate))
            sessions = [(stock.make_cache(), candidate.make_cache()) for _ in range(2)]
            for session, (ref_cache, opt_cache) in enumerate(sessions):
                prompt = mx.array([[1, 2, 3] + ([4] if session else [])])
                label = f"matched-stack prefill session={session}"
                self.assert_exact(candidate(prompt, opt_cache), stock(prompt, ref_cache), f"{label} logits")
                self.assert_model_cache(opt_cache, ref_cache, label)
            tokens = [mx.array([[5]]), mx.array([[6]])]
            for step in range(6):
                for session, (ref_cache, opt_cache) in enumerate(sessions):
                    actual = candidate(tokens[session], opt_cache)
                    expected = stock(tokens[session], ref_cache)
                    label = f"matched-stack decode step={step} session={session}"
                    self.assert_exact(actual, expected, f"{label} logits")
                    self.assert_exact(mx.argmax(actual, -1), mx.argmax(expected, -1), f"{label} greedy")
                    self.assert_model_cache(opt_cache, ref_cache, label)
                    tokens[session] = (mx.array([[7 + session]]) if step == 0
                                       else mx.argmax(expected[:, -1:, :], -1))
            self.assertEqual(candidate.layers[-1]._attention_calls, 12)
            self.assertEqual(candidate.layers[0]._decode_calls, 12)
        finally:
            for restore in reversed(undo):
                restore()
                restore()
        self.assertEqual(original_classes, [(type(layer), type(layer.mlp)) for layer in candidate.layers])
        ref_cache, opt_cache = sessions[0]
        self.assert_exact(candidate(mx.array([[10]]), opt_cache), stock(mx.array([[10]]), ref_cache))

    def test_fallbacks_keep_stock_attention(self):
        mx.random.seed(281)
        stock = make_half_model().layers[-1]
        candidate = make_half_model().layers[-1]
        candidate.load_weights(tree_flatten(stock.parameters()))
        model = SimpleNamespace(layers=[candidate])
        undo = [compile_decode(model), apply(model)]
        self.addCleanup(lambda: [restore() for restore in reversed(undo)])
        for batch, length, training, kind, mask in (
                (1, 3, False, KVCache, "causal"),
                (2, 1, False, KVCache, None),
                (1, 1, True, KVCache, None),
                (1, 1, False, KVCache, "causal"),
                (1, 1, False, lambda: RotatingKVCache(16), None)):
            with self.subTest(batch=batch, length=length, training=training, kind=kind, mask=mask):
                stock.train(training)
                candidate.train(training)
                ref_cache, opt_cache = kind(), kind()
                prefix = mx.random.normal((batch, 3, 256)).astype(mx.float16)
                self.assert_exact(candidate(prefix, mask="causal", cache=opt_cache),
                                  stock(prefix, mask="causal", cache=ref_cache))
                x = mx.random.normal((batch, length, 256)).astype(mx.float16)
                self.assert_exact(candidate(x, mask=mask, cache=opt_cache), stock(x, mask=mask, cache=ref_cache))
                for left, right in zip(opt_cache.state, ref_cache.state):
                    self.assert_exact(left, right)
        stock.eval()
        candidate.eval()
        x = mx.random.normal((1, 1, 256)).astype(mx.float16)
        for ref_cache, opt_cache in ((None, None), (KVCache(), KVCache())):
            self.assert_exact(candidate(x, cache=opt_cache), stock(x, cache=ref_cache))
        self.assertEqual(candidate._attention_calls, 0)

    def test_install_validation_is_atomic_and_undo_idempotent(self):
        model = SimpleNamespace(layers=[make_half_model().layers[-1], make_half_model().layers[-1]])
        with self.assertRaises(TypeError):
            apply(model)
        undo_compile = compile_decode(model)
        self.addCleanup(undo_compile)
        attention = model.layers[-1].self_attn
        rope = attention.rope
        attention.rope = lambda x, offset=0: x
        with self.assertRaises(ValueError):
            apply(model)
        self.assertTrue(all(type(layer) is DecoderLayer for layer in model.layers))
        attention.rope = rope
        undo = apply(model)
        self.addCleanup(undo)
        self.assertIs(type(model.layers[-1]), _CompiledAttentionLayer)
        with self.assertRaises(TypeError):
            apply(model)
        undo()
        undo()
        self.assertIs(type(model.layers[-1]), DecoderLayer)
        self.assertNotIn("_attention_before", model.layers[-1])


if __name__ == "__main__":
    unittest.main()
