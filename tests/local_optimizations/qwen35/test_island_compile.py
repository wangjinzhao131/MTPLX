"""GPU checks: ./python -m unittest discover -s tests -p test_island_compile.py."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_lm.models.qwen3_5 import DecoderLayer, Model as BaseModel
from mlx_lm.models.qwen3_5_moe import Model, ModelArgs

from mtplx.local_optimizations.qwen35 import engine
from mtplx.local_optimizations.qwen35 import island_compile
from mtplx.local_optimizations.qwen35.island_compile import _IslandModel, _IslandMoEModel, _ready, apply
from .test_opt import make_model


def make_full_stack_model():
    args = make_model().args
    config = {**args.text_config, "hidden_size": 256, "head_dim": 256,
              "num_hidden_layers": 8, "full_attention_interval": 4,
              "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512}
    model = Model(ModelArgs(args.model_type, config))
    model.set_dtype(mx.float16)
    nn.quantize(model, group_size=64, bits=6,
                class_predicate=lambda path, module: hasattr(module, "to_quantized")
                and model.quant_predicate(path, module))
    model.eval()
    return model


class IslandCompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(307)
        cls.reference, cls.candidate = make_full_stack_model(), make_full_stack_model()
        cls.candidate.load_weights(tree_flatten(cls.reference.parameters()))
        mx.eval(cls.reference.parameters(), cls.candidate.parameters())
        stack = ExitStack()
        cls.addClassCleanup(stack.close)
        # Match every other engine stage; install islands explicitly in each test.
        with patch("mtplx.local_optimizations.qwen35.island_compile.apply", return_value=lambda: None):
            stack.callback(engine.apply(cls.reference))
            stack.callback(engine.apply(cls.candidate))

    def setUp(self):
        self.reference.eval()
        self.candidate.eval()

    def assert_exact(self, actual, expected, label=""):
        self.assertEqual(actual.shape, expected.shape, label)
        self.assertEqual(actual.dtype, expected.dtype, label)
        if not mx.array_equal(actual, expected).item():
            error = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
            self.fail(f"{label}: unequal={mx.sum(actual != expected).item()} "
                      f"max_abs={mx.max(error).item()} shape={actual.shape}")

    def assert_caches(self, actual, expected):
        for index, (left, right) in enumerate(zip(actual, expected)):
            self.assertIs(type(left), type(right))
            if type(left) is KVCache:
                self.assertEqual(left.offset, right.offset)
                pairs = ((left.keys, right.keys), (left.values, right.values))
            else:
                pairs = zip(left.state, right.state)
            for slot, (a, b) in enumerate(pairs):
                self.assert_exact(a, b, f"layer={index} cache={slot}")

    def session(self, prompt=(1, 2, 3)):
        caches = self.reference.make_cache(), self.candidate.make_cache()
        inputs = mx.array([prompt])
        self.assert_exact(self.candidate(inputs, caches[1]), self.reference(inputs, caches[0]), "prefill")
        self.assert_caches(caches[1], caches[0])
        return caches

    def test_full_stack_sessions_boundaries_trim_and_undo(self):
        model = self.candidate
        parameters = [(name, id(value)) for name, value in tree_flatten(model.parameters())]
        undo = apply(model)
        self.addCleanup(undo)
        self.assertIs(type(model), _IslandMoEModel)
        self.assertIs(model.sanitize.__func__, Model.sanitize)
        self.assertEqual(parameters, [(name, id(value)) for name, value in tree_flatten(model.parameters())])
        sessions = [self.session(), self.session((1, 4, 5, 6))]
        # Repeated valid KV prefixes exercise long-context dispatch without an expensive MoE prefill.
        for length, caches in zip((253, 1021), sessions):
            capacity = ((length + 255) // 256) * 256
            for cache in caches:
                for entry in cache:
                    if type(entry) is KVCache:
                        entry.keys = mx.repeat(entry.keys[:, :, :1, :], capacity, axis=2)
                        entry.values = mx.repeat(entry.values[:, :, :1, :], capacity, axis=2)
                        entry.offset = length
            self.assert_caches(caches[1], caches[0])
        self.assertEqual(model._island_calls, 0)
        seen_lengths = []
        native = island_compile.scaled_dot_product_attention

        def record(queries, keys, values, **kwargs):
            seen_lengths.append(keys.shape[2])
            self.assertEqual(keys.shape[2], kwargs["cache"].offset)
            self.assertEqual(values.shape[2], keys.shape[2])
            self.assertIsNone(kwargs["mask"])
            return native(queries, keys, values, **kwargs)

        tokens = [mx.array([[7]], dtype=mx.uint32), mx.array([[8]], dtype=mx.uint32)]
        with patch("mtplx.local_optimizations.qwen35.island_compile.scaled_dot_product_attention", side_effect=record):
            for step in range(6):
                for session, (reference, candidate) in enumerate(sessions):
                    actual = model(tokens[session], candidate)
                    expected = self.reference(tokens[session], reference)
                    self.assert_exact(actual, expected, f"decode step={step} session={session}")
                    self.assert_exact(mx.argmax(actual, -1), mx.argmax(expected, -1), "greedy")
                    self.assert_caches(candidate, reference)
                    tokens[session] = mx.argmax(expected[:, -1, :], -1)[:, None]
            for reference, candidate in sessions:
                for left, right in zip(reference, candidate):
                    if type(left) is KVCache:
                        self.assertEqual(left.trim(3), right.trim(3))
                token = mx.array([[9]], dtype=mx.uint32)
                self.assert_exact(model(token, candidate), self.reference(token, reference), "after trim")
                self.assert_caches(candidate, reference)
        self.assertTrue({255, 256, 257, 1023, 1024, 1025}.issubset(seen_lengths))
        self.assertEqual(model._island_calls, 14)
        self.assertEqual(model._island_traces, [1, 1, 1])
        self.assertEqual(parameters, [(name, id(value)) for name, value in tree_flatten(model.parameters())])
        undo()
        undo()
        self.assertIs(type(model), Model)
        self.assertFalse(hasattr(model, "_island_blocks"))
        reference, candidate = sessions[0]
        self.assert_exact(model(mx.array([[10]]), candidate),
                          self.reference(mx.array([[10]]), reference), "after undo")
        self.assert_caches(candidate, reference)

    def test_fallbacks_and_cache_readiness(self):
        model = self.candidate
        self.addCleanup(apply(model))
        for inputs in (mx.array([[1, 2]]), mx.array([[1], [2]]), mx.array([[1]])):
            self.assert_exact(model(inputs), self.reference(inputs), "no cache or nondecode shape")
        reference, candidate = self.session()
        token = mx.array([[5]])
        embeddings = model.language_model.model.embed_tokens(token)
        self.assert_exact(model(token, candidate, input_embeddings=embeddings),
                          self.reference(token, reference, input_embeddings=embeddings), "embeddings")
        self.assert_caches(candidate, reference)
        self.assertEqual(model._island_calls, 0)
        self.assertTrue(_ready(model, candidate))
        for name, value in (("lengths", mx.array([1])), ("left_padding", mx.array([0]))):
            setattr(candidate[0], name, value)
            self.assertFalse(_ready(model, candidate))
            setattr(candidate[0], name, None)
        original = candidate[0]
        for invalid in (None, ArraysCache(2), KVCache(), RotatingKVCache(32)):
            candidate[0] = invalid
            self.assertFalse(_ready(model, candidate))
        candidate[0] = original
        candidate[-1].offset -= 1
        self.assertFalse(_ready(model, candidate))
        candidate[-1].offset += 1
        self.assertFalse(_ready(model, None))
        self.assertFalse(_ready(model, candidate[:-1]))
        self.reference.train()
        model.train()
        try:
            self.assert_exact(model(mx.array([[6, 7]])), self.reference(mx.array([[6, 7]])), "training")
        finally:
            self.reference.eval()
            model.eval()
        self.assertEqual(model._island_calls, 0)

    def test_install_validation_is_atomic(self):
        model = self.candidate
        original = type(model.layers[-1])
        model.layers[-1].__class__ = DecoderLayer
        try:
            with self.assertRaises(TypeError):
                apply(model)
            self.assertIs(type(model), Model)
            self.assertFalse(hasattr(model, "_island_blocks"))
        finally:
            model.layers[-1].__class__ = original
        model.layers[0].mlp.sharding_group = object()
        try:
            with self.assertRaises(ValueError):
                apply(model)
            self.assertFalse(hasattr(model, "_island_blocks"))
        finally:
            model.layers[0].mlp.sharding_group = None
        with patch("mtplx.local_optimizations.qwen35.island_compile.mx.compile", side_effect=RuntimeError("compile setup failed")):
            with self.assertRaisesRegex(RuntimeError, "compile setup failed"):
                apply(model)
        self.assertFalse(hasattr(model, "_island_blocks"))
        undo = apply(model)
        self.addCleanup(undo)
        with self.assertRaises(TypeError):
            apply(model)
        undo()
        undo()

    def test_known_model_types_preserve_sanitize_and_reject_subclasses(self):
        model = self.candidate

        class UnknownModel(Model):
            pass

        try:
            for original, patched in ((BaseModel, _IslandModel), (Model, _IslandMoEModel)):
                model.__class__ = original
                undo = apply(model)
                try:
                    self.assertIs(type(model), patched)
                    self.assertIs(model.sanitize.__func__, original.sanitize)
                finally:
                    undo()
                self.assertIs(type(model), original)
                self.assertIs(model.sanitize.__func__, original.sanitize)
            model.__class__ = UnknownModel
            with self.assertRaises(TypeError):
                apply(model)
            self.assertIs(type(model), UnknownModel)
            self.assertFalse(hasattr(model, "_island_blocks"))
        finally:
            model.__class__ = Model


if __name__ == "__main__":
    unittest.main()
