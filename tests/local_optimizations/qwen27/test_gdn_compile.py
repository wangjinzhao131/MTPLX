"""Strict GDN capture compilation and partial-acceptance GPU checks."""

import importlib
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache

from mtplx.local_optimizations.qwen27 import engine
from mtplx.local_optimizations.qwen27 import gdn_compile as candidate
from .test_decode_fusion import make_model


def capture_module():
    return importlib.import_module("mtplx.gdn_capture")


def cache_from(conv, state):
    cache = ArraysCache(2)
    cache[0], cache[1] = conv, state
    return cache


class GDNCompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = capture_module()

    def setUp(self):
        mx.random.seed(499)
        # Fixed stock semantics; the environment-switch test changes them later.
        environment = {name: "0" for name in candidate._ENV_BRANCHES}
        environment.update(MTPLX_CAPTURE_CUSTOM_KERNEL="stock",
                           MTPLX_CAPTURE_CONTIGUOUS_GDN_STATE="0")
        self.environment = patch.dict(os.environ, environment)
        self.environment.start()
        self.addCleanup(self.environment.stop)

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

    def assert_capture(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            self.assert_exact(actual[name], expected[name])

    def test_two_sessions_prefix_commit_correction_and_undo(self):
        reference, optimized = make_model(), make_model()
        optimized.load_weights(tree_flatten(reference.parameters()))
        mx.eval(optimized.parameters())
        undo_reference, undo_engine = engine.apply(reference), engine.apply(optimized)
        for layer in optimized.layers:
            projection = layer.linear_attn.in_proj_qkv if layer.is_linear else layer.self_attn.q_proj
            projection._fusion.partitioned_outputs = True
        before = tree_flatten(optimized.parameters())
        original = self.capture._gdn_forward_with_capture
        undo = candidate.apply(optimized)
        gdns = [layer.linear_attn for layer in optimized.layers if layer.is_linear]
        sessions = [(reference.make_cache(), optimized.make_cache()) for _ in range(2)]
        next_ids = []
        try:
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                candidate.apply(optimized)
            undo_other = candidate.apply(reference)
            undo_other()
            for session, (left, right) in enumerate(sessions):
                ids = mx.arange(33 + 2 * session, dtype=mx.uint32)[None, :] % 512
                expected, actual = reference(ids, left), optimized(ids, right)
                self.assert_exact(actual, expected)
                self.assert_caches(right, left)
                next_ids.append(mx.argmax(expected[:, -1], -1).astype(mx.uint32)[:, None])
            self.assertTrue(all(gdn._capture_calls == 0 for gdn in gdns))
            # Partial M2/M3 commits discard rejected rows; each following M1
            # processes the correction, just like MTPLX pending_primary.
            for length, keep in ((1, 1), (2, 1), (1, 1), (2, 2),
                                 (3, 1), (1, 1), (3, 2), (1, 1), (3, 3), (1, 1)):
                for session, (left, right) in enumerate(sessions):
                    ids = mx.repeat(next_ids[session], length, axis=1)
                    expected, hidden, left_capture = self.capture.forward_with_gdn_capture(
                        reference, ids, left, return_hidden=True, capture_backend="stock")
                    actual, actual_hidden, right_capture = self.capture.forward_with_gdn_capture(
                        optimized, ids, right, return_hidden=True, capture_backend="stock")
                    self.assert_exact(actual, expected)
                    self.assert_exact(actual_hidden, hidden)
                    self.assert_exact(mx.argmax(actual, -1), mx.argmax(expected, -1))
                    self.assert_caches(right, left)
                    self.assertEqual(left_capture.keys(), right_capture.keys())
                    for layer_index in left_capture:
                        self.assert_capture(right_capture[layer_index], left_capture[layer_index])
                        self.assertEqual(right_capture[layer_index]["conv_states"].shape[1], length)
                        self.assertEqual(right_capture[layer_index]["states"].shape[1], length)
                    self.assertTrue(self.capture.commit_captured_prefix(left, left_capture, keep, length))
                    self.assertTrue(self.capture.commit_captured_prefix(right, right_capture, keep, length))
                    self.assert_caches(right, left)
                    for layer_index, captured in right_capture.items():
                        self.assert_exact(right[layer_index][0], captured["conv_states"][:, keep - 1])
                        self.assert_exact(right[layer_index][1], captured["states"][:, keep - 1])
                    next_ids[session] = mx.argmax(expected[:, keep - 1], -1).astype(mx.uint32)[:, None]
            for gdn in gdns:
                self.assertEqual(gdn._capture_calls, 20)
                self.assertEqual(gdn._capture_traces, 3)
                self.assertIsNone(gdn.in_proj_qkv._fusion.outputs)
            # The same global dispatcher served the reference without patching it.
            self.assertFalse(hasattr(reference.layers[0].linear_attn, "_compiled_capture"))
            undo()
            undo()
            self.assertIs(self.capture._gdn_forward_with_capture, original)
            for gdn in gdns:
                for name in ("_compiled_capture", "_capture_calls", "_capture_traces"):
                    self.assertFalse(hasattr(gdn, name))
            after = tree_flatten(optimized.parameters())
            self.assertEqual([name for name, _ in before], [name for name, _ in after])
            for (_, old), (_, current) in zip(before, after):
                self.assertIs(current, old)
            expected, left_capture = self.capture.forward_with_gdn_capture(
                reference, next_ids[0], sessions[0][0], capture_backend="stock")
            actual, right_capture = self.capture.forward_with_gdn_capture(
                optimized, next_ids[0], sessions[0][1], capture_backend="stock")
            self.assert_exact(actual, expected)
            self.assert_caches(sessions[0][1], sessions[0][0])
        finally:
            undo()
            undo_engine()
            undo_reference()

    def test_native_gdn_prefix_tapes_and_fallbacks(self):
        model = make_model()
        gdn = model.layers[0].linear_attn
        original = self.capture._gdn_forward_with_capture
        undo = candidate.apply(model)
        try:
            for batch, length, dtype, masked, training in (
                    (1, 1, mx.float16, False, False),
                    (1, 2, mx.float16, False, False),
                    (1, 3, mx.float16, False, False),
                    (1, 4, mx.float16, False, False),
                    (2, 3, mx.float16, False, False),
                    (1, 3, mx.float32, False, False),
                    (1, 3, mx.float16, True, False),
                    (1, 3, mx.float16, False, True)):
                x = mx.random.normal((batch, length, 512)).astype(dtype)
                conv = mx.random.normal((batch, 3, gdn.conv_dim)).astype(dtype)
                state = mx.random.normal((batch, gdn.num_v_heads, 128, 128)).astype(mx.float32)
                mx.eval(x, conv, state)
                left, right = cache_from(conv, state), cache_from(conv, state)
                mask = mx.ones((batch, length), dtype=mx.bool_) if masked else None
                gdn.train(training)
                expected, left_capture = original(gdn, x, mask, left, capture_backend="stock")
                actual, right_capture = self.capture.gdn_forward_with_capture(
                    gdn, x, mask, right, capture_backend="stock")
                self.assert_exact(actual, expected)
                self.assert_capture(right_capture, left_capture)
                self.assert_caches([right], [left])
                gdn.eval()
            self.assertEqual(gdn._capture_calls, 3)
            self.assertEqual(gdn._capture_traces, 3)
            # Cold and absent caches also delegate to the original zero-state path.
            x = mx.ones((1, 3, 512), dtype=mx.float16)
            for absent in (False, True):
                left, right = (None, None) if absent else (ArraysCache(2), ArraysCache(2))
                expected, left_capture = original(gdn, x, cache=left, capture_backend="stock")
                actual, right_capture = self.capture.gdn_forward_with_capture(
                    gdn, x, cache=right, capture_backend="stock")
                self.assert_exact(actual, expected)
                self.assert_capture(right_capture, left_capture)
            self.assertEqual(gdn._capture_calls, 3)
        finally:
            undo()

    def test_dynamic_environment_and_cache_guards(self):
        model = make_model()
        gdn = model.layers[0].linear_attn
        undo = candidate.apply(model)
        try:
            x = mx.ones((1, 3, 512), dtype=mx.float16)
            cache = cache_from(mx.zeros((1, 3, gdn.conv_dim), mx.float16),
                               mx.zeros((1, gdn.num_v_heads, 128, 128), mx.float32))
            original = self.capture.gdn_forward_with_capture
            out, _ = original(gdn, x, cache=cache, capture_backend="stock")
            mx.eval(out, cache.state)
            for key in candidate._ENV_BRANCHES:
                with patch.dict(os.environ, {key: "1"}):
                    self.assertFalse(candidate._ready(gdn, x, None, cache, "stock", self.capture))
            with patch.object(self.capture, "_linear_gated_delta_kernel", None):
                self.assertFalse(candidate._ready(gdn, x, None, cache, "stock", self.capture))
            with patch.dict(os.environ, {"MTPLX_CAPTURE_CUSTOM_KERNEL": "linear-gdn"}):
                self.assertFalse(candidate._ready(gdn, x, None, cache, None, self.capture))
            self.assertFalse(candidate._ready(gdn, x, None, cache, "linear-gdn", self.capture))
            for attr in ("lengths", "left_padding"):
                setattr(cache, attr, mx.array([3]))
                self.assertFalse(candidate._ready(gdn, x, None, cache, "stock", self.capture))
                setattr(cache, attr, None)
            gdn.out_proj.train()
            self.assertFalse(candidate._ready(gdn, x, None, cache, "stock", self.capture))
            gdn.out_proj.eval()
            # Cache leaf policy is dynamic Python behavior, not a frozen trace.
            with patch.dict(os.environ, {"MTPLX_CAPTURE_CONTIGUOUS_GDN_STATE": "1"}):
                out, captured = original(gdn, x, cache=cache, capture_backend="stock")
                mx.eval(out, cache.state)
                self.assert_exact(cache[1], captured["states"][:, -1])
            self.assertEqual(gdn._capture_calls, 2)
            self.assertEqual(gdn._capture_traces, 1)
        finally:
            undo()

    def test_atomic_validation_and_compile_setup_failure(self):
        original = self.capture._gdn_forward_with_capture
        for corruption in ("projection", "sharding"):
            model = make_model()
            gdn = model.layers[0].linear_attn
            if corruption == "projection":
                class Unknown(nn.QuantizedLinear):
                    pass
                gdn.out_proj.__class__ = Unknown
            else:
                gdn.sharding_group = object()
            with self.assertRaises((TypeError, ValueError)):
                candidate.apply(model)
            self.assertIs(self.capture._gdn_forward_with_capture, original)
            self.assertFalse(hasattr(gdn, "_compiled_capture"))
        model = make_model()
        with patch.object(candidate, "_compile_capture", side_effect=RuntimeError("test compile setup")):
            with self.assertRaisesRegex(RuntimeError, "test compile setup"):
                candidate.apply(model)
        self.assertIs(self.capture._gdn_forward_with_capture, original)
        self.assertFalse(hasattr(model.layers[0].linear_attn, "_compiled_capture"))


if __name__ == "__main__":
    unittest.main()
