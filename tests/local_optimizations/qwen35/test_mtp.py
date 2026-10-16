"""Real MTP injection, hidden states and rejected-prefix cache correctness."""

from contextlib import ExitStack
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten

from mtplx import gdn_capture, local_optimizations, mtp_patch
from mtplx.local_optimizations.capture import _ENV_BRANCHES
from .test_island_compile import make_full_stack_model
from mtplx.local_optimizations.qwen35 import engine, island_compile


class MTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stack = ExitStack()
        cls.addClassCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {**{k: '0' for k in _ENV_BRANCHES},
            'MTPLX_CAPTURE_CUSTOM_KERNEL': 'stock', 'MTPLX_CAPTURE_CONTIGUOUS_GDN_STATE': '0'}))
        mx.random.seed(731)
        cls.reference, cls.candidate = make_full_stack_model(), make_full_stack_model()
        cls.candidate.load_weights(tree_flatten(cls.reference.parameters()))
        mx.eval(cls.reference.parameters(), cls.candidate.parameters())
        with patch.object(island_compile, 'apply', return_value=lambda: None):
            stack.callback(engine.apply(cls.reference))
        cls.undo_candidate = local_optimizations.apply(cls.candidate, 'qwen35-mtp')
        stack.callback(cls.undo_candidate)
        for model in (cls.reference, cls.candidate):
            config = {'model_type': model.args.model_type,
                      'text_config': {**model.args.text_config, 'mtp_num_hidden_layers': 1}}
            # The real injector installs the wrappers; target tests do not use draft weights.
            with patch.object(mtp_patch, '_load_embedded_mtp_weights',
                              return_value={'fc.weight': mx.zeros((256, 512))}):
                assert mtp_patch.inject_mtp_support(model, Path('/nonexistent-mtp-test'), config)

    def exact(self, actual, expected):
        mx.eval(actual, expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(mx.array_equal(actual, expected).item(),
                        f'max error {mx.max(mx.abs(actual - expected)).item()}')

    def caches_equal(self, actual, expected):
        for left, right in zip(actual, expected):
            self.assertEqual(getattr(left, 'offset', None), getattr(right, 'offset', None))
            for a, b in zip(left.state, right.state):
                self.exact(a, b)

    def session(self):
        caches = self.reference.make_cache(), self.candidate.make_cache()
        ids = mx.array([[1, 2, 3]])
        self.exact(self.candidate(ids, cache=caches[1]), self.reference(ids, cache=caches[0]))
        self.caches_equal(caches[1], caches[0])
        return caches

    def test_injected_forward_keeps_islands_and_hidden_variants(self):
        left, right = self.session()
        calls = self.candidate._island_calls
        for variant, emit in (('pre_norm', True), ('post_norm', True), (None, False)):
            ids = mx.array([[4]])
            expected = self.reference(ids, cache=left, return_hidden=True,
                                      hidden_variant=variant, emit_logits=emit, logits_keep=1)
            actual = self.candidate(ids, cache=right, return_hidden=True,
                                    hidden_variant=variant, emit_logits=emit, logits_keep=1)
            if emit:
                self.exact(actual[0], expected[0])
            else:
                self.assertIsNone(actual[0])
            self.exact(actual[1], expected[1])
            self.caches_equal(right, left)
        self.assertEqual(self.candidate._island_calls - calls, 3)
        # Multi-token forward must use the MTP wrapper, including its hidden-state contract.
        expected = self.reference(mx.array([[5, 6]]), cache=left, return_hidden=True)
        actual = self.candidate(mx.array([[5, 6]]), cache=right, return_hidden=True)
        self.exact(actual[0], expected[0])
        self.exact(actual[1], expected[1])
        self.caches_equal(right, left)

    def test_capture_rejected_prefix_and_correction(self):
        left, right = self.session()
        for length, keep in ((1, 1), (2, 1), (1, 1), (2, 2), (3, 1), (3, 2), (3, 3), (1, 1)):
            ids = mx.array([[7] * length])
            expected, hidden, ref_capture = gdn_capture.forward_with_gdn_capture(
                self.reference, ids, left, return_hidden=True, capture_backend='stock')
            actual, actual_hidden, capture = gdn_capture.forward_with_gdn_capture(
                self.candidate, ids, right, return_hidden=True, capture_backend='stock')
            self.exact(actual, expected)
            self.exact(actual_hidden, hidden)
            self.assertEqual(capture.keys(), ref_capture.keys())
            for index in capture:
                for key in ('conv_states', 'states'):
                    self.exact(capture[index][key], ref_capture[index][key])
            self.assertTrue(gdn_capture.commit_captured_prefix(left, ref_capture, keep, length))
            self.assertTrue(gdn_capture.commit_captured_prefix(right, capture, keep, length))
            self.caches_equal(right, left)
        self.assertGreater(self.candidate._capture_island_calls, 0)
        for layer in self.candidate.layers:
            if layer.is_linear:
                self.assertGreater(layer.linear_attn._capture_calls, 0)

    def test_capture_fallback_and_pre_norm_hidden(self):
        for length, environment in ((4, {}), (2, {'MTPLX_TARGET_LAYER_EVAL_EVERY': '1'})):
            left, right = self.session()
            calls = self.candidate._capture_island_calls
            with patch.dict(os.environ, environment):
                ids = mx.array([[7] * length])
                expected, hidden, ref_capture = gdn_capture.forward_with_gdn_capture(
                    self.reference, ids, left, return_hidden=True, hidden_variant='pre_norm',
                    capture_backend='stock')
                actual, actual_hidden, capture = gdn_capture.forward_with_gdn_capture(
                    self.candidate, ids, right, return_hidden=True, hidden_variant='pre_norm',
                    capture_backend='stock')
                self.exact(actual, expected)
                self.exact(actual_hidden, hidden)
                self.caches_equal(right, left)
                for index in capture:
                    for key in ('conv_states', 'states'):
                        self.exact(capture[index][key], ref_capture[index][key])
            self.assertEqual(self.candidate._capture_island_calls, calls)

    def test_z_undo_preserves_injected_mtp_wrapper(self):
        left, right = self.session()
        type(self).undo_candidate()
        self.assertFalse(hasattr(self.candidate, '_island_blocks'))
        self.assertTrue(callable(self.candidate.mtp_forward))
        ids = mx.array([[8, 9]])
        expected = self.reference(ids, cache=left, return_hidden=True)
        actual = self.candidate(ids, cache=right, return_hidden=True)
        self.exact(actual[0], expected[0])
        self.exact(actual[1], expected[1])
        self.caches_equal(right, left)
