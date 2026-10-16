"""Selection, load ordering, and native-runtime integration."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx

from mtplx import local_optimizations, runtime
from .qwen27.test_decode_fusion import make_model as make_dense
from .qwen35.test_island_compile import make_full_stack_model


class ProfileTests(unittest.TestCase):
    def test_profile_preserves_original_draft_head(self):
        from mtplx.draft_lm_head import _install_draft_lm_head

        report = _install_draft_lm_head(SimpleNamespace(local_optimization="qwen27-mtp"),
                                        bits=4, group_size=64, mode="affine")
        self.assertFalse(report["installed"])
        self.assertEqual(report["reason"], "local_optimization_original_weights")

    def test_selection_and_rejection(self):
        dense = {"model_type": "qwen3_5", "text_config": {"num_experts": 0}}
        with patch.dict(os.environ, {"MTPLX_LOCAL_OPT": ""}):
            self.assertIsNone(local_optimizations.resolve(dense))
        with patch.dict(os.environ, {"MTPLX_LOCAL_OPT": "typo"}):
            with self.assertRaisesRegex(ValueError, "MTPLX_LOCAL_OPT"):
                local_optimizations.resolve(dense)
        with patch.dict(os.environ, {"MTPLX_LOCAL_OPT": "qwen27-mtp"}):
            self.assertEqual(local_optimizations.resolve(dense), "qwen27-mtp")
            with self.assertRaisesRegex(ValueError, "original weights"):
                local_optimizations.resolve(dense, proj_requant="4")
        with patch.dict(os.environ, {"MTPLX_LOCAL_OPT": "qwen35-ar"}):
            with self.assertRaisesRegex(ValueError, "256 experts"):
                local_optimizations.resolve(dense)

    def test_loader_installs_before_injection_and_skips_upstream_fusions(self):
        for name, factory in (("qwen35-ar", make_full_stack_model), ("qwen27-mtp", make_dense)):
            model = factory()
            config = {"model_type": model.args.model_type, "text_config": model.args.text_config}
            def inject(loaded, *_args):
                self.assertTrue(hasattr(loaded.layers[0].linear_attn, "_local_capture"))
                return True
            with TemporaryDirectory() as directory, \
                    patch.dict(os.environ, {"MTPLX_LOCAL_OPT": name}), \
                    patch.object(runtime, "load_config", return_value=config), \
                    patch.object(runtime, "_load_base_model", return_value=(model, object())), \
                    patch.object(runtime, "inject_mtp_support", side_effect=inject) as injection, \
                    patch.object(runtime, "validate_mtp_support", return_value=True), \
                    patch("mtplx.attention_split.configure_split_full_attention") as upstream:
                loaded = runtime.load(Path(directory), mtp=True)
                self.assertEqual(loaded.local_optimization, name)
                self.assertEqual(loaded.mtp_enabled, name == "qwen27-mtp")
                self.assertEqual(injection.call_count, int(name == "qwen27-mtp"))
                upstream.assert_not_called()
                cache = model.make_cache()
                out = loaded.forward_ar(mx.array([[1, 2, 3]]), cache)
                mx.eval(out, [c.state for c in cache])
                out = loaded.forward_ar(mx.array([[4]]), cache)
                mx.eval(out, [c.state for c in cache])
                if name == "qwen35-ar":
                    self.assertEqual(model._island_calls, 1)
