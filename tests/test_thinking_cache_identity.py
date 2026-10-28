import pytest

from mtplx.server.openai import _policy_fingerprint
from tests.test_bank_history_policy import _state


@pytest.mark.parametrize("profile", ["qwen35-ar", "qwen35-mtp"])
def test_qwen_thinking_flip_preserves_existing_on_cache_identity(profile):
    state = _state(mtp_enabled=True)
    legacy_on = _policy_fingerprint(state, thinking_enabled=True)
    state.runtime.local_optimization = profile
    assert _policy_fingerprint(state, thinking_enabled=False) == legacy_on
    assert _policy_fingerprint(state, thinking_enabled=True) == legacy_on


def test_other_backends_keep_their_thinking_identity():
    state = _state(mtp_enabled=True)
    state.runtime.local_optimization = "qwen27-mtp"
    assert _policy_fingerprint(state, thinking_enabled=False) != _policy_fingerprint(state, thinking_enabled=True)


@pytest.mark.parametrize("field,value", [("template_hash", "other-template"), ("draft_head_identity", "other-draft")])
def test_thinking_flip_does_not_bypass_other_identity_fields(field, value):
    state = _state(mtp_enabled=True)
    state.runtime.local_optimization = "qwen35-mtp"
    before = _policy_fingerprint(state, thinking_enabled=True)
    setattr(state, field, value)
    assert _policy_fingerprint(state, thinking_enabled=False) != before
