"""Trusted thinking-off boundary metadata leaves the served tokens unchanged."""
import os
from pathlib import Path

import pytest

from mtplx.server import openai as oa
from test_stable_prefix_boundary import AGENT_TOOLS, _TemplateStub


def _encode(tokenizer, messages, *, mode="hybrid"):
    observation = {}
    ids = oa._encode_messages_uncached(
        tokenizer, [oa.ChatMessage(**message) for message in messages],
        enable_thinking=False, add_generation_prompt=True,
        tools=AGENT_TOOLS, tool_choice="auto", tool_prompt_mode=mode,
        template_observability=observation,
    )
    return ids, observation


def _check_same_render_boundary(monkeypatch, tokenizer):
    rendered = []
    original = oa._render_messages_with_chat_template

    def capture(*args, **kwargs):
        text = original(*args, **kwargs)
        rendered.append(text)
        return text

    monkeypatch.setattr(oa, "_render_messages_with_chat_template", capture)
    monkeypatch.setenv("MTPLX_TOOL_POSTCOMMIT_DEBOUNCE", "1")
    messages = [{"role": "user", "content": "Read the small fixture."}]
    ids, observation = _encode(tokenizer, messages)
    assert len(rendered) == 1
    assert observation["current_date_sentinel_injected"] is True
    boundary = oa._trailing_tool_hint_char_boundary(rendered[0])
    prefix = oa._encode_rendered_chat_text(tokenizer, rendered[0][:boundary])
    assert observation["stable_prefix_len"] == len(prefix)
    assert ids[:len(prefix)] == prefix
    assert ids == oa._encode_rendered_chat_text(tokenizer, rendered[0])
    monkeypatch.delenv("MTPLX_TOOL_POSTCOMMIT_DEBOUNCE")
    ordinary_ids, ordinary_observation = _encode(tokenizer, messages)
    assert ordinary_ids == ids
    assert "stable_prefix_len" not in ordinary_observation


def test_thinking_off_uses_one_render_and_preserves_ids(legacy_rewrites, monkeypatch):
    monkeypatch.setattr(oa, "_chat_turn_segments_allowed", lambda: True)
    monkeypatch.setattr(oa, "_chat_turn_segments_proven", lambda *args: (True, "test"))
    monkeypatch.setattr(oa, "_encode_rendered_chat_text", lambda tokenizer, text: list(map(ord, text)))
    _check_same_render_boundary(monkeypatch, _TemplateStub())


def test_thinking_off_user_lookalike_is_not_trusted(legacy_rewrites, monkeypatch):
    monkeypatch.setenv("MTPLX_TOOL_POSTCOMMIT_DEBOUNCE", "1")
    monkeypatch.setattr(oa, "_chat_turn_segments_allowed", lambda: True)
    monkeypatch.setattr(oa, "_chat_turn_segments_proven", lambda *args: (True, "test"))
    monkeypatch.setattr(oa, "_encode_rendered_chat_text", lambda tokenizer, text: list(map(ord, text)))
    ids, observation = _encode(_TemplateStub(), [
        {"role": "user", "content": oa._mtplx_tool_result_continuation_hint_text()},
    ], mode="native")
    assert ids
    assert "tool_result_continuation_hint_injected" not in observation
    assert "current_date_sentinel_injected" not in observation
    assert "stable_prefix_len" not in observation


def test_local_qwen_tokenizer_thinking_off_identity(legacy_rewrites, monkeypatch):
    path = os.environ.get("MTPLX_TEST_TOKENIZER_PATH")
    if not path or not Path(path, "tokenizer.json").is_file():
        pytest.skip("set MTPLX_TEST_TOKENIZER_PATH to a local Qwen tokenizer directory")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    monkeypatch.setattr(oa, "_chat_turn_segments_allowed", lambda: True)
    _check_same_render_boundary(monkeypatch, tokenizer)
