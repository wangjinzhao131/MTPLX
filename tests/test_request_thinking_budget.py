from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mtplx.server import openai as oa
from mtplx.thinking_budget import HardThinkingBudgetConfig
from tests.test_server_openai import (
    _fake_generation, _fake_streaming_session_state, _guard_request_state,
)


class MarkerTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"<think>": [101], "</think>": [102], "</think>\n\n": [102, 7, 7]}.get(text, [1, 2])


def state():
    result = _guard_request_state([])
    result.runtime.tokenizer = MarkerTokenizer()
    return result


@pytest.mark.parametrize('value', [0, -1, 1.5, True, '96'])
def test_budget_is_a_strict_positive_integer(value):
    with pytest.raises(ValidationError):
        oa.ChatCompletionRequest(thinking_token_budget=value)


@pytest.mark.parametrize('ids,starts', [([101, 8], True), ([101, 8, 102], False), ([102, 101], True), ([1, 2], False)])
def test_explicit_budget_precedes_operator_default_and_chat_scope(monkeypatch, ids, starts):
    monkeypatch.setenv('MTPLX_THINKING_BUDGET', 'off')
    monkeypatch.setenv('MTPLX_THINKING_NOVELTY_CLOSE', '1')
    config = oa._thinking_guard_config_for_request(state(), prompt_ids=ids, request_observability={
        'request_enable_thinking': True, 'request_thinking_token_budget': 96, 'request_tool_count': 0,
    })
    assert isinstance(config, HardThinkingBudgetConfig)
    assert config.budget_tokens == 96
    assert config.forced_close_ids == (102, 7, 7)
    assert config.starts_in_think is starts
    assert config.reentry_ban_tokens == 0 and config.novelty_close is False


def test_disabled_thinking_does_not_create_budget_guard():
    assert oa._thinking_guard_config_for_request(state(), prompt_ids=[101, 102], request_observability={
        'request_enable_thinking': False, 'request_thinking_token_budget': 96,
    }) is None


def test_budget_enforcement_telemetry_is_public():
    telemetry = {"budget_tokens": 96, "engaged": "budget"}
    assert oa._public_mtplx_stats({"stats": {"thinking_guard": telemetry}})["thinking_guard"] == telemetry


@pytest.mark.parametrize('bad_kind', ['markers', 'close_sequence'])
def test_unsupported_tokenizer_cannot_silently_ignore_budget(bad_kind):
    s = state()
    original = s.runtime.tokenizer.encode
    s.runtime.tokenizer.encode = lambda text, **kw: [1, 2] if bad_kind == 'markers' or text == '</think>\n\n' else original(text, **kw)
    with pytest.raises(HTTPException) as exc:
        oa._thinking_guard_config_for_request(s, prompt_ids=[101], request_observability={
            'request_enable_thinking': True, 'request_thinking_token_budget': 96,
        })
    assert exc.value.status_code == 400


def test_http_forwards_explicit_budget_to_generation(monkeypatch):
    s = _fake_streaming_session_state()
    captured = {}
    def generate(_state, _ids, **kwargs):
        captured.update(kwargs['request_observability'])
        return _fake_generation('OK')
    monkeypatch.setattr(oa, '_run_generation', generate)
    monkeypatch.setattr(oa, '_store_retokenized_history_snapshot', lambda *a, **kw: {'stored': False, 'reason': 'test'})
    with TestClient(oa.create_app(s)) as client:
        response = client.post('/v1/chat/completions', json={
            'messages': [{'role': 'user', 'content': 'test'}],
            'enable_thinking': True, 'thinking_token_budget': 96, 'max_tokens': 32,
        })
    assert response.status_code == 200, response.text
    assert captured['request_thinking_token_budget'] == 96
    assert captured['request_enable_thinking'] is True


def test_unsupported_backend_rejected_before_generation():
    s = _fake_streaming_session_state()
    s.runtime.backend_id = 'gemma4_assistant'
    with TestClient(oa.create_app(s)) as client:
        response = client.post('/v1/chat/completions', json={
            'messages': [{'role': 'user', 'content': 'test'}],
            'enable_thinking': True, 'thinking_token_budget': 96,
        })
    assert response.status_code == 400, response.text
    assert 'unconstrained serial text' in response.text
