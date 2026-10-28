from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.engine_session import EngineSessionManager
from mtplx.server import openai
from mtplx.session_bank import SessionBank
from tests.test_server_openai import (
    _fake_generation,
    _fake_state,
    _fake_streaming_session_state,
)


@pytest.mark.parametrize(
    ("path", "input_body"),
    [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say OK"}]}),
        ("/v1/responses", {"input": "Say OK"}),
    ],
)
@pytest.mark.parametrize(
    ("metadata", "headers", "expected_id", "expected_source"),
    [
        ({}, {}, "cache-key", "metadata.session_id"),
        ({"session_id": "metadata-id"}, {}, "metadata-id", "metadata.session_id"),
        (
            {"session_id": "metadata-id"},
            {"x-mtplx-session-id": "header-id"},
            "header-id",
            "header.x-mtplx-session-id",
        ),
        ({"chat_id": "alias-id"}, {}, "cache-key", "metadata.session_id"),
    ],
)
def test_chat_and_responses_prompt_cache_key_select_the_same_session(
    monkeypatch, path, input_body, metadata, headers, expected_id, expected_source
):
    state = _fake_streaming_session_state()
    captured = {}

    def fake_generation(_state, _prompt_ids, **kwargs):
        captured.update(kwargs)
        return _fake_generation("OK")

    monkeypatch.setattr(openai, "_run_generation", fake_generation)
    monkeypatch.setattr(
        openai,
        "_store_retokenized_history_snapshot",
        lambda *_args, **_kwargs: {"stored": False, "reason": "test"},
    )
    with TestClient(openai.create_app(state)) as client:
        response = client.post(
            path,
            headers=headers,
            json={
                **input_body,
                "prompt_cache_key": "cache-key",
                "metadata": metadata,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200, response.text
    assert captured["session_id"] == expected_id
    assert captured["request_observability"]["request_session_source"] == expected_source
    assert state.sessions.peek(expected_id) is not None


@pytest.mark.parametrize("key", [None, "cache-key", " padded-key "])
def test_prompt_cache_key_does_not_mutate_request_metadata(key):
    request = openai.ChatCompletionRequest(
        prompt_cache_key=key, metadata={"client": "test"}
    )
    metadata = openai._request_metadata(request)

    assert request.metadata == {"client": "test"}
    if key is None:
        assert metadata is request.metadata
    else:
        assert metadata == {"client": "test", "session_id": key}


@pytest.mark.parametrize(
    ("path", "input_body"),
    [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say OK"}]}),
        ("/v1/responses", {"input": "Say OK"}),
    ],
)
@pytest.mark.parametrize(
    ("key", "status"),
    [("", 400), (" \t\n", 400), (17, 422), (True, 422), ([], 422), ({}, 422)],
)
def test_chat_and_responses_reject_invalid_prompt_cache_keys(path, input_body, key, status):
    with TestClient(openai.create_app(_fake_state())) as client:
        response = client.post(
            path,
            json={
                **input_body,
                "prompt_cache_key": key,
            },
        )
    assert response.status_code == status
    if status == 400:
        assert "prompt_cache_key must not be empty" in response.text


def test_same_prompt_cache_key_cannot_restore_a_different_token_prefix():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    manager = EngineSessionManager(bank=bank)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    request = openai.ChatCompletionRequest(prompt_cache_key="cache-key")
    metadata = openai._request_metadata(request)
    session_id, source = manager.resolve_session_id(
        metadata=metadata, prompt_ids=[1, 2, 3]
    )
    assert (session_id, source) == ("cache-key", "metadata.session_id")
    assert bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits="logits",
        hidden="hidden",
        session_id=session_id,
        nbytes_override=128,
    ) is not None

    changed_session_id, _ = manager.resolve_session_id(
        metadata=metadata, prompt_ids=[9, 2, 3]
    )
    assert changed_session_id == session_id
    assert bank.restore(
        runtime, [9, 2, 3], session_id=changed_session_id, cache_factory=lambda: []
    ) is None
    assert bank.near_prefix_candidates([9, 2, 3], min_matched_tokens=1) == []

    restored = bank.restore(
        runtime, [1, 2, 3, 4], session_id=session_id, cache_factory=lambda: []
    )
    assert restored is not None
    assert restored.entry.prefix_len == 3
