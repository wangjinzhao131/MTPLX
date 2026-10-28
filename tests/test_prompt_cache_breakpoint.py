"""Explicit text boundaries must describe the exact prompt the model receives."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mtplx.chat_encode_cache import ChatEncodeCache
from mtplx.server import openai as oa
from tests.test_server_openai import (
    QwenToolHistoryBoundaryTokenizer,
    _fake_state,
    _named_tool_schema,
)


def part(text, marked=False):
    result = {"type": "text", "text": text}
    if marked:
        result["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return result


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("mode", ["native", "hybrid", "compact"])
def test_marker_preserves_prompt_ids_and_tracks_exact_text_edge(monkeypatch, thinking, mode):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    tokenizer = QwenToolHistoryBoundaryTokenizer()
    messages = [
        oa.ChatMessage(role="system", content="Stable system instructions."),
        oa.ChatMessage(role="user", content=[part("Reference alpha. ", True), part("Question one.")]),
    ]
    original = deepcopy(messages)
    plain = deepcopy(messages)
    plain[1].content[0].pop("prompt_cache_breakpoint")
    tools = [_named_tool_schema("read_file")]
    kwargs = dict(enable_thinking=thinking, tools=tools, tool_prompt_mode=mode)
    observation = {}
    ids = oa._encode_messages(tokenizer, messages, template_observability=observation, **kwargs)
    assert ids == oa._encode_messages(tokenizer, plain, **kwargs)
    boundary = observation["explicit_cache_boundary"]
    assert tokenizer.decode(ids[:boundary]).endswith("Reference alpha. ")
    assert "Question one." in tokenizer.decode(ids[boundary:])
    assert messages == original
    assert "MTPLXCACHEBOUNDARY" not in tokenizer.decode(ids)
    assert boundary != observation.get("stable_prefix_len")


def test_marker_survives_adjacent_user_normalization_and_repeated_text(monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    tokenizer = QwenToolHistoryBoundaryTokenizer()
    messages = [
        oa.ChatMessage(role="user", content="Repeated reference."),
        oa.ChatMessage(role="user", content=[part("Repeated reference.", True), part(" Changed tail.")]),
    ]
    observation = {}
    ids = oa._encode_messages(tokenizer, messages, enable_thinking=True, template_observability=observation)
    prefix = tokenizer.decode(ids[:observation["explicit_cache_boundary"]])
    assert prefix.count("Repeated reference.") == 2
    assert prefix.endswith("Repeated reference.")


def test_encode_cache_keeps_boundary_metadata_and_distinguishes_edges(monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "on")
    monkeypatch.setattr(oa, "GLOBAL_CHAT_ENCODE_CACHE", ChatEncodeCache(max_entries=8))
    tokenizer = QwenToolHistoryBoundaryTokenizer()
    messages = [oa.ChatMessage(role="user", content=[part("Shared.", True), part(" Tail.")])]
    first, second = {}, {}
    ids = oa._encode_messages(tokenizer, messages, enable_thinking=True, template_observability=first)
    assert oa._encode_messages(tokenizer, messages, enable_thinking=True, template_observability=second) == ids
    assert second["chat_encode_cache"] == "hit"
    assert second["explicit_cache_boundary"] == first["explicit_cache_boundary"]
    moved = [oa.ChatMessage(role="user", content=[part("Shared."), part(" Tail.", True)])]
    last = {}
    assert oa._encode_messages(tokenizer, moved, enable_thinking=True, template_observability=last) == ids
    assert last["explicit_cache_boundary"] > first["explicit_cache_boundary"]
    assert last["chat_encode_cache"] == "miss"


class MergedTokenizer(QwenToolHistoryBoundaryTokenizer):
    def encode(self, text, **kwargs):
        result = []
        while text:
            if text.startswith("AB"):
                result.append(1000000)
                text = text[2:]
            else:
                result.append(ord(text[0]))
                text = text[1:]
        return result

    def decode(self, ids, **kwargs):
        return "".join("AB" if int(value) == 1000000 else chr(int(value)) for value in ids)


def test_bpe_token_crossing_requested_edge_is_rejected(monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    tokenizer = MergedTokenizer()
    messages = [oa.ChatMessage(role="user", content=[part("A", True), part("B")])]
    with pytest.raises(HTTPException, match="not an exact served token boundary"):
        oa._encode_messages(tokenizer, messages, enable_thinking=False)


def test_canonical_token_splice_remaps_boundary_on_actual_served_ids():
    tokenizer = MergedTokenizer()
    original = [ord(c) for c in "AB prefix. tail."]
    prefix = "AB prefix."
    import hashlib
    observation = {
        "explicit_cache_boundary": len(prefix),
        "explicit_cache_boundary_chars": len(prefix),
        "explicit_cache_boundary_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
    }
    spliced = tokenizer.encode("AB prefix. tail.")
    oa._refresh_explicit_cache_boundary(tokenizer, spliced, observation)
    assert observation["explicit_cache_boundary"] == len(prefix) - 1
    assert tokenizer.decode(spliced[:observation["explicit_cache_boundary"]]) == prefix
    assert tokenizer.decode(original) == tokenizer.decode(spliced)
    with pytest.raises(HTTPException, match="changed during canonicalization"):
        oa._refresh_explicit_cache_boundary(tokenizer, tokenizer.encode("AB changed. tail."), observation)


@pytest.mark.parametrize("marker", [None, True, "explicit", {}, {"mode": "auto"}, {"mode": "explicit", "ttl": "1h"}])
def test_rejects_malformed_marker(marker):
    message = oa.ChatMessage(role="user", content=[{"type": "text", "text": "prefix", "prompt_cache_breakpoint": marker}])
    with pytest.raises(HTTPException, match="on a text part"):
        oa._prepare_explicit_cache_breakpoint([message])


def test_rejects_multiple_markers_and_vision_without_fetching():
    with pytest.raises(HTTPException, match="only one"):
        oa._prepare_explicit_cache_breakpoint([oa.ChatMessage(role="user", content=[part("one", True), part("two", True)])])
    with pytest.raises(HTTPException, match="text-only"):
        oa._prepare_explicit_cache_breakpoint([
            oa.ChatMessage(role="system", content=[part("prefix", True)]),
            oa.ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": "https://invalid/image"}}]),
        ])


def test_private_marker_does_not_survive_client_json_or_changed_content(monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    prepared = oa._prepare_explicit_cache_breakpoint([
        oa.ChatMessage(role="user", content=[part("prefix", True), part(" tail")]),
    ])[0]
    assert prepared._mtplx_cache_breakpoint == ("prefix tail", len("prefix"))
    public = prepared.model_dump()
    assert "prompt_cache_breakpoint" not in str(public)
    assert "_mtplx_cache_breakpoint" not in public
    assert oa.ChatMessage.model_validate(public)._mtplx_cache_breakpoint is None
    with pytest.raises(HTTPException, match="content changed during normalization"):
        oa._encode_messages(QwenToolHistoryBoundaryTokenizer(), [prepared.model_copy(update={"content": "changed"})], enable_thinking=True)


def test_http_rejects_marker_with_explicit_cache_bypass():
    with TestClient(oa.create_app(_fake_state())) as client:
        response = client.post("/v1/chat/completions", headers={"x-mtplx-cache-mode": "off"}, json={
            "messages": [{"role": "user", "content": [part("reference", True), part(" tail")]}],
            "max_tokens": 64,
        })
    assert response.status_code == 400, response.text
    assert "serial Qwen3.5 text cache path" in response.text


def test_final_bank_values_carry_independent_boundary():
    final = SimpleNamespace(final_trunk_cache=[], final_logits=None, final_hidden=None,
                            gdn_boundaries=[(7, None, None)], explicit_cache_boundary=7)
    values = oa._generation_final_bank_values(_fake_state(), final, prompt_ids=[1, 2], final_token_ids=[1, 2, 3])
    assert values["token_ids"] == [1, 2, 3]
    assert values["explicit_cache_boundary"] == 7


@pytest.fixture(scope="module")
def served_tokenizer():
    """Optional tokenizer-only coverage; does not open a model weight file."""
    import json
    from pathlib import Path
    from mtplx.runtime import _load_tokenizer_resilient

    path = Path.home() / ".mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16"
    if not (path / "tokenizer.json").is_file():
        pytest.skip("local Qwen3.5 tokenizer fixture unavailable")
    return _load_tokenizer_resilient(path, json.loads((path / "config.json").read_text()))


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("text", ["Stable English.", "中文参考。", "😀🚀", "e\u0301", "文😀字", "👨\u200d👩\u200d👧"])
def test_real_tokenizer_unicode_edges_keep_exact_original_ids(served_tokenizer, thinking, text):
    tokenizer = served_tokenizer
    marked = [oa.ChatMessage(role="user", content=[part(text, True), part("\nQuestion.")])]
    plain = [oa.ChatMessage(role="user", content=text + "\nQuestion.")]
    observation = {}
    ids = oa._encode_messages(tokenizer, marked, enable_thinking=thinking, template_observability=observation)
    assert ids == oa._encode_messages(tokenizer, plain, enable_thinking=thinking)
    boundary = observation["explicit_cache_boundary"]
    prefix = tokenizer.decode(ids[:boundary])
    # The served tokenizer can normalize combining characters (e + accent -> é).
    normalized = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))
    assert prefix.endswith(normalized)
    assert tokenizer.decode(ids).startswith(prefix)
    assert tokenizer.decode(ids[boundary:]).startswith("\nQuestion.")


def test_real_tokenizer_rejects_a_newline_edge_inside_one_token(served_tokenizer):
    messages = [oa.ChatMessage(role="user", content=[part("Reference.\n", True), part("\nQuestion.")])]
    with pytest.raises(HTTPException, match="not an exact served token boundary"):
        oa._encode_messages(served_tokenizer, messages, enable_thinking=False)


def test_http_forwards_boundary_without_truncating_prompt(monkeypatch):
    from tests.test_server_openai import _fake_generation, _fake_streaming_session_state

    state = _fake_streaming_session_state()
    state.runtime.tokenizer = QwenToolHistoryBoundaryTokenizer()
    state.runtime.model = SimpleNamespace(args=SimpleNamespace(model_type="qwen3_5_moe_text"))
    captured = {}

    def generate(_state, prompt_ids, **kwargs):
        captured.update(kwargs)
        captured["prompt_ids"] = prompt_ids
        return _fake_generation("OK")

    monkeypatch.setattr(oa, "_run_generation", generate)
    monkeypatch.setattr(oa, "_store_retokenized_history_snapshot", lambda *a, **kw: {"stored": False, "reason": "test"})
    with TestClient(oa.create_app(state)) as client:
        response = client.post("/v1/chat/completions", json={
            "prompt_cache_key": "explicit-test-session",
            "messages": [{"role": "user", "content": [part("Stable reference.", True), part(" Variable question.")]}],
            "max_tokens": 64,
        })
    assert response.status_code == 200, response.text
    boundary = captured["request_observability"]["explicit_cache_boundary"]
    assert state.runtime.tokenizer.decode(captured["prompt_ids"][:boundary]).endswith("Stable reference.")
    assert "Variable question." in state.runtime.tokenizer.decode(captured["prompt_ids"][boundary:])
    assert captured["session_id"] == "explicit-test-session"


@pytest.mark.parametrize("model_type", [None, "llama", "qwen4_exp", "gemma4"])
def test_http_rejects_unsupported_model_before_generation(model_type):
    state = _fake_state()
    state.runtime.model = SimpleNamespace(args=SimpleNamespace(model_type=model_type))
    with TestClient(oa.create_app(state)) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": [part("reference", True), part(" tail")]}],
            "max_tokens": 64,
        })
    assert response.status_code == 400, response.text
    assert "serial Qwen3.5 text cache path" in response.text


@pytest.mark.parametrize(("path", "body"), [
    ("/v1/responses", {"input": [{"role": "user", "content": [{"type": "input_text", "text": "reference", "prompt_cache_breakpoint": {"mode": "explicit"}}]}]}),
    ("/v1/completions", {"prompt": "reference", "prompt_cache_breakpoint": {"mode": "explicit"}}),
])
def test_unsupported_endpoints_do_not_silently_drop_the_marker(path, body):
    with TestClient(oa.create_app(_fake_state())) as client:
        response = client.post(path, json=body)
    assert response.status_code == 400, response.text
    assert "prompt_cache_breakpoint" in response.text


def test_anthropic_adapter_does_not_silently_drop_the_marker():
    with pytest.raises(HTTPException, match="only by /v1/chat/completions"):
        oa._anthropic_content_to_chat_content([part("reference", True)])


def test_title_fast_path_rejects_explicit_cache_marker(monkeypatch):
    monkeypatch.setattr(oa, "_opencode_title_response", lambda *a, **kw: pytest.fail("marked request entered title fast path"))
    with TestClient(oa.create_app(_fake_state())) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": "You are a title generator. Generate a brief title. Never use tools."},
                {"role": "user", "content": [part("conversation reference", True)]},
            ],
            "max_tokens": 64,
        })
    assert response.status_code == 400, response.text
    assert "title fast path" in response.text
