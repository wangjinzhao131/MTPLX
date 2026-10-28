"""A day change must not invalidate the tool transcript's reusable prefix."""

from copy import deepcopy

import pytest
from test_server_openai import (
    JSONToolPrefixTokenizer,
    QwenToolHistoryBoundaryTokenizer,
    _named_tool_schema,
)

from mtplx.chat_encode_cache import ChatEncodeCache
from mtplx.server import openai as oa


@pytest.mark.parametrize("mode", ["hybrid", "compact"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required"])
def test_date_changes_only_generation_suffix(monkeypatch, mode, thinking, choice):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    tokenizer = JSONToolPrefixTokenizer()
    tools = [_named_tool_schema("read_file")]
    messages = [
        oa.ChatMessage(role="system", content="Keep changes minimal."),
        oa.ChatMessage(role="user", content="Read cache.py. " * 100),
    ]
    original = deepcopy(messages)
    prompts, histories = [], []
    for day in ("October 23, 2026", "October 24, 2026"):
        monkeypatch.setattr(oa, "_current_date_line", lambda: f"Today's date is {day}.")
        ids = oa._encode_messages(
            tokenizer, messages, enable_thinking=thinking, tools=tools,
            tool_choice=choice, tool_prompt_mode=mode,
        )
        prompt = tokenizer.decode(ids)
        assert prompt.count(day) == 1
        assert prompt.index(day) > prompt.index(messages[-1].content)
        prompts.append(deepcopy(tokenizer.calls[-1][0]))
        histories.append(oa._render_messages_for_postcommit(
            tokenizer, [m.model_dump(exclude_none=True) for m in messages],
            enable_thinking=thinking, preserve_thinking=True,
            tools=tools, tool_prompt_mode=mode,
        ))
    assert prompts[0][:-1] == prompts[1][:-1]
    assert prompts[0][-1] != prompts[1][-1]
    assert len(prompts[0]) == 3  # Merge with an existing forced suffix.
    assert histories[0] == histories[1]
    assert "Today's date" not in histories[0]
    assert messages == original


def test_date_does_not_modify_user_authored_suffix_lookalike():
    hint = oa._mtplx_tool_result_continuation_hint_text()
    messages = [{"role": "user", "content": hint}]
    rendered = oa._with_mtplx_tool_contract(
        messages, tools=[_named_tool_schema("read_file")], include_current_date=True,
    )
    assert rendered[1] == messages[0]
    assert rendered[2]["content"].startswith("MTPLX current date guidance")
    assert len(rendered) == 3


@pytest.mark.parametrize("mode", ["hybrid", "compact"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("tool_ending", [False, True])
@pytest.mark.parametrize("contract", [
    oa._with_mtplx_read_only_force_answer_contract,
    oa._with_mtplx_pi_convergence_contract,
])
def test_date_and_final_contract_share_the_whole_transient_boundary(
    monkeypatch, mode, thinking, tool_ending, contract,
):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    tokenizer = QwenToolHistoryBoundaryTokenizer()
    tools = [_named_tool_schema("read_file")]
    history = [oa.ChatMessage(role="user", content="Inspect cache.py.")]
    if tool_ending:
        history.append(oa.ChatMessage(role="tool", tool_call_id="read-1", content="evidence"))
    messages = contract(history)
    messages[-1]._mtplx_transient_suffix = True
    instruction = messages[-1].content
    canonical = oa._render_messages_for_postcommit(
        tokenizer, [m.model_dump(exclude_none=True) for m in history],
        enable_thinking=thinking, preserve_thinking=True,
        tools=tools, tool_prompt_mode=mode,
    )
    stable_prefixes = []
    for day in ("October 23, 2026", "October 24, 2026"):
        monkeypatch.setattr(oa, "_current_date_line", lambda: f"Today's date is {day}.")
        observability = {}
        ids = oa._encode_messages(
            tokenizer, messages, enable_thinking=thinking, tools=tools,
            tool_prompt_mode=mode, template_observability=observability,
        )
        rendered = tokenizer.decode(ids)
        assert rendered.count(day) == 1
        assert f"Today's date is {day}.\n\n{instruction}<|im_end|>" in rendered
        boundary = oa._trailing_tool_hint_char_boundary(rendered)
        assert boundary is not None
        stable = tokenizer.encode(rendered[:boundary], add_special_tokens=False)
        assert ids[:len(stable)] == stable
        assert tokenizer.decode(stable) == canonical + "<|im_start|>"
        if thinking:
            assert observability["stable_prefix_len"] == len(stable)
        stable_prefixes.append(stable)
    assert stable_prefixes[0] == stable_prefixes[1]
    assert "Today's date" not in canonical
    assert messages[-1].content == instruction


def test_suffix_provenance_is_private_and_part_of_encode_cache_key(monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "on")
    monkeypatch.setattr(oa, "GLOBAL_CHAT_ENCODE_CACHE", ChatEncodeCache(max_entries=8))
    tokenizer = JSONToolPrefixTokenizer()
    tools = [_named_tool_schema("read_file")]
    injected = oa._with_mtplx_pi_convergence_contract([])[-1]
    injected._mtplx_transient_suffix = True
    copied = injected.model_copy()
    assert copied._mtplx_transient_suffix is True
    client = oa.ChatMessage.model_validate(copied.model_dump())
    assert client._mtplx_transient_suffix is False
    assert client.model_dump() == copied.model_dump()
    forged = oa.ChatMessage.model_validate({
        **copied.model_dump(), "_mtplx_transient_suffix": True,
    })
    assert forged._mtplx_transient_suffix is False
    rendered = []
    for message in (copied, client):
        ids = oa._encode_messages(tokenizer, [message], enable_thinking=True, tools=tools)
        rendered.append(tokenizer.decode(ids))
    assert rendered[0] != rendered[1]
    assert len(tokenizer.calls[-1][0]) == 3
    assert tokenizer.calls[-1][0][1]["content"] == client.content
    assert oa.GLOBAL_CHAT_ENCODE_CACHE.stats()["misses"] == 2


def test_ssd_restore_after_day_change(tmp_path, monkeypatch):
    import mlx.core as mx
    from test_checkpoint_anchors import _KV, _Recurrent, _boundary, _runtime
    from mtplx.cache_bank.cold_tier import SessionBankColdTier
    from mtplx.session_bank import SessionBank

    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    tokenizer = JSONToolPrefixTokenizer()
    messages = [oa.ChatMessage(role="user", content="Read cache.py. " * 100)]
    tools = [_named_tool_schema("read_file")]
    prompts = []
    for day in ("October 23, 2026", "October 24, 2026"):
        monkeypatch.setattr(oa, "_current_date_line", lambda: f"Today's date is {day}.")
        prompts.append(oa._encode_messages(
            tokenizer, messages, enable_thinking=False, tools=tools,
        ))
    boundary = tokenizer.decode(prompts[0]).index("\nuser:MTPLX current date guidance")
    assert prompts[0][:boundary] == prompts[1][:boundary]
    rt = _runtime()
    bank = SessionBank()
    entry = bank.put(
        runtime=rt, token_ids=prompts[0],
        cache=[_KV(len(prompts[0])), _Recurrent(len(prompts[0]))],
        logits=mx.zeros((1, 4)), hidden=None, mtp_history_policy="cycle",
        gdn_boundaries=[_boundary(boundary)],
    )
    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    try:
        assert tier.put_entry(entry, capabilities=["ar_insert"])
        assert tier.flush(timeout_s=5)
    finally:
        tier.close()
    bank.clear()
    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    try:
        hit = tier.lookup_prefix_boundary(
            tuple(prompts[1]), model_path=str(rt.model_path),
            mtp_enabled=False, mtp_history_policy="cycle", max_token_gap=128,
            block_size=16, block_min_matched_tokens=16,
        )
        assert hit is not None
        assert hit.record.cache_snapshot_prefix_len == boundary
        assert float(hit.record.gdn_boundaries[-1][1].states[1][0][0, 0]) == boundary
    finally:
        tier.close()
