"""Small CPU fixtures prove hard closing goes through sampling and cache repair."""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.cache import KVCache
import pytest

from mtplx.generation import generate_ar, generate_mtpk
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from mtplx.thinking_budget import HardThinkingBudget, HardThinkingBudgetConfig


BODY, CLOSE, NEWLINE_1, NEWLINE_2, OPEN, ANSWER = range(1, 7)
CLOSING = (CLOSE, NEWLINE_1, NEWLINE_2)


def _config(**overrides):
    values = dict(
        enabled=True,
        budget_tokens=96,
        think_open_token=OPEN,
        think_close_token=CLOSE,
        forced_close_ids=CLOSING,
        starts_in_think=True,
        reentry_ban_tokens=0,
    )
    values.update(overrides)
    return HardThinkingBudgetConfig(**values)


def test_speculative_preview_crosses_budget_without_mutating_committed_state():
    guard = HardThinkingBudget(_config())
    guard.observe([BODY] * 94)
    assert guard.forced_token_for([BODY] * 95) is None
    assert guard.forced_token_for([BODY] * 96) == CLOSE
    assert guard.forced_token_for([BODY] * 96 + [CLOSE]) == NEWLINE_1
    assert guard.forced_token_for([BODY] * 96 + [CLOSE, NEWLINE_1]) == NEWLINE_2
    assert guard.summary()["think_tokens"] == 94
    assert guard.engaged is None
    guard.observe([BODY] * 96 + list(CLOSING))
    assert guard.engaged_at == 96
    assert guard.summary()["think_tokens"] == 96
    assert guard.summary()["forced_emitted"] == len(CLOSING)
    assert guard.forced_token_for([BODY] * 96 + list(CLOSING)) is None


def test_budget_remains_cumulative_after_reopening():
    guard = HardThinkingBudget(_config(budget_tokens=2))
    tokens = [BODY, BODY, *CLOSING, ANSWER, OPEN]
    guard.observe(tokens)
    assert guard.forced_token_for(tokens) == CLOSE
    assert guard.summary()["think_tokens"] == 2


class _Tokenizer:
    def decode(self, tokens, **kwargs):
        text = {CLOSE: "</think>", NEWLINE_1: "\n", NEWLINE_2: "\n"}
        return "".join(text.get(int(token), "x") for token in tokens)


class _Model:
    def __init__(self, *, natural_close=False, closing_draft=False, opens=False):
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])
        self.natural_close = natural_close
        self.closing_draft = closing_draft
        self.opens = opens

    def make_cache(self):
        return [KVCache()]

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **kwargs):
        return hidden_states

    def _logits(self, tokens, *, draft=False):
        rows = []
        for token in tokens:
            next_token = BODY
            if self.opens and token == 0:
                next_token = OPEN
            elif self.natural_close:
                next_token = {BODY: CLOSE, CLOSE: ANSWER, ANSWER: ANSWER}.get(token, BODY)
            elif draft and self.closing_draft:
                next_token = {BODY: CLOSE, CLOSE: NEWLINE_1, NEWLINE_1: NEWLINE_2}.get(token, BODY)
            row = [0.0] * 8
            row[next_token] = 20_000.0  # Exceeds the old guard's soft boost.
            rows.append(row)
        return mx.array([rows])

    def __call__(self, input_ids, *, cache=None, return_hidden=False, emit_logits=True, logits_keep=None, **kwargs):
        tokens = input_ids.reshape(-1).tolist()
        if cache:
            values = input_ids[:, None, :, None].astype(mx.float32)
            cache[0].update_and_fetch(values, values)
        keep = len(tokens) if logits_keep is None else min(len(tokens), max(1, logits_keep))
        logits = self._logits(tokens[-keep:]) if emit_logits else None
        hidden = mx.zeros((1, len(tokens), 2))
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden_states, next_token_ids, *, return_hidden=False, **kwargs):
        logits = self._logits(next_token_ids.reshape(-1).tolist(), draft=True)
        hidden = mx.zeros((1, int(next_token_ids.shape[1]), 2))
        return (logits, hidden) if return_hidden else logits


def _generate(mode, *, config, sampled=False, stream=False, model=None, max_tokens=101):
    runtime = MTPLXRuntime(
        model=model or _Model(), tokenizer=_Tokenizer(), model_path=Path("tiny-thinking"),
        mtp_enabled=True, contract=MTPContract(),
    )
    streamed = []
    kwargs = dict(
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.7 if sampled else 0.0, top_k=4, top_p=0.95),
        seed=7, stop_token_ids=set(), thinking_guard=config,
        capture_final_state=True, token_callback=streamed.extend if stream else None,
    )
    if mode == "ar":
        result = generate_ar(runtime, [0], **kwargs)
    else:
        result = generate_mtpk(runtime, [0], speculative_depth=3,
                               verify_strategy=mode, mtp_history_policy="committed", **kwargs)
    assert result.final_state is not None and result.final_state.safe_to_commit
    assert result.final_state.generated_token_ids == tuple(result.tokens)
    cache = result.final_state.final_trunk_cache[0]
    # Every accepted close/newline occupies the same cache position as its output.
    assert cache.keys[..., :cache.offset, :].reshape(-1).tolist() == [0, *result.tokens]
    if stream:
        assert streamed == result.tokens
    return result


@pytest.mark.parametrize("mode", ["ar", "batched", "target_prefix"])
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_exact_96_token_cap_closing_sequence_and_cache(mode, sampled, stream):
    result = _generate(mode, config=_config(), sampled=sampled, stream=stream)
    assert result.tokens == [BODY] * 96 + list(CLOSING) + [BODY] * 2
    assert "</think>\n\n" in result.text
    assert result.stats.thinking_guard["think_tokens"] == 96
    assert result.stats.thinking_guard["forced_emitted"] == len(CLOSING)


@pytest.mark.parametrize("mode", ["ar", "batched", "target_prefix"])
def test_natural_close_and_no_budget_preserve_model_tokens(mode):
    early = _generate(mode, config=_config(), model=_Model(natural_close=True), max_tokens=8)
    assert early.tokens == [BODY, CLOSE] + [ANSWER] * 6
    assert early.stats.thinking_guard["engaged"] is None
    unlimited = _generate(mode, config=None, max_tokens=8)
    assert unlimited.tokens == [BODY] * 8
    assert unlimited.stats.thinking_guard == {}
    generated_open = _generate(mode, config=_config(budget_tokens=2, starts_in_think=False),
                               model=_Model(opens=True), max_tokens=7)
    assert generated_open.tokens == [OPEN, BODY, BODY, *CLOSING, BODY]


@pytest.mark.parametrize("mode", ["batched", "target_prefix"])
@pytest.mark.parametrize("sampled", [False, True])
def test_correct_closing_drafts_are_accepted_in_same_round(mode, sampled):
    result = _generate(mode, config=_config(budget_tokens=3), sampled=sampled,
                       model=_Model(closing_draft=True), max_tokens=8)
    assert result.tokens == [BODY] * 3 + list(CLOSING) + [BODY] * 2
    assert any(event.get("accepted_depths", 0) >= 3 for event in result.stats.events)
