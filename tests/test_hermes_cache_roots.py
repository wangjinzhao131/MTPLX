"""Changing the user question must retain the expensive system/tool prefix."""
from types import SimpleNamespace

import pytest

from mtplx import generation
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.checkpoint_anchors import AnchorPlan, CheckpointSink
from mtplx.server.openai import ChatMessage, _short_stateless_cache_request
from mtplx.session_bank import SessionBank
from test_checkpoint_anchors import _boundary, _put, _runtime


class ChatMLTokenizer:
    def encode(self, text, **kwargs):
        return [100] if text == '<|im_start|>' else [100, 101, 102]


def test_system_prefix_is_a_proven_token_boundary():
    rt = SimpleNamespace(tokenizer=ChatMLTokenizer())
    assert generation._system_prefix_boundary(rt, [9] * 22987 + [100, 101, 102, 5]) == 22987
    assert generation._system_prefix_boundary(rt, [100, 101, 102, 5]) is None
    assert generation._system_prefix_boundary(rt, [9] * 12) is None
    assert generation._system_prefix_boundary(SimpleNamespace(), [9]) is None


def test_root_and_transient_edges_shape_cold_and_warm_prefill():
    sink = generation._checkpoint_sink(
        [], session_bank=SessionBank(), prompt_len=24000,
        system_prefix_len=22987, stable_prefix_len=23900,
    )
    assert generation._checkpoint_edges(sink, 23900, 23999) == (22987, 23900)
    assert generation._checkpoint_edges(sink, 23900, 3000, 22000) == (987, 1900)
    assert generation._checkpoint_edges(sink, 23900, 1000, 23000) == (900,)
    sink.cuts_forwards = False
    assert generation._checkpoint_edges(sink, 23900, 23999) == ()


def test_system_checkpoint_survives_86k_growth_and_ssd_restart(tmp_path):
    root = 22987
    tier = SessionBankColdTier(base_dir=tmp_path / 'bank', mode='on', min_prefix_tokens=1)
    bank = SessionBank(cold_tier=tier)
    records = []
    try:
        # Simulate repeated turn publication, with the same eight-record budget.
        start = 64
        for end in (23040, 32768, 49152, 65536, 86278):
            sink = CheckpointSink(records, plan=AnchorPlan(
                prompt_end=end, system_prefix=root, restore_point=start - 64,
            ))
            for at in sorted(set(range(start, end, 64)) | {end - 1} | ({root} if start < root < end else set())):
                generation._append_gdn_boundary_record(sink, at, *_boundary(at)[1:])
                assert len(sink) <= 8
            entry = _put(bank, end, list(sink))
            records = generation._inherited_gdn_boundaries(entry, end)
            assert root in [row[0] for row in records]
            start = end
        assert entry.recurrent_boundary_at_or_below(22991)[0] == root
        assert tier.put_entry(entry, capabilities=['ar_insert'])
        assert tier.flush(timeout_s=5)
        bank.clear()
    finally:
        tier.close()
    tier = SessionBankColdTier(base_dir=tmp_path / 'bank', mode='on', min_prefix_tokens=1)
    bank = SessionBank(cold_tier=tier)
    try:
        rows = bank.near_prefix_candidates(
            list(range(22991)) + [999999] * 54,
            model_path=str(_runtime().model_path), mtp_enabled=False,
            hidden_variant=None, template_hash=None, mtp_history_policy='cycle',
            draft_head_identity=None, policy_fingerprint=None, allow_block_prefix=True,
        )
        assert rows and rows[0][0].recurrent_boundary_at_or_below(rows[0][1])[0] == root
    finally:
        tier.close()


@pytest.mark.parametrize('source', ['new', 'longest_prefix', 'common_prefix_reuse', 'pending_postcommit_near_prefix'])
def test_short_anonymous_approval_does_not_enter_cache(source):
    assert _short_stateless_cache_request(
        messages=[ChatMessage(role='system', content='Approve or deny.'), ChatMessage(role='user', content='Read file?')],
        prompt_tokens=492, tools_active=False, session_source=source,
    )


@pytest.mark.parametrize('tokens,tools,source,roles', [
    (512, False, 'new', ['system', 'user']),
    (400, True, 'new', ['system', 'user']),
    (400, False, 'header.x-session-id', ['system', 'user']),
    (400, False, 'metadata.session_id', ['system', 'user']),
    (400, False, 'user', ['system', 'user']),
    (400, False, 'new', ['system', 'user', 'assistant', 'user']),
])
def test_cache_remains_available_for_sessions_and_real_history(tokens, tools, source, roles):
    assert not _short_stateless_cache_request(
        messages=[ChatMessage(role=role, content='text') for role in roles],
        prompt_tokens=tokens, tools_active=tools, session_source=source,
    )


APPROVAL_SYSTEM = (
    'You are a security reviewer for an AI coding agent. '
    'You assess whether shell commands are safe to execute.\n'
    'Respond with exactly one word: APPROVE, DENY, or ESCALATE'
)


@pytest.mark.parametrize('tokens', [634, 752, 2048, 8192])
@pytest.mark.parametrize('policy', ['', '\nAdditional policy rules from the operator: read-only.'])
def test_hermes_approval_cache_bypass_has_no_command_length_cliff(tokens, policy):
    assert _short_stateless_cache_request(
        messages=[ChatMessage(role='system', content=APPROVAL_SYSTEM + policy),
                  ChatMessage(role='user', content='<command>echo hello</command>')],
        prompt_tokens=tokens, tools_active=False, session_source='new',
    )


@pytest.mark.parametrize('source,tools,roles', [
    ('header.x-session-id', False, ['system', 'user']),
    ('metadata.session_id', False, ['system', 'user']),
    ('new', True, ['system', 'user']),
    ('new', False, ['system', 'user', 'assistant', 'user']),
    ('new', False, ['user']),
])
def test_approval_text_does_not_disable_explicit_or_conversational_cache(source, tools, roles):
    assert not _short_stateless_cache_request(
        messages=[ChatMessage(role=role, content=APPROVAL_SYSTEM) for role in roles],
        prompt_tokens=752, tools_active=tools, session_source=source,
    )


@pytest.mark.parametrize('text', [APPROVAL_SYSTEM.split('\n')[0],
                               'Respond with exactly one word: APPROVE, DENY, or ESCALATE'])
def test_generic_review_prompt_keeps_cache(text):
    assert not _short_stateless_cache_request(
        messages=[ChatMessage(role='system', content=text), ChatMessage(role='user', content='review')],
        prompt_tokens=752, tools_active=False, session_source='new',
    )
