"""The tool response boundary includes RAM publication, but no idle rewrite."""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from mtplx.server import openai
from test_server_openai import _fake_streaming_session_state, _tool_schema
from test_stream_response_tail import _Engine, _final_stats
from test_tool_postcommit_debounce import Timer


@pytest.mark.parametrize('enabled, parsed_call, finish_reason, stored', [
    (True, True, 'stop', True),
    (True, True, 'stop', False),
    (True, False, 'tool_calls', True),
    (False, True, 'stop', True),
    (True, False, 'stop', True),
])
def test_tool_terminal_waits_for_ram_and_receipt_includes_its_work(
    monkeypatch, enabled, parsed_call, finish_reason, stored,
):
    monkeypatch.setenv('MTPLX_TOOL_POSTCOMMIT_DEBOUNCE', '1' if enabled else '0')
    monkeypatch.setenv('MTPLX_STREAM_TERMINAL_FRAME_BEFORE_COMMIT', '1')
    monkeypatch.setattr(openai, 'Timer', Timer)
    state = _fake_streaming_session_state()
    state.runtime.local_optimization = 'qwen35-mtp'
    state.sessions.bank.hold_session_persistence = lambda sid: None
    state.sessions.bank.release_session_persistence = lambda sid: None
    engine = _Engine(stored=stored, hold='store')

    def generate(*args, **kwargs):
        result = engine.run_generation(*args, **kwargs)
        if parsed_call:
            result['text'] = '<tool_call>\n<function=session_status>\n</function>\n</tool_call>'
        result['finish_reason'] = finish_reason
        return result

    def work_snapshot(_state):
        return {
            'monotonic_s': time.monotonic(),
            'started_by_batch_key': {'postcommit.stream.final': int(engine.held.is_set())},
            'completed_by_kind': {}, 'ssd_writes_enqueued': 0, 'ssd_pending_writes': 0,
        }

    monkeypatch.setattr(openai, '_run_generation', generate)
    monkeypatch.setattr(openai, '_store_generation_final_history_snapshot', engine.store_generation_final)
    monkeypatch.setattr(openai, '_tool_postcommit_work_snapshot', work_snapshot)
    completed = threading.Event()
    responses = []
    errors = []
    should_wait = enabled and (parsed_call or finish_reason == 'tool_calls')

    with TestClient(openai.create_app(state)) as client:
        def request():
            try:
                responses.append(client.post('/v1/chat/completions',
                    headers={'x-mtplx-session-id': 'tool-boundary'}, json={
                        'messages': [{'role': 'user', 'content': 'Check status. ' * 80}],
                        'tools': [_tool_schema()], 'tool_choice': 'auto',
                        'stream': True, 'enable_thinking': False, 'max_tokens': 128,
                    }))
            except BaseException as exc:
                errors.append(exc)
            finally:
                completed.set()

        reader = threading.Thread(target=request)
        reader.start()
        try:
            assert engine.held.wait(5), errors
            assert completed.wait(0.1 if should_wait else 5) is not should_wait
        finally:
            engine.gate.set()
            reader.join(5)
        assert completed.is_set() and not errors, errors
        session = state.sessions.peek('tool-boundary')
        session.wait_for_response_tail(5)

    response = responses[0]
    assert response.status_code == 200
    assert response.text.rstrip().endswith('data: [DONE]')
    stats = _final_stats(response.text)
    if should_wait:
        receipt = stats['tool_postcommit_work']
        assert receipt['entry']['started_by_batch_key']['postcommit.stream.final'] == 0
        assert receipt['exit']['started_by_batch_key']['postcommit.stream.final'] == 1
        assert receipt['exit']['monotonic_s'] >= receipt['entry']['monotonic_s']
        if parsed_call:
            assert stats['session_postcommit_snapshot']['mode'] == 'idle_deferred'
            assert stats['session_postcommit_snapshot']['idle_s'] == 15
    else:
        assert stats['session_postcommit_snapshot']['mode'] == 'after_response'
