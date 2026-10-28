"""A client closing the initial role frame must not strand a paused tool intent."""
import asyncio

from starlette.requests import Request

from mtplx.server import openai
from test_server_openai import _fake_streaming_session_state
from test_tool_postcommit_debounce import Timer


def test_initial_role_frame_close_rearms_prior_tool_intent(monkeypatch):
    state = _fake_streaming_session_state()
    monkeypatch.setenv('MTPLX_TOOL_POSTCOMMIT_DEBOUNCE', '1')
    monkeypatch.setattr(openai, 'Timer', Timer)
    previous = Timer(15, lambda: None, ())
    state._tool_postcommits = {'early-close': {
        'active': False, 'request_id': 'previous', 'token': 'previous',
        'intent': {'expected_session_revision': 0}, 'timer': previous,
    }}
    app = openai.create_app(state)
    endpoint = next(route.endpoint for route in app.routes if getattr(route, 'path', '') == '/v1/chat/completions')
    raw_request = Request({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions',
        'headers': [(b'x-mtplx-session-id', b'early-close')], 'app': app})
    request = openai.ChatCompletionRequest(messages=[openai.ChatMessage(role='user', content='Say OK')],
        enable_thinking=False, stream=True, max_tokens=4)

    async def close_after_role():
        response = await endpoint(raw_request, request)
        assert state._tool_postcommits['early-close']['active']
        chunk = await anext(response.body_iterator)
        assert '"role": "assistant"' in chunk
        await response.body_iterator.aclose()

    try:
        asyncio.run(close_after_role())
        record = state._tool_postcommits['early-close']
        assert not record['active']
        assert previous.cancelled
        assert record['timer'].started
        assert record['intent'] == {'expected_session_revision': 0}
    finally:
        state.generation_executor.shutdown(wait=True)
