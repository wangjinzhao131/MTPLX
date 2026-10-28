"""CPU-only checks of real server deferral functions without loading MLX."""
import ast
import copy
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch


FUNCTIONS = {
    '_tool_postcommit_experiment_enabled', '_tool_postcommit_enabled', '_tool_postcommit_owner_call', '_hold_tool_postcommit', '_pause_tool_postcommit',
    '_release_tool_postcommit', '_arm_tool_postcommit', '_defer_tool_postcommit',
    '_tool_postcommit_failed', '_launch_tool_postcommit', '_tool_postcommit_work_snapshot',
    '_schedule_idle_postcommit_snapshot', '_generation_final_postcommit_compatibility',
    '_run_generation_dispatched',
}


class Timer:
    def __init__(self, delay, callback, args):
        self.delay, self.callback, self.args = delay, callback, args
        self.cancelled = self.started = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.callback(*self.args)


class Message:
    def __init__(self, content):
        self.content = content

    def model_copy(self, deep=False):
        return copy.deepcopy(self) if deep else copy.copy(self)


def server_functions():
    path = Path(__file__).parents[1] / 'mtplx/server/openai.py'
    source = ast.parse(path.read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    scope = dict(Timer=Timer, time=time, json=json, uuid=uuid, os=os, LOGGER=logging.getLogger(__name__),
                 _TOOL_POSTCOMMIT_LOCK=threading.Lock(), _TOOL_POSTCOMMIT_IDLE_S=15.0,
                 _STATS_FOOTER_RE=re.compile('impossible-footer'), STATS_FOOTER_MARKER='stats-footer')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope


class DeferredToolPostcommitTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'MTPLX_TOOL_POSTCOMMIT_DEBOUNCE': '1'})
        env.start()
        self.addCleanup(env.stop)
        self.fn = server_functions()
        self.events = []
        self.session = SimpleNamespace(revision=1)
        self.state = SimpleNamespace(
            runtime=SimpleNamespace(local_optimization='qwen35-mtp'),
            model_scheduler=SimpleNamespace(is_owner_thread=lambda: True, stats=lambda: {}),
            sessions=SimpleNamespace(
                bank=SimpleNamespace(
                    hold_session_persistence=lambda sid: self.events.append(('hold', sid)),
                    release_session_persistence=lambda sid: self.events.append(('release', sid))),
                peek=lambda sid: self.session),
        )

    def call(self, name, *args, **kwargs):
        return self.fn[name](self.state, *args, **kwargs)

    def hold(self, request='r1'):
        self.call('_hold_tool_postcommit', 's1', request)

    def defer(self, revision=1, request='r1'):
        return self.call('_defer_tool_postcommit', session_id='s1', request_id=request,
                         payload={'expected_session_revision': revision, 'assistant_content': 'CPU history'})

    def test_idle_timer_has_no_worker_until_due_and_keeps_cpu_payload(self):
        self.hold()
        outcome = self.defer()
        record = self.state._tool_postcommits['s1']
        timer = record['timer']
        self.assertEqual(outcome['idle_s'], 15.0)
        self.assertEqual(self.events, [('hold', 's1')])
        self.assertEqual(timer.args, (self.state, 's1', record['token']))
        scheduled = []
        self.fn['_schedule_idle_postcommit_snapshot'] = lambda state, **kwargs: scheduled.append(kwargs)
        timer.fire()
        self.assertEqual(len(scheduled), 1)
        self.assertFalse(scheduled[0]['_defer_tools'])
        self.assertNotIn('final_state', scheduled[0])
        self.assertEqual(scheduled[0]['expected_session_revision'], 1)

    def test_successor_suspends_then_replaces_intent_and_old_timer_is_stale(self):
        self.hold(); self.defer()
        old = self.state._tool_postcommits['s1']['timer']
        self.hold('r2')
        self.assertTrue(old.cancelled)
        old.fire()
        self.assertEqual(len(self.events), 2)
        self.session.revision = 2
        self.defer(2, 'r2')
        record = self.state._tool_postcommits['s1']
        self.assertEqual(record['intent']['expected_session_revision'], 2)
        self.assertFalse(record['active'])
        self.assertEqual(self.defer(1, 'r1')['mode'], 'abandoned_stale')
        self.assertEqual(record['intent']['expected_session_revision'], 2)

    def test_failed_successor_rearms_previous_intent(self):
        self.hold(); self.defer()
        self.hold('r2')
        self.call('_tool_postcommit_failed', 's1', 'r2')
        record = self.state._tool_postcommits['s1']
        self.assertTrue(record['timer'].started)
        self.assertEqual(record['intent']['expected_session_revision'], 1)
        self.assertFalse(record['active'])
        self.assertNotIn(('release', 's1'), self.events)

    def test_stop_flushes_current_hold_and_cancelled_timer_cannot_run(self):
        self.hold(); self.defer()
        timer = self.state._tool_postcommits['s1']['timer']
        self.hold('stop')
        self.call('_release_tool_postcommit', 's1', request_id='stop')
        timer.fire()
        self.assertEqual(self.events[-1], ('release', 's1'))
        self.assertEqual(self.state._tool_postcommits, {})

    def test_failure_without_prior_intent_and_stale_revision_release(self):
        self.hold(); self.call('_tool_postcommit_failed', 's1', 'r1')
        self.assertEqual(self.state._tool_postcommits, {})
        self.hold(); self.defer()
        timer = self.state._tool_postcommits['s1']['timer']
        self.session.revision = 7
        timer.fire()
        self.assertEqual(self.state._tool_postcommits, {})

    def test_arrival_pauses_without_owner_work_and_failure_rearms(self):
        self.hold(); self.defer()
        timer = self.state._tool_postcommits['s1']['timer']
        self.assertTrue(self.call('_pause_tool_postcommit', 's1', 'r2'))
        self.assertTrue(timer.cancelled)
        self.assertEqual(self.events, [('hold', 's1')])
        self.assertFalse(self.call('_pause_tool_postcommit', 's1', 'overlap'))
        self.call('_tool_postcommit_failed', 's1', 'r2')
        self.assertTrue(self.state._tool_postcommits['s1']['timer'].started)

    def test_timer_identity_cannot_collide_after_release_and_recreate(self):
        self.hold(); self.defer()
        old = self.state._tool_postcommits['s1']['timer']
        self.call('_release_tool_postcommit', 's1', request_id='r1')
        self.hold('r2'); self.defer(request='r2')
        scheduled = []
        self.fn['_schedule_idle_postcommit_snapshot'] = lambda state, **kwargs: scheduled.append(kwargs)
        old.fire()
        self.assertEqual(scheduled, [])
        self.state._tool_postcommits['s1']['timer'].fire()
        self.assertEqual(len(scheduled), 1)

    def test_real_scheduler_defers_before_fan_or_model_submission(self):
        self.hold()
        original = [Message('system'), Message('user')]
        outcome = self.call('_schedule_idle_postcommit_snapshot', session_id='s1',
            messages=original, assistant_content='', assistant_tool_calls=[{'id': 'call'}],
            thinking_enabled=False, policy_fingerprint='p', unsafe_reason='rewrite',
            expected_session_revision=1, request_id='r1')
        self.assertEqual(outcome['mode'], 'idle_deferred')
        payload = self.state._tool_postcommits['s1']['intent']
        original[0].content = 'changed'
        self.assertEqual(payload['messages'][0].content, 'system')
        self.assertNotIn('session', payload)
        self.assertNotIn('generated', payload)
        self.assertEqual(self.events, [('hold', 's1')])

    def test_tool_ram_key_validates_served_tokens_without_history_render(self):
        self.fn['_history_ids_for_postcommit'] = lambda *a, **k: self.fail('history rendered on immediate tool path')
        generated = {'tokens': [3, 4], '_final_state': SimpleNamespace(safe_to_commit=True, generated_token_ids=(3, 4))}
        args = dict(prompt_ids=[1, 2], generated=generated, messages=[Message('system')],
                    assistant_content='', assistant_tool_calls=[{'id': 'call'}], thinking_enabled=False,
                    defer_tool_history=True)
        result = self.call('_generation_final_postcommit_compatibility', **args)
        self.assertEqual(result['token_ids'], [1, 2, 3, 4])
        self.assertEqual(result['mode'], 'generation_final_ram')
        generated['_final_state'].generated_token_ids = (3,)
        self.assertEqual(self.call('_generation_final_postcommit_compatibility', **args)['reason'], 'generated_token_mismatch')

    def test_cancel_release_never_waits_for_busy_owner_and_rechecks_identity(self):
        self.hold(); self.defer()
        token = self.state._tool_postcommits['s1']['token']
        self.state.model_scheduler.is_owner_thread = lambda: False
        queued = []
        def submit(state, fn, **kwargs):
            queued.append(fn)
            return SimpleNamespace(result=lambda: self.fail('cancel release blocked for owner'))
        self.fn['_submit_foreground_model_work'] = submit
        self.call('_release_tool_postcommit', 's1', token='stale', wait=False)
        self.assertEqual(queued, [])
        self.call('_release_tool_postcommit', 's1', token=token, wait=False)
        self.assertEqual(len(queued), 1)
        self.call('_pause_tool_postcommit', 's1', 'next')
        queued[0]()
        self.assertIn('s1', self.state._tool_postcommits)

    def test_bank_hold_failure_cleans_the_active_record(self):
        def fail(sid):
            self.events.append(('hold', sid))
            raise RuntimeError('hold queue cancellation failed')
        self.state.sessions.bank.hold_session_persistence = fail
        with self.assertRaisesRegex(RuntimeError, 'hold queue'):
            self.call('_run_generation_dispatched', [1], batch_key='ar', response_id='r1', session_id='s1')
        self.assertEqual(self.state._tool_postcommits, {})
        self.assertEqual(self.events[-1], ('release', 's1'))

    def test_default_off_does_not_hold_or_add_receipts(self):
        os.environ.pop('MTPLX_TOOL_POSTCOMMIT_DEBOUNCE')
        self.fn['_run_generation_dispatched_impl'] = lambda *args, **kwargs: {'stats': {}, 'tokens': [1]}
        result = self.call('_run_generation_dispatched', [1], batch_key='mtp', response_id='r1', session_id='s1')
        self.assertEqual(self.events, [])
        self.assertNotIn('tool_postcommit_work', result['stats'])
        self.assertFalse(self.call('_tool_postcommit_enabled'))
        with patch.dict(os.environ, {'MTPLX_IDLE_POSTCOMMIT_TOOL_REWRITE': '0'}):
            result = self.call('_schedule_idle_postcommit_snapshot',
                session_id='s1', messages=[object()], assistant_content='',
                assistant_tool_calls=[{'id': 'c'}], thinking_enabled=False,
                policy_fingerprint='p', unsafe_reason='rewrite')
        self.assertEqual(result['mode'], 'disabled')  # No new history copy while off.

    def test_dispatch_holds_before_each_lane_and_reports_counters(self):
        for lane in ('mtp', 'ar', 'mtp_batch'):
            def run(state, prompt_ids, **kwargs):
                self.assertEqual(self.events[-1], ('hold', 's1'))
                return {'stats': {}, 'tokens': [1]}
            self.fn['_run_generation_dispatched_impl'] = run
            result = self.call('_run_generation_dispatched', [1], batch_key=lane,
                               response_id='r1', session_id='s1')
            receipt = result['stats']['tool_postcommit_work']
            self.assertLessEqual(receipt['entry']['monotonic_s'], receipt['exit']['monotonic_s'])
            self.assertEqual(receipt['exit']['ssd_pending_writes'], 0)
            self.call('_tool_postcommit_failed', 's1', 'r1')


if __name__ == '__main__':
    unittest.main()
