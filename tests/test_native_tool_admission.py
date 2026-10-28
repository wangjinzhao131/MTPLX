"""Native ownership can avoid COW, but never a required checkpoint or KV growth."""
from types import SimpleNamespace
import pytest

import mtplx.session_bank as bank
from mtplx.server import openai as srv
from test_memguard_admission import _served_profile  # noqa: F401


@pytest.fixture(autouse=True)
def experimental_native_tools(monkeypatch):
    monkeypatch.setenv('MTPLX_TOOL_POSTCOMMIT_DEBOUNCE', '1')


def test_native_bill_requires_proof_and_the_requested_recurrent_checkpoint(monkeypatch):
    monkeypatch.setattr(bank, '_native_lease_cache', lambda cache: True)
    entry = SimpleNamespace(cache_ref=[object()], live_ref_only=True, has_recurrent=True,
                            prefix_len=200, has_recurrent_boundary=lambda point: point == 128)
    bill = srv._admission_restore_copies_prefix
    assert bill(entry, 'reference', 'same')
    assert bill(entry, 'clone', 'same', native_ownership_proven=True)
    assert not bill(entry, 'reference', 'same', native_ownership_proven=True)
    assert not bill(entry, 'reference', 'same', native_ownership_proven=True, required_boundary=128)
    assert bill(entry, 'reference', 'same', native_ownership_proven=True, required_boundary=64)
    assert not bill(entry, 'reference', 'same', native_ownership_proven=True, required_boundary=200)


def test_native_shape_reads_allocated_capacity_and_draft_bytes(monkeypatch):
    monkeypatch.setattr(bank, '_native_lease_cache', lambda cache: True)
    def layer(width):
        leaf = SimpleNamespace(shape=(1, 1, 256, width), nbytes=256 * width)
        return SimpleNamespace(keys=leaf, values=leaf, step=256)
    entry = SimpleNamespace(cache_ref=[layer(16)], mtp_history_cache_ref=[layer(8)])
    shape = srv._lease_cache_shape(entry)
    assert shape == {'paged': False, 'native_capacity_tokens': 256,
                     'native_step': 256, 'native_bytes_per_token': 48}


def test_owned_native_kv_growth_is_billed_separately_from_restore_copy():
    geometry = srv._AdmissionGeometry(live_bytes_per_token=64, paged_bytes_per_token=64,
        context_transient_bytes_per_token=0, flat_transient_bytes=0, weights_bytes=1)
    def growth(prompt):
        return srv._admission_growth(geometry, prompt_tokens=prompt, reused_tokens=200,
            restore_copies_prefix=False, layout='contiguous_dense_decode', source_layout=None,
            output_tokens=0, publish=False, scratch_bytes=0, prefill_chunk_tokens=128,
            lease={'paged': False, 'native_capacity_tokens': 256, 'native_step': 256,
                   'native_bytes_per_token': 64})
    fits = growth(220)
    assert fits['restore_copy_bytes'] == fits['lease_grow_transient_bytes'] == 0
    assert fits['live_prefill_bytes'] == 0
    grows = growth(400)
    assert grows['restore_copy_bytes'] == 0
    assert grows['lease_grow_transient_bytes'] == 256 * 64
    assert grows['live_prefill_bytes'] == 200 * 64  # KVCache: 200 used + 256 new rows.
    assert grows['growth_bytes'] >= (256 + 200) * 64


def test_tool_ram_put_preserves_raw_ids_and_trusted_stable_metadata(monkeypatch):
    from tests.test_openai_bridge import _final_state, _postcommit_state
    state = _postcommit_state()
    state.runtime.local_optimization = 'qwen35-mtp'
    state.sessions.bank.hold_session_persistence = lambda session_id: None
    state.sessions.bank.release_session_persistence = lambda session_id: None
    monkeypatch.setattr(srv, '_history_ids_for_postcommit', lambda *a, **k: pytest.fail('retokenized tool history'))
    generated = {'tokens': [7, 8], '_final_state': _final_state([7, 8]),
                 'stats': {'stable_prefix_len': 2, 'cache_source': 'none', 'session_cache_hit': False}}
    result = srv._store_generation_final_history_snapshot(
        state, session_id='s', prompt_ids=[1, 2, 3, 4], generated=generated,
        messages=[srv.ChatMessage(role='user', content='check')], assistant_content='',
        assistant_tool_calls=[{'id': 'c', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}],
        thinking_enabled=False, policy_fingerprint='policy', keep_live_ref=True,
    )
    assert result['stored'] and result['mode'] == 'generation_final_ram'
    publication = state.sessions.bank.puts[0]
    assert publication['token_ids'] == [1, 2, 3, 4, 7, 8]
    assert publication['native_stable_prefix_len'] == 2
    assert publication['native_owned'] is True


@pytest.mark.parametrize('proven, measured, auxiliary, cap, expected', [
    (True, 1000, 1500, None, 2500),
    (True, 2000, 1500, 1, 4000),  # A live lease is still published above the snapshot cap.
    (True, None, 1500, None, None),
    (True, 1000, None, None, None),
    (False, 1000, 1500, None, None),
])
def test_native_publication_bill_requires_proof_and_actual_anchor_bytes(
    monkeypatch, proven, measured, auxiliary, cap, expected,
):
    from test_memguard_admission import _flash_next_state, _install, _Machine, _manager
    manager = _manager()
    state = _flash_next_state(manager)
    state.runtime.local_optimization = 'qwen35-mtp'
    state.runtime.one_copy_anchor_nbytes = measured
    state.lock = SimpleNamespace(locked=lambda: True)
    state.model_scheduler = SimpleNamespace(is_owner_thread=lambda: True)
    entry = SimpleNamespace(cache_ref=[object()], live_ref_only=True, session_id='s',
                            prefix_len=10000, has_recurrent=False, lease_aux_nbytes=auxiliary)
    monkeypatch.setattr(manager.bank, 'native_lease_can_extend', lambda *args, **kwargs: proven)
    monkeypatch.setattr(manager.bank, 'restore_plan', lambda *args, **kwargs: {
        'reuse_tokens': 10000, 'mode': 'exact', 'source': entry, 'keys': (),
    })
    monkeypatch.setattr(bank, '_native_lease_cache', lambda cache: True)
    monkeypatch.setattr(srv, '_lease_cache_shape', lambda entry: {
        'paged': False, 'native_capacity_tokens': 16384, 'native_step': 256,
        'native_bytes_per_token': 64,
    })
    if cap is not None:
        manager.bank.per_session_max_bytes = cap
    _install(monkeypatch, _Machine(manager.bank, base_gib=40, cache_gib=0, host_gib=0))
    pricing = {}
    receipt = srv._prefill_admission_shed(
        state, prompt_ids=list(range(11000)), session_bank=manager.bank, session_id='s',
        restore_mode='reference', commit_prompt_prefix=True, max_new_tokens=100,
        pricing=pricing,
    )
    assert receipt is None
    billed = pricing['growth']['publish_copy_bytes']
    if expected is not None:
        assert billed == expected
    else:
        assert billed > 11000 * 64
