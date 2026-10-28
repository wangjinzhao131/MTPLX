"""Interrupted immutable snapshots resume without pinning tensors or trusting missing blobs."""
import gc
import weakref

import mlx.core as mx
import pytest

from mtplx.cache_bank.codec import ColdEncodeInterrupted, TreeCodec
from mtplx.session_bank import SessionBank
from test_ssd_spill import FakeRuntime, LOOKUP_IDENTITY, TrimmableKV, make_cold


def entry(bank):
    return bank.put(runtime=FakeRuntime(), token_ids=list(range(1024)),
                    cache=[TrimmableKV(rows=1024)], logits=mx.array([[1., 2.]]),
                    hidden=mx.array([[[3., 4.]]]), session_id='resume',
                    template_hash='template-a', policy_fingerprint='policy-a')


@pytest.mark.parametrize('remove_blob', [False, True])
def test_repeated_yields_resume_and_restore_exactly(tmp_path, monkeypatch, remove_blob):
    cold = make_cold(tmp_path)
    bank = SessionBank()
    saved = entry(bank)
    busy = False
    writes = 0
    captures = 0
    write = cold._write_blob
    materialize = TreeCodec._materialize

    def capture(self, kind, value):
        nonlocal captures
        captures += 1
        return materialize(self, kind, value)

    def limited_write(digest, raw):
        nonlocal busy, writes
        result = write(digest, raw)
        writes += 1
        if writes % 2 == 0:
            busy = True
        return result

    monkeypatch.setattr(TreeCodec, '_materialize', capture)
    monkeypatch.setattr(cold, '_write_blob', limited_write)
    cold.foreground_busy = lambda: busy
    try:
        for attempt in range(10):
            busy = False
            try:
                done = cold.spill_entry(saved, raise_on_yield=True)
            except ColdEncodeInterrupted:
                assert saved._cold_spill_progress
                assert not cold._inflight_blob_hashes and not cold._reserved_bytes
                if attempt == 0 and remove_blob:
                    blob = next(iter(saved._cold_spill_progress.values()))[1]
                    cold._blob_path(blob['sha256']).unlink()
                continue
            assert done
            break
        else:
            pytest.fail('Every idle window restarted the same work')
        # Four KV blocks, logits and hidden; one extra capture if cleanup removed a blob.
        assert captures == 6 + int(remove_blob)
        assert not saved._cold_spill_progress
        assert cold.stats()['spill_resumed_blobs'] > 0
        busy = False
        restored = cold.lookup(list(range(1024)), **LOOKUP_IDENTITY)
        assert restored is not None
        assert mx.array_equal(restored.cache_snapshot.states[0], saved.cache_snapshot.states[0]).item()
        assert mx.array_equal(restored.logits, saved.logits).item()
        assert mx.array_equal(restored.hidden, saved.hidden).item()
    finally:
        cold.close()


def test_abandoned_progress_does_not_keep_snapshot_arrays_alive(tmp_path):
    cold = make_cold(tmp_path)
    bank = SessionBank()
    saved = entry(bank)
    cold.foreground_busy = lambda: bool(saved._cold_spill_progress)
    try:
        assert not cold.spill_entry(saved)
        progress = saved._cold_spill_progress
        refs = [row[0] for row in progress.values()]
        snapshot = weakref.ref(saved.cache_snapshot)
        cold.foreground_busy = lambda: False
        bank.clear()
        del saved
        gc.collect()
        assert snapshot() is None
        assert refs and all(ref() is None for ref in refs)
    finally:
        cold.close()
