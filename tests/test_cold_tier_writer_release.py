"""Completed SSD writes must release their payload while the writer is idle."""

import weakref

import pytest

from mtplx.cache_bank.cold_tier import PendingWrite, SessionBankColdTier


@pytest.mark.parametrize("outcome", [True, False, "error"])
def test_idle_writer_releases_finished_payload(tmp_path, monkeypatch, outcome):
    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on")

    def write(pending):
        if outcome == "error":
            raise OSError("test write failure")
        return outcome

    monkeypatch.setattr(tier, "_write_pending", write)
    # Pytest's captured exception argument would itself retain the write frame.
    monkeypatch.setattr("mtplx.cache_bank.cold_tier.logger.warning", lambda *args: None)
    pending = PendingWrite(
        entry_id="test", token_ids=(1,),
        metadata={"prefix_len": 1, "nbytes": 1024},
        payload_spec={}, tensors={"tensor": b"x" * 1024}, pinned_nbytes=1024,
    )
    released = weakref.ref(pending)
    try:
        assert tier._admit_write(pending.pinned_nbytes)
        tier._queue.put(pending)
        del pending
        assert tier.flush(timeout_s=5)
        assert released() is None
        assert tier._pending_bytes == 0
        assert tier.stats()["write_failures"] == int(outcome == "error")
    finally:
        tier.close()
