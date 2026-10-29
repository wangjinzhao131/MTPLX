"""Promoting an SSD donor must not inherit a RAM lease's admission bill."""

from types import SimpleNamespace

import pytest

from mtplx.server import openai as srv
from test_memguard_admission import (
    GIB, _Machine, _flash_next_state, _install, _manager, _put,
    _served_profile,  # noqa: F401
)


@pytest.mark.parametrize("prompt_len", [11_000, 200_000])
@pytest.mark.parametrize("native_owner", [False, True])
def test_ssd_promotion_prices_full_miss_plus_donor(monkeypatch, prompt_len, native_owner):
    manager = _manager()
    state = _flash_next_state(manager)
    _install(monkeypatch, _Machine(manager.bank, base_gib=40, cache_gib=0, host_gib=0))
    prompt = list(range(prompt_len))
    baseline = {}
    assert srv._prefill_admission_shed(
        state, prompt_ids=prompt, session_bank=manager.bank, session_id="s", pricing=baseline,
    ) is None
    if native_owner:
        monkeypatch.setenv("MTPLX_TOOL_POSTCOMMIT_DEBOUNCE", "1")
        state.lock = SimpleNamespace(locked=lambda: True)
        state.model_scheduler = SimpleNamespace(is_owner_thread=lambda: True)
        monkeypatch.setattr(manager.bank, "native_lease_can_extend", lambda *a, **kw: True)
    manager.bank.cold_tier = SimpleNamespace(mode="read-write")
    calls = []
    donor_bytes = 2 * GIB

    def plan(tokens, **kwargs):
        calls.append(kwargs)
        return {"keys": (), "reuse_tokens": prompt_len - 100, "mode": "ssd_exact",
                "source": None, "restore_donor_nbytes": donor_bytes}

    monkeypatch.setattr(manager.bank, "restore_plan", plan)
    monkeypatch.setattr(srv, "_lease_cache_shape", lambda *a: pytest.fail("SSD cannot borrow a RAM lease"))
    pricing = {}
    assert srv._prefill_admission_shed(
        state, prompt_ids=prompt, session_bank=manager.bank, session_id="s",
        pricing=pricing, required_boundary=128,
    ) is None
    assert calls and calls[0]["include_cold"] is True
    assert calls[0]["required_boundary"] == 128
    growth = pricing["growth"]
    assert growth["planned_reused_tokens"] == prompt_len - 100
    assert growth["restore_priced_as_full_miss"] is True
    assert growth["reused_tokens"] == 0  # Cost includes failed hydration and a shorter RAM fallback.
    assert growth["restore_donor_bytes"] == donor_bytes
    for peak in ("prefill_end_bytes", "repage_bytes", "decode_start_bytes", "growth_bytes"):
        assert growth[peak] == baseline["growth"][peak] + donor_bytes


def test_healthy_full_miss_shortcut_does_not_skip_ssd_donor(monkeypatch):
    manager = _manager()
    state = _flash_next_state(manager)
    _install(monkeypatch, _Machine(manager.bank, base_gib=80, cache_gib=0, host_gib=0))
    prompt = list(range(11_000))
    assert srv._prefill_admission_shed(
        state, prompt_ids=prompt, session_bank=manager.bank, session_id="s",
    ) is None
    manager.bank.cold_tier = SimpleNamespace(mode="read-write")
    monkeypatch.setattr(manager.bank, "restore_plan", lambda *a, **kw: {
        "keys": (), "reuse_tokens": 10_000, "mode": "ssd_exact", "source": None,
        "restore_donor_nbytes": 20 * GIB,
    })
    receipt = srv._prefill_admission_shed(
        state, prompt_ids=prompt, session_bank=manager.bank, session_id="s",
    )
    assert receipt["refused"] is True
    assert receipt["reusable_prefix_tokens"] == 10_000
    assert receipt["reusable_prefix_mode"] == "ssd_exact"
    assert receipt["restore_copies_prefix"] is True
    assert receipt["growth"]["restore_donor_bytes"] == 20 * GIB


def test_reclamation_reprices_the_ssd_donor(monkeypatch):
    manager = _manager()
    state = _flash_next_state(manager)
    _put(manager.bank, [99], session_id="idle", row_bytes=8 * GIB)
    _install(monkeypatch, _Machine(manager.bank, base_gib=80, cache_gib=0, host_gib=0))
    manager.bank.cold_tier = SimpleNamespace(mode="read-write")
    calls = []

    def plan(*args, **kwargs):
        donor = 10 * GIB if manager.bank._entries else GIB
        calls.append(donor)
        return {"keys": (), "reuse_tokens": 10_000, "mode": "ssd_exact", "source": None,
                "restore_donor_nbytes": donor}

    monkeypatch.setattr(manager.bank, "restore_plan", plan)
    receipt = srv._prefill_admission_shed(
        state, prompt_ids=list(range(11_000)), session_bank=manager.bank, session_id="s",
    )
    assert calls[0] == 10 * GIB and calls[-1] == GIB
    assert receipt["growth"]["restore_donor_bytes"] == GIB
    assert receipt["restore_replanned"][-1]["restore_donor_bytes"] == GIB
    assert receipt.get("refused") is not True
