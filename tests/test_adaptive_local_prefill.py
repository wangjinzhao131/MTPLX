"""Local Qwen adaptive widths preserve checkpoint edges and suffix safety."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx import generation as gen, system_memory as sm
from mtplx.session_bank import SessionBank
from test_checkpoint_pre_image import _runtime

GIB = 1024**3


def reading(headroom):
    return sm.SystemMemory(
        total_bytes=64 * GIB, available_bytes=int((2 + headroom) * GIB),
        wired_bytes=32 * GIB, level_percent=20,
    )


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "512")
    monkeypatch.setenv("MTPLX_SMALL_SUFFIX_FUSED_MAX", "512")
    monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", "1")
    monkeypatch.setattr(sm, "read_system_memory", lambda: reading(10))
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 0)


@pytest.mark.parametrize("headroom,position,requested,expected", [
    (10, 0, 512, 512), (3.5, 32768, 512, 512),
    (3.49, 0, 512, 64), (10, 49152, 512, 64),
    (10, 100000, 32, 32), (None, 0, 512, 64),
])
def test_width_policy(monkeypatch, headroom, position, requested, expected):
    rt = SimpleNamespace(local_optimization="qwen35-mtp")
    monkeypatch.setattr(sm, "read_system_memory", lambda: None if headroom is None else reading(headroom))
    assert gen._adaptive_prefill_chunk_size(rt, position, requested) == expected


def test_policy_does_not_change_other_profiles_or_widen_admission():
    assert gen._adaptive_prefill_chunk_size(SimpleNamespace(), 100000, 512) == 512
    with gen.prefill_chunk_size_override(64):
        assert gen._adaptive_prefill_chunk_size(
            SimpleNamespace(local_optimization="qwen35-mtp"), 0, 512,
        ) == 64


def test_reclaimable_allocator_pool_counts_as_available(monkeypatch):
    monkeypatch.setattr(sm, "read_system_memory", lambda: reading(3))
    monkeypatch.setattr(mx, "get_cache_memory", lambda: GIB)
    assert gen._adaptive_prefill_chunk_size(
        SimpleNamespace(local_optimization="qwen35-mtp"), 0, 512,
    ) == 512


def test_spans_recheck_memory_lazily_and_preserve_edges(monkeypatch):
    rt = SimpleNamespace(local_optimization="qwen35-mtp")
    spans = gen._adaptive_prefill_spans(rt, [(0, 512), (512, 661), (661, 1024)])
    assert next(spans) == (0, 512)
    monkeypatch.setattr(sm, "read_system_memory", lambda: reading(3))
    remaining = list(spans)
    assert remaining[0] == (512, 576)
    assert all(end - start <= 64 for start, end in remaining)
    assert 661 in [end for _, end in remaining]
    assert remaining[-1][1] == 1024
    assert all(left[1] == right[0] for left, right in zip(remaining, remaining[1:]))


@pytest.mark.parametrize("policy", ["cycle", "committed"])
def test_cold_and_warm_fallback_keep_exact_state_and_marker(monkeypatch, policy):
    rt = _runtime()
    rt.local_optimization = "qwen35-mtp"
    ids = [i % 14 for i in range(1025)]
    bank = SessionBank()
    calls = 0

    def memory():
        nonlocal calls
        calls += 1
        return reading(10 if calls == 1 else 3)

    monkeypatch.setattr(sm, "read_system_memory", memory)
    state = gen.restore_or_prefill_prompt_state(
        rt, ids, session_bank=bank, mtp_history_policy=policy,
        store_prefix_snapshot=True, explicit_cache_boundary=661,
    )
    assert rt.model.forwards[0] == 512
    assert max(rt.model.forwards[1:]) <= 64
    assert 661 in [row[0] for row in state.gdn_boundaries]
    rt.model.forwards.clear()
    extended = ids + [14] * 256
    warm = gen.restore_or_prefill_prompt_state(
        rt, extended, session_bank=bank, mtp_history_policy=policy,
        store_prefix_snapshot=False,
    )
    assert warm.cached_tokens == len(ids)
    assert max(rt.model.forwards) <= 64  # Fused suffix must honor headroom too.
    expected = rt.forward_ar(mx.array([extended]), cache=rt.make_cache(), logits_keep=1)
    assert mx.array_equal(warm.logits, expected[:, -1, :]).item()
    assert warm.trunk_cache[0].offset == len(extended)
    assert warm.explicit_cache_boundary == 661


def test_admission_can_price_validated_smaller_local_widths():
    from mtplx.server.openai import _admission_narrow_widths

    rt = SimpleNamespace(local_optimization="qwen35-mtp")
    assert _admission_narrow_widths(rt, [512]) == [256, 64]
    assert _admission_narrow_widths(rt, [64]) == []
