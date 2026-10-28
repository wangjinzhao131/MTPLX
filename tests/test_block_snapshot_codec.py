from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mtplx.cache_bank.codec import decode_payload, decode_payload_prefix, encode_payload
from mtplx.cache_state import BlockOwnedKVCache, CacheSnapshot, restore_cache


def _state(offset=10):
    keys = tuple(
        mx.arange(start * 2, (start + 4) * 2, dtype=mx.float32).reshape(1, 1, 4, 2)
        for start in range(0, offset, 4)
    )
    return {
        "kind": "mtplx_block_kv_v1",
        "block_size": 4,
        "offset": offset,
        "keys": keys,
        "values": tuple(block + 100 for block in keys),
    }


def _encode(state, *, codec_block_size=256, mtp_state=None):
    snapshot = CacheSnapshot(
        states=(state,), meta_states=(("4", str(state["offset"]), "contiguous_eval"),)
    )
    return encode_payload(
        cache_snapshot=snapshot,
        logits=None,
        hidden=None,
        mtp_history_snapshot=(
            None if mtp_state is None else CacheSnapshot(states=(mtp_state,), meta_states=(None,))
        ),
        block_size=codec_block_size,
    )


def _fields(encoded):
    return {
        key["value"]: value
        for key, value in encoded.spec["cache_snapshot"]["states"]["items"][0]["items"]
    }


@pytest.mark.parametrize("codec_block_size", [2, 256])
def test_block_snapshot_full_round_trip_preserves_physical_blocks(codec_block_size):
    state = _state()
    encoded = _encode(state, codec_block_size=codec_block_size)
    decoded = decode_payload(encoded.spec, encoded.tensors.__getitem__)
    result = decoded.cache_snapshot.states[0]

    assert (result["kind"], result["block_size"], result["offset"]) == (
        "mtplx_block_kv_v1", 4, 10
    )
    for name in ("keys", "values"):
        assert isinstance(result[name], tuple)
        assert len(result[name]) == 3
        assert [block.tolist() for block in result[name]] == [block.tolist() for block in state[name]]


@pytest.mark.parametrize("codec_block_size", [2, 256])
@pytest.mark.parametrize("prefix", [0, 1, 4, 5, 8, 9, 10, 20])
def test_block_snapshot_prefix_reads_only_global_prefix(codec_block_size, prefix):
    state = _state()
    encoded = _encode(state, codec_block_size=codec_block_size)
    reads = []

    def read_tensor(name):
        reads.append(name)
        return encoded.tensors[name]

    decoded = decode_payload_prefix(encoded.spec, read_tensor, cache_prefix_len=prefix)
    result = decoded.cache_snapshot.states[0]
    limit = min(prefix, state["offset"])
    assert result["offset"] == limit
    assert result["block_size"] == 4
    assert decoded.cache_snapshot.meta_states == ((None, None, None),)
    expected_reads = set()
    for name in ("keys", "values"):
        assert len(result[name]) == (limit + 3) // 4
        for index, block in enumerate(result[name]):
            take = min(4, limit - index * 4)
            assert block.tolist() == state[name][index][..., :take, :].tolist()
            block_spec = _fields(encoded)[name]["items"][index]
            if block_spec["kind"] == "tensor":
                expected_reads.add(block_spec["name"])
            else:
                expected_reads.update(
                    chunk["name"] for chunk in block_spec["blocks"] if chunk["start"] < take
                )
    assert set(reads) == expected_reads


def test_empty_block_snapshot_prefix_and_full_round_trip():
    state = _state(0)
    encoded = _encode(state)
    assert encoded.tensors == {}
    assert decode_payload(encoded.spec, encoded.tensors.__getitem__).cache_snapshot.states == (state,)
    decoded = decode_payload_prefix(encoded.spec, encoded.tensors.__getitem__, cache_prefix_len=4)
    assert decoded.cache_snapshot.states == (state,)


def test_block_snapshot_mtp_prefix_is_one_token_behind_target():
    encoded = _encode(_state(), codec_block_size=2, mtp_state=_state(9))
    decoded = decode_payload_prefix(
        encoded.spec,
        encoded.tensors.__getitem__,
        cache_prefix_len=5,
        mtp_history_prefix_len=4,
    )
    target = decoded.cache_snapshot.states[0]
    history = decoded.mtp_history_snapshot.states[0]
    assert target["offset"] == 5
    assert [block.shape[2] for block in target["keys"]] == [4, 1]
    assert history["offset"] == 4
    assert [block.shape[2] for block in history["keys"]] == [4]


@pytest.mark.parametrize("field,value", [
    ("block_size", {"kind": "int", "value": 0}),
    ("block_size", {"kind": "int", "value": True}),
    ("offset", {"kind": "int", "value": -1}),
    ("offset", {"kind": "str", "value": "10"}),
    ("keys", {"kind": "tuple", "items": []}),
    ("keys", {"kind": "tuple", "items": 4}),
    ("values", {"kind": "none"}),
])
def test_block_snapshot_prefix_rejects_invalid_geometry_before_reads(field, value):
    encoded = _encode(_state())
    fields = _fields(encoded)
    fields[field].clear()
    fields[field].update(value)
    reads = []
    with pytest.raises(ValueError):
        decode_payload_prefix(encoded.spec, lambda name: reads.append(name), cache_prefix_len=1)
    assert reads == []


@pytest.mark.parametrize("shape", [[1, 1, 3, 2], [1, 1, 5, 2], [1, 1, "4", 2], [1, 1, 4]])
def test_block_snapshot_prefix_rejects_invalid_tensor_shape_before_reads(shape):
    encoded = _encode(_state())
    _fields(encoded)["keys"]["items"][0]["shape"] = shape
    reads = []
    with pytest.raises(ValueError, match="tensor shape"):
        decode_payload_prefix(encoded.spec, lambda name: reads.append(name), cache_prefix_len=1)
    assert reads == []


def test_legacy_native_ssd_snapshot_restores_into_block_cache():
    keys = mx.arange(20, dtype=mx.float32).reshape(1, 1, 10, 2)
    encoded = encode_payload(
        cache_snapshot=CacheSnapshot(states=((keys, keys + 100),), meta_states=("",)),
        logits=None, hidden=None, mtp_history_snapshot=None,
    )
    decoded = decode_payload(encoded.spec, encoded.tensors.__getitem__)
    cache = BlockOwnedKVCache(block_size=4)
    restore_cache([cache], decoded.cache_snapshot)
    assert cache.offset == 10
    assert len(cache.key_blocks) == 3
    assert cache.keys.tolist() == keys.tolist()
    assert cache.values.tolist() == (keys + 100).tolist()


@pytest.mark.parametrize("prefix", [None, 0, 5])
def test_block_ssd_snapshot_restores_into_native_cache(prefix):
    state = _state()
    encoded = _encode(state)
    if prefix is None:
        decoded = decode_payload(encoded.spec, encoded.tensors.__getitem__)
    else:
        decoded = decode_payload_prefix(encoded.spec, encoded.tensors.__getitem__, cache_prefix_len=prefix)
    cache = KVCache()
    restore_cache([cache], decoded.cache_snapshot)
    limit = state["offset"] if prefix is None else prefix
    assert cache.offset == limit
    if limit:
        keys = mx.arange(limit * 2, dtype=mx.float32).reshape(1, 1, limit, 2)
        assert cache.state[0].tolist() == keys.tolist()
        assert cache.state[1].tolist() == (keys + 100).tolist()
    else:
        assert cache.keys is None and cache.values is None


def test_block_ssd_prefix_restores_and_appends_to_short_tail():
    encoded = _encode(_state(), codec_block_size=2)
    decoded = decode_payload_prefix(encoded.spec, encoded.tensors.__getitem__, cache_prefix_len=5)
    cache = BlockOwnedKVCache(block_size=8)
    restore_cache([cache], decoded.cache_snapshot)
    keys = mx.arange(10, 18, dtype=mx.float32).reshape(1, 1, 4, 2)
    cache.update_without_fetch(keys, keys + 100)

    assert cache.block_size == 4
    assert cache.offset == 9
    expected = mx.arange(18, dtype=mx.float32).reshape(1, 1, 9, 2)
    assert cache.keys.tolist() == expected.tolist()
    assert cache.values.tolist() == (expected + 100).tolist()
