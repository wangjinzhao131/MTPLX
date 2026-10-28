"""Select inherited checkpoints from SSD metadata before allocating arrays."""

import mlx.core as mx
import pytest

from mtplx.cache_bank.codec import (
    _encoded_tree_nbytes,
    decode_gdn_boundaries,
    decode_payload_prefix,
    encode_payload,
)
from mtplx.cache_state import CacheSnapshot
from mtplx.checkpoint_anchors import AnchorPlan, record_nbytes, retain_checkpoints


def _fixture():
    records = [
        (
            position,
            CacheSnapshot(
                states=(mx.full((1, 1, 4, 2), position, dtype=mx.float32),),
                meta_states=(None,),
            ),
            mx.full((1, 1, 2), position, dtype=mx.float32),
        )
        for position in (128, 256, 512, 768, 1024)
    ]
    encoded = encode_payload(
        cache_snapshot=CacheSnapshot((), ()), logits=None, hidden=None,
        mtp_history_snapshot=None, gdn_boundaries=records, block_size=2,
    )
    return records, encoded


def _tensor_names(value):
    if isinstance(value, dict):
        if "name" in value:
            return {value["name"]}
        return set().union(*(_tensor_names(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(_tensor_names(item) for item in value))
    return set()


@pytest.mark.parametrize("capacity", [0, 1, 2, 8])
def test_inheritance_reads_only_budgeted_records_below_frontier(capacity):
    records, encoded = _fixture()
    plan = AnchorPlan(
        budget_bytes=capacity * record_nbytes(records[0]),
        prompt_end=768, restore_point=768, tail_backoff=0, ceiling=768,
    )
    selected = retain_checkpoints(records, plan)
    # The partial restore already paid for this serving checkpoint.
    existing = [records[3]]
    expected_positions = [int(r[0]) for r in selected]
    expected_reads = set().union(*(
        _tensor_names(spec)
        for spec in encoded.spec["gdn_boundaries"]
        if int(spec["tokens"]) in expected_positions and int(spec["tokens"]) != 768
    ))
    reads = []

    def read(name):
        reads.append(name)
        return encoded.tensors[name]

    decoded = decode_gdn_boundaries(
        encoded.spec, read, plan=plan, existing_records=existing,
    )
    assert [r[0] for r in decoded] == expected_positions
    assert set(reads) == expected_reads
    assert len(reads) == len(expected_reads)
    assert sum(record_nbytes(r) for r in decoded) <= plan.budget_bytes
    if 768 in expected_positions:
        assert next(r for r in decoded if r[0] == 768) is existing[0]


def test_metadata_prices_chunked_tensors_once_and_includes_hidden():
    records, encoded = _fixture()
    for record, spec in zip(records, encoded.spec["gdn_boundaries"]):
        assert _encoded_tree_nbytes(spec) == record_nbytes(record) == 40


def test_missing_unselected_blob_does_not_break_inheritance():
    records, encoded = _fixture()
    plan = AnchorPlan(
        budget_bytes=40, prompt_end=512, restore_point=512,
        tail_backoff=0, ceiling=512,
    )
    selected_spec = encoded.spec["gdn_boundaries"][2]
    available = {name: encoded.tensors[name] for name in _tensor_names(selected_spec)}
    decoded = decode_gdn_boundaries(encoded.spec, available.__getitem__, plan=plan)
    assert [r[0] for r in decoded] == [512]
    assert float(decoded[0][2][0, 0, 0]) == 512


@pytest.mark.parametrize("prefix", [0, 1, 190, 191, 300])
def test_short_native_snapshot_prefix_slices_unchunked_tensors(prefix):
    tensor = mx.arange(191, dtype=mx.float32).reshape(1, 1, 191, 1)
    encoded = encode_payload(
        cache_snapshot=CacheSnapshot(states=((tensor, tensor),), meta_states=(None,)),
        logits=None, hidden=None, mtp_history_snapshot=None,
    )
    assert encoded.spec["cache_snapshot"]["states"]["items"][0]["items"][0]["kind"] == "tensor"
    decoded = decode_payload_prefix(
        encoded.spec, encoded.tensors.__getitem__, cache_prefix_len=prefix,
    )
    for actual in decoded.cache_snapshot.states[0]:
        assert actual.shape == (1, 1, min(prefix, 191), 1)
        assert bool(mx.array_equal(actual, tensor[:, :, :prefix, :]))
