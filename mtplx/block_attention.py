"""Blockwise attention over BlockOwnedKVCache physical blocks."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def _apply_causal_block_mask(
    scores: mx.array,
    *,
    block_start: int,
    cached_prefix_len: int,
) -> mx.array:
    q_len = int(scores.shape[-2])
    block_len = int(scores.shape[-1])
    if block_start + block_len <= cached_prefix_len:
        return scores
    query_positions = cached_prefix_len + mx.arange(q_len)
    key_positions = block_start + mx.arange(block_len)
    allowed = query_positions[:, None] >= key_positions[None, :]
    return mx.where(allowed, scores, mx.finfo(scores.dtype).min)


def blockwise_attention(
    *,
    queries: mx.array,
    cache: Any,
    scale: float,
    cached_prefix_len: int,
) -> mx.array:
    """Compute causal GQA attention over cache physical blocks without concat.

    This is a high-level Step 11B probe. It avoids constructing one growing K/V
    tensor from independent blocks, but it is not a final Metal kernel and must
    pass the normal output-token parity gates before any product claim.
    """

    blocks = cache.active_block_slices()
    if not blocks:
        raise ValueError("blockwise attention requires at least one active KV block")
    query_heads = int(queries.shape[1])
    kv_heads = int(blocks[0][1].shape[1])
    if query_heads % kv_heads:
        raise ValueError(f"query_heads={query_heads} must be divisible by kv_heads={kv_heads}")
    # Broadcast each physical KV head across its query group without repeating it.
    q = queries.astype(mx.float32).reshape(
        queries.shape[0], kv_heads, query_heads // kv_heads, *queries.shape[2:]
    )
    running_max = None
    running_denom = None
    running_acc = None

    for block_start, keys, values in blocks:
        k = keys.astype(mx.float32)[:, :, None]
        v = values.astype(mx.float32)[:, :, None]
        scores = mx.matmul(q, k.transpose(0, 1, 2, 4, 3)) * float(scale)
        scores = _apply_causal_block_mask(
            scores,
            block_start=int(block_start),
            cached_prefix_len=int(cached_prefix_len),
        )
        local_max = mx.max(scores, axis=-1, keepdims=True)
        weights = mx.exp(scores - local_max)
        local_denom = mx.sum(weights, axis=-1, keepdims=True)
        local_acc = mx.matmul(weights, v)

        if running_max is None:
            running_max = local_max
            running_denom = local_denom
            running_acc = local_acc
            continue

        new_max = mx.maximum(running_max, local_max)
        old_scale = mx.exp(running_max - new_max)
        new_scale = mx.exp(local_max - new_max)
        running_acc = running_acc * old_scale + local_acc * new_scale
        running_denom = running_denom * old_scale + local_denom * new_scale
        running_max = new_max

    if running_acc is None or running_denom is None:
        raise ValueError("blockwise attention did not accumulate any blocks")
    return (running_acc / running_denom).reshape(
        queries.shape[0], query_heads, queries.shape[2], -1
    ).astype(queries.dtype)
