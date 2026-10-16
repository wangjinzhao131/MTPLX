# Local Qwen optimization profiles

This fork integrates the accepted local kernels into the common MTPLX loader. Set `MTPLX_LOCAL_OPT` before starting Python; leaving it unset preserves upstream behavior. There are no additional runtime dependencies. Models retain their original weights.

| Profile | Checkpoint | Execution |
| --- | --- | --- |
| `qwen35-mtp` | `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16` | Fused MoE verification for 1–3 tokens, compiled layer islands and complete GDN prefix capture; original MTP weights |
| `qwen35-ar` | `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16` | Affine6/group64 FP16 MoE, fused decode and compiled layer islands; AR only |
| `qwen27-mtp` | `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` | Dense projection kernels, prefill GEMM, compiled MLP and GDN capture; original MTP weights |

The optimized paths target Apple Silicon, batch size one and FP16 activations. Unsupported tensor shapes use their native fallback where supported; installation rejects incompatible models or quantization. These profiles skip the upstream Qwen fusion installers because they replace the same layers. Adapters and projection requantization are rejected. Use the serial scheduler and the original model/draft head. The profiles are text-only; the server advertises no vision and rejects image requests. Unset the variable to restore upstream behavior and vision support.

## Install

```sh
git clone https://github.com/wangjinzhao131/MTPLX.git
cd MTPLX
git checkout local-qwen-optimizations
uv venv --python 3.14
uv pip install -e '.[server]' 'mlx-lm==0.31.3' 'transformers==5.14.1' pytest
```

MLX is pinned to 0.32.2 by the project. The validation environment used Python 3.14.5 on an M1 Max with 64 GB unified memory. Substitute your downloaded checkpoint directories below. The server binds only to localhost.

## OpenAI-compatible API

These recipes select `performance-cold` to retain native caches and in-model prefill and disable background warmup. The 35B recipe restores a 128K context window; the 27B recipe retains the 4K integration-test setting. The default `sustained` profile enables external prefill and paged caches that can bypass these kernels. Full-window and concurrent-load performance have not been validated for the port.

Enable `MTPLX_SUSTAINED_PREFILL=1` separately to split long input into bounded chunks while retaining native caches. Without this switch, the AR path forwards the entire prompt even when `--prefill-chunk-tokens` is set. The 35B recipe uses 64-token chunks; this does not limit the context window. For this checkpoint, chunks of 128 tokens or more can trigger MLX 0.32.2's sorted expert-matmul path to copy the fused gate/up weight views, adding about 16.25 GiB of temporary allocations across the model.

Set both the global chunk size and the restored-suffix limit below. The request-level CLI option alone does not cover background session commits, and the default 512-token suffix fast path bypasses chunking. These settings bound both foreground and background prefill while retaining the decode kernels and memory guards. Background warmup remains disabled in this recipe.

35B now supports MTP through `qwen35-mtp`. The injected model preserves single-token layer islands, and short verification islands return the hidden states and every recurrent prefix needed to commit accepted drafts or roll back rejected ones. Use depth 1 below; depth 2 is correct but was slower in the measured workload. For the AR comparison profile, select `qwen35-ar` and replace the MTP mode/depth/verification flags with `--stock-ar`.

```sh
MTPLX_LOCAL_OPT=qwen35-mtp MTPLX_CONTEXT_COPY=0 MTPLX_WARMUP_EXTENDED=0 \
MTPLX_SUSTAINED_PREFILL=1 \
MTPLX_PREFILL_CHUNK_SIZE=64 MTPLX_SMALL_SUFFIX_FUSED_MAX=64 \
MLX_MAX_MB_PER_BUFFER=16384 MLX_MAX_OPS_PER_BUFFER=4096 \
.venv/bin/python -m mtplx.server.openai \
  --model /path/to/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16 \
  --model-id local-qwen35 --host 127.0.0.1 --port 8008 \
  --profile performance-cold --generation-mode mtp --depth 1 \
  --verify-strategy capture_commit --verify-core stock --scheduler-mode serial --fan-mode default \
  --context-window 131072 --prefill-chunk-tokens 64 --warmup-tokens 0 --no-enable-thinking
```

27B retains MTP and all prefix states needed to commit only the accepted draft prefix. Select stock verification so the compiled capture hook can engage; the server's upstream default selects a different capture kernel. The default Qwen backend uses committed MTP history and the stock draft core. Local profiles bypass draft-head requantization in both CLI and server.

```sh
MTPLX_LOCAL_OPT=qwen27-mtp MTPLX_CONTEXT_COPY=0 MTPLX_SUSTAINED_PREFILL=1 \
MTPLX_WARMUP_EXTENDED=0 \
.venv/bin/python -m mtplx.server.openai \
  --model /path/to/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16 \
  --model-id local-qwen27 --host 127.0.0.1 --port 8008 \
  --profile performance-cold --generation-mode mtp --depth 1 \
  --verify-strategy capture_commit --verify-core stock --scheduler-mode serial --fan-mode default --context-window 4096 --warmup-tokens 0 --no-enable-thinking
```

`GET /health` and response statistics expose `local_optimization`. The profiles also work through `mtplx.runtime.load`; request `mtp=True` and use the settings above with `generate_mtpk` for either MTP profile. The separate `qwen35-ar` profile continues to disable MTP explicitly.

## Reproduce correctness and timing

```sh
.venv/bin/pytest -q tests/local_optimizations
# Run these separately with the same model, environment, and mode:
.venv/bin/python scripts/local_opt_smoke.py /path/to/model --mode ar --output baseline.json
MTPLX_LOCAL_OPT=qwen35-ar .venv/bin/python scripts/local_opt_smoke.py /path/to/model --mode ar --output optimized.json
```

Use `--mode mtp --depth 1` with `qwen35-mtp` or `qwen27-mtp` for MTP comparisons; `--depth 2` measures two-token drafting. For 35B set the two `MLX_MAX_*` variables in both runs. For either MTP comparison set `MTPLX_CONTEXT_COPY=0 MTPLX_SUSTAINED_PREFILL=1 MTPLX_CAPTURE_CUSTOM_KERNEL=stock` in both runs. Each invocation warms up, runs English/Chinese/code prompts with greedy sampling, checks streamed tokens against committed output, respects EOS, and saves token IDs, timings, versions, checkpoint config hash, memory, and execution counters. Compare the `rows[*].tokens` arrays between reports. Default output length is 128 tokens; `--tokens` changes it.

## Provenance and scope

- 35B modules were ported from `mlx-qwen35-opt` commit `1300481`. The original acceptance used 3 prompts × 512 tokens × 3 repetitions: all 4,608 tokens matched, with about 58–59% faster decode in its standalone mlx-lm AR loop. That number is not an MTPLX service benchmark.
- 27B modules were ported from `mlx-qwen38-27b-opt` commit `8c45e99`. The original accepted dense stack matched all 4,608 greedy tokens and improved decode by roughly 27–42% and long prefill by roughly 17–23% against its baseline. Later GDN capture improvements were measured separately. Neither original 50% goal was reached; gains from separate experiments must not be multiplied.
- Package imports were made relative; the 35B MTP wrapper preserves the optimized target forward and hidden-state contract; GDN capture is an instance-local hook into this checkout instead of a monkeypatch against a separately pinned MTPLX tree. Tests cover exact outputs, cache states, partial-prefix commit, fallback and undo. Existing MLX copyright/license notices are retained.
- Rejected AMX/ANE experiments, approximate heads, adaptive depth, LoRA and unrelated oMLX work are not installed.

## Fork integration measurements (2026-10-16)

Separate processes, one warmup, greedy English/Chinese/code prompts × 128 generated tokens, original checkpoints, no thinking. Baseline is this MTPLX checkout without `MTPLX_LOCAL_OPT`; optimized uses the corresponding profile. These are short integration checks, not repetitions of the earlier standalone acceptance benchmarks.

| Profile | Baseline decode tok/s (EN / ZH / code) | Optimized decode tok/s | Gain | Exact tokens |
| --- | --- | --- | --- | --- |
| 35B AR | 47.54 / 47.31 / 47.55 | 66.82 / 67.01 / 66.58 | 40.0–41.6% | 384 / 384 |
| 27B MTP depth 1 | 21.43 / 19.94 / 21.43 | 25.72 / 23.89 / 25.70 | 19.8–20.0% | 384 / 384 |

35B recorded 400 layer-island calls; 27B recorded 10,272 compiled GDN capture calls, including warmup. Peak MLX allocation was approximately 28.81 GB for both 35B runs and 20.44 → 20.41 GB for 27B. Raw token/timing reports are in [benchmarks/local-optimizations](benchmarks/local-optimizations). No long-context or concurrent-load performance claim is made for this port.

Validation: 560 combined kernel, loader, cache, MTP, and server tests passed. A subsequent targeted run passed 8 profile/draft-head tests (566 distinct tests across both runs). Both real-model HTTP services passed health/profile checks, identical ordinary/SSE output for 64 greedy tokens, and HTTP 400 rejection of image requests. The 27B health receipt confirms that draft-head requantization was skipped. See [HTTP receipts](benchmarks/local-optimizations/http-smoke.json). The full upstream suite was not run.

The earlier 35B AR service validation reported 131,072 tokens for both its configured and execution windows. Two consecutive streamed requests with mixed code/prose and an OpenAI tool schema used 8,860 and 9,007 prompt tokens. The cold request used 139 chunks and delivered its first token in 29.63 seconds; the second reused 8,906 cached tokens and delivered its first token in 0.49 seconds. Both decoded at about 62 tok/s. With the 64-token foreground, global and suffix limits above, peak allocation stayed at 30.75 GB after background session commits, with no prefill memory aborts. See the [long-input HTTP receipt](benchmarks/local-optimizations/long-context-http-smoke.json). This is a multi-turn integration check, not a full 128K stress test or a replay of a private Hermes conversation.


## Combined 35B MTP integration (2026-10-16)

The combined profile extends fused routing, gate/up, down and weighted reduction to 1–3 tokens, then compiles short verification islands with complete GDN prefix capture. Single-token target calls retain the original islands through MTP injection. Draft weights and target weights remain unchanged.

Separate-process comparison using the same three prompts, 128 greedy tokens each, one warmup, and the environment documented above:

| Execution | Decode tok/s (EN / ZH / code) | Mean tok/s | Peak MLX GB |
| --- | --- | --- | --- |
| Native MTP, depth 1 | 49.51 / 51.98 / 50.60 | 50.70 | 29.38 |
| Optimized AR | 67.30 / 67.02 / 67.04 | 67.12 | 28.81 |
| Combined optimized MTP, depth 1 | 66.15 / 68.36 / 66.36 | 66.96 | 29.38 |
| Combined optimized MTP, depth 2 | 65.01 / 65.11 / 63.03 | 64.39 | 29.38 |

All four runs produced exactly the same 384 tokens. Combined depth 1 improves mean throughput by 32.1% over native MTP and is effectively tied with optimized AR (0.24% lower in this run). Depth 2 is slower, so depth 1 is the serving default. These single runs do not establish a universal fastest configuration or an additional speedup over optimized AR. Reports are `qwen35-*-mtp-integration.json` and `qwen35-optimized-mtp-depth2.json` in the benchmark directory. The depth-1 combined run recorded 226 capture-island calls and 6,780 GDN capture calls.

Validation passed the affected kernel/profile/MTP suite, plus 643 serving/cache/MTP tests (4 skipped, 557 subtests). Tests exercise 1–3 token routing, exact captured recurrent prefixes, rejection rollback and correction, hidden-state variants, fallbacks, and undo after MTP injection. The full upstream suite was not run.

The running combined service retains port 8008 and a 131,072-token window. A short greedy HTTP AR response matched SSE MTP output. Two streamed tool-schema requests used 8,859 and 9,006 prompt tokens, decoding at 60.54 and 58.74 tok/s; the second reused 8,905 cached tokens and delivered its first token in 0.484 seconds. Peak MLX allocation after background commits was 32.09 GB. See [combined MTP HTTP receipt](benchmarks/local-optimizations/qwen35-mtp-http-smoke.json). This verifies roughly 9K multi-turn serving; full 128K stress and concurrent load remain untested.
