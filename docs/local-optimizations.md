# Local Qwen optimization profiles

This fork integrates the accepted local kernels into the common MTPLX loader. Set `MTPLX_LOCAL_OPT` before starting Python; leaving it unset preserves upstream behavior. There are no additional runtime dependencies. Models retain their original weights.

| Profile | Checkpoint | Execution |
| --- | --- | --- |
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

These recipes select `performance-cold` to retain native caches and in-model prefill, disable background warmup, and use a 4K context window. The default `sustained` profile enables external prefill and paged caches that can bypass these kernels. Larger windows and concurrent serving have not been validated for the port.

35B uses AR because the measured layer islands span the native model forward; MTP injection would replace that forward. The loader disables MTP for this profile. Pass `--stock-ar` so the server also selects AR (the flag is the upstream name for `--generation-mode ar --no-load-mtp`; local kernels are still active).

```sh
MTPLX_LOCAL_OPT=qwen35-ar MTPLX_WARMUP_EXTENDED=0 \
MLX_MAX_MB_PER_BUFFER=16384 MLX_MAX_OPS_PER_BUFFER=4096 \
.venv/bin/python -m mtplx.server.openai \
  --model /path/to/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16 \
  --model-id local-qwen35 --host 127.0.0.1 --port 8008 \
  --profile performance-cold --stock-ar --scheduler-mode serial --fan-mode default --context-window 4096 --warmup-tokens 0 --no-enable-thinking
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

`GET /health` and response statistics expose `local_optimization`. The profiles also work through `mtplx.runtime.load`; request `mtp=False`/AR for 35B and use the settings above with `generate_mtpk` for 27B.

## Reproduce correctness and timing

```sh
.venv/bin/pytest -q tests/local_optimizations
# Run these separately with the same model, environment, and mode:
.venv/bin/python scripts/local_opt_smoke.py /path/to/model --mode ar --output baseline.json
MTPLX_LOCAL_OPT=qwen35-ar .venv/bin/python scripts/local_opt_smoke.py /path/to/model --mode ar --output optimized.json
```

Use `--mode mtp` and `qwen27-mtp` for 27B. For 35B set the two `MLX_MAX_*` variables in both runs. For 27B set `MTPLX_CONTEXT_COPY=0 MTPLX_SUSTAINED_PREFILL=1` in both runs. Each invocation warms up, runs English/Chinese/code prompts with greedy sampling, checks streamed tokens against committed output, respects EOS, and saves token IDs, timings, versions, checkpoint config hash, memory, and execution counters. Compare the `rows[*].tokens` arrays between reports. Default output length is 128 tokens; `--tokens` changes it.

## Provenance and scope

- 35B modules were ported from `mlx-qwen35-opt` commit `1300481`. The original acceptance used 3 prompts × 512 tokens × 3 repetitions: all 4,608 tokens matched, with about 58–59% faster decode in its standalone mlx-lm AR loop. That number is not an MTPLX service benchmark.
- 27B modules were ported from `mlx-qwen38-27b-opt` commit `8c45e99`. The original accepted dense stack matched all 4,608 greedy tokens and improved decode by roughly 27–42% and long prefill by roughly 17–23% against its baseline. Later GDN capture improvements were measured separately. Neither original 50% goal was reached; gains from separate experiments must not be multiplied.
- Package imports were made relative; the 35B forward accepts MTPLX's `return_hidden=False`; GDN capture is an instance-local hook into this checkout instead of a monkeypatch against a separately pinned MTPLX tree. Tests cover exact outputs, cache states, partial-prefix commit, fallback and undo. Existing MLX copyright/license notices are retained.
- Rejected AMX/ANE experiments, approximate heads, adaptive depth, LoRA and unrelated oMLX work are not installed.

## Fork integration measurements (2026-10-16)

Separate processes, one warmup, greedy English/Chinese/code prompts × 128 generated tokens, original checkpoints, no thinking. Baseline is this MTPLX checkout without `MTPLX_LOCAL_OPT`; optimized uses the corresponding profile. These are short integration checks, not repetitions of the earlier standalone acceptance benchmarks.

| Profile | Baseline decode tok/s (EN / ZH / code) | Optimized decode tok/s | Gain | Exact tokens |
| --- | --- | --- | --- | --- |
| 35B AR | 47.54 / 47.31 / 47.55 | 66.82 / 67.01 / 66.58 | 40.0–41.6% | 384 / 384 |
| 27B MTP depth 1 | 21.43 / 19.94 / 21.43 | 25.72 / 23.89 / 25.70 | 19.8–20.0% | 384 / 384 |

35B recorded 400 layer-island calls; 27B recorded 10,272 compiled GDN capture calls, including warmup. Peak MLX allocation was approximately 28.81 GB for both 35B runs and 20.44 → 20.41 GB for 27B. Raw token/timing reports are in [benchmarks/local-optimizations](benchmarks/local-optimizations). No long-context or concurrent-load performance claim is made for this port.

Validation: 560 combined kernel, loader, cache, MTP, and server tests passed. A subsequent targeted run passed 8 profile/draft-head tests (566 distinct tests across both runs). Both real-model HTTP services passed health/profile checks, identical ordinary/SSE output for 64 greedy tokens, and HTTP 400 rejection of image requests. The 27B health receipt confirms that draft-head requantization was skipped. See [HTTP receipts](benchmarks/local-optimizations/http-smoke.json). The full upstream suite was not run.
