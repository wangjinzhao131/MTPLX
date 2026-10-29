# Hermes-oriented optimizations and Chinese dashboard

This branch publishes the current MTPLX source and regression tests developed
on an M1 Max with 64 GiB unified memory. It is based on the upstream MTPLX fork;
original licenses and attribution are retained. It is a development snapshot,
not a claim that every optional path has passed production acceptance.

## Included

- Native Qwen cache ownership, recurrent checkpoints, shared-prefix selection,
  SSD boundary inheritance and resumable persistence.
- Chat `prompt_cache_key` and one explicit text-part
  `prompt_cache_breakpoint: {"mode": "explicit"}`. Reuse still requires matching
  token prefixes and compatible model/cache identities.
- Adaptive local prefill, sorted multi-token expert projection, native tool-loop
  leases, and deferred heavy tool postcommit work.
- Optional hard thinking-token budgets and Qwen thinking-mode cache identity.
- Chinese dashboard source and its compiled browser bundle.

External Hermes profile plugins and core patches are not part of this MTPLX
repository. Private conversations, local credentials, model weights and raw
machine-specific experiment artifacts are excluded.

## Upstream fixes integrated on 2026-10-29

These changes adapt the following upstream pull requests to this branch's
native ownership, explicit checkpoints and Chinese dashboard:

- [#656](https://github.com/youssofal/MTPLX/pull/656): partially decoded cache
  snapshots skip each layer's metadata as a whole, including tuple metadata.
- [#668](https://github.com/youssofal/MTPLX/pull/668) (also proposed in
  [#608](https://github.com/youssofal/MTPLX/pull/608)): exact SSD restores use
  the caller's target and MTP cache factories and preserve their layouts.
- [#609](https://github.com/youssofal/MTPLX/pull/609): metadata-only SSD prefix
  planning prevents a shorter RAM match from hiding a longer compatible SSD
  entry. The local adaptation includes the decoded donor in memory admission
  and retains RAM fallback if hydration fails. Exact token, identity, epoch and
  recurrent-checkpoint checks still govern reuse.
- [#652](https://github.com/youssofal/MTPLX/pull/652): cancelled requests retain
  completed prefill cache results; unknown results remain unknown.
- [#617](https://github.com/youssofal/MTPLX/pull/617): request details expose
  scheduler, lock, postcommit and response-tail waits.
- [#622](https://github.com/youssofal/MTPLX/pull/622): separate RAM and SSD
  cache summaries, including manifest age, session and restore coverage.
- [#623](https://github.com/youssofal/MTPLX/pull/623): settings show the real
  launch command, effective environment and value sources. Credentials are
  redacted; copied commands with redacted values need those values restored.

SSD preference is subject to the existing memory limits. Admission includes
the incoming donor plus a conservative copying estimate covering a failed
SSD restore's RAM fallback. It may therefore refuse a longer SSD restore under
low headroom. This integration does not claim a new inference speedup.

The integration passed focused SSD selection/fallback, cache-layout, memory
admission, native ownership, cancellation, launch-redaction and dashboard tests.
The Chinese dashboard passed TypeScript and production-build checks. The local
8008 service was restarted with unchanged launch files; the new monitoring
endpoint and one short generation request responded successfully. The RAM/SSD
overview and launch-settings pages were checked in a browser. No full upstream
suite or new long-context performance benchmark was run for this integration.

## Local serving settings

Install using the existing project instructions and local optimization profile
instructions in `docs/local-optimizations.md`. For the current 35B text-only
profile, the following supersedes that document's older fixed-64 example:

```sh
MTPLX_LOCAL_OPT=qwen35-mtp MTPLX_CONTEXT_COPY=0 MTPLX_WARMUP_EXTENDED=0 \
MTPLX_SUSTAINED_PREFILL=1 \
MTPLX_PREFILL_CHUNK_SIZE=512 MTPLX_SMALL_SUFFIX_FUSED_MAX=512 \
MTPLX_SESSION_BANK_IDLE_TTL_S=86400 MTPLX_TOOL_POSTCOMMIT_DEBOUNCE=1 \
MLX_MAX_MB_PER_BUFFER=16384 MLX_MAX_OPS_PER_BUFFER=4096 \
python -m mtplx.server.openai \
  --model /path/to/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16 \
  --model-id local-qwen35 --host 127.0.0.1 --port 8008 \
  --profile performance-cold --generation-mode mtp --depth 1 \
  --verify-strategy capture_commit --verify-core stock --scheduler-mode serial \
  --fan-mode default --context-window 131072 --prefill-chunk-tokens 512 \
  --warmup-tokens 0 --enable-thinking
```

This is a configuration example, not a universal memory guarantee. Keep all
memory guards enabled. RAM/SSD capacity and pressure eviction take precedence
over the 24-hour idle TTL. Adaptive width respects admission decisions and may
fall back to 64 tokens. Open `/dashboard/` for the Chinese monitoring interface.

## Evidence and limits

Earlier isolated local checks measured 8K cold prefill at 10.84 s versus
25.76 s and a 1K warm suffix at 1.65 s versus 3.49 s. A separately controlled
32K/near-128K tool-continuation run verified native ownership, zero restore-copy
charge and no added swapouts in that run. These measurements are workload- and
headroom-specific, not guarantees for arbitrary requests.

Low-headroom requests can still correctly fail memory admission, even with an
empty cache. The historical whole-loop 45% improvement target was not met.
Hard-budget tests cover CPU toy models; compiled production-model hard-budget
engagement still needs live acceptance. Shared-block attention remains an
experimental optional path, with known performance/parity limitations; native
cache serving is the validated path. Later global stop-debounce, startup root
warming and a three-band adaptive policy are not included as accepted features.

Publication checks include Python syntax, dashboard TypeScript checks and diff
whitespace validation. Existing focused runtime test results belong to their
individual changes; a full upstream suite and fresh model benchmark were not
rerun merely to publish this snapshot.
