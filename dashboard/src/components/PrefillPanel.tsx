import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { Loader2 } from "lucide-react";
import { Card } from "./Card";
import { fmtNumber, fmtSeconds, fmtTokS, truncateMiddle } from "../lib/utils";
import { usePrimaryActivePrefill, useDashboardStore } from "../state/store";
import { usePrefillHistory } from "../hooks/usePolling";

/**
 * PrefillPanel — the hero panel for "model is currently chewing the prompt".
 *
 * Live state (during chunked prefill):
 *   - Big progress bar with %, tokens-done / tokens-total
 *   - Live prefill tok/s computed from elapsed + tokens-done
 *   - ETA computed from remaining / live tok/s
 *   - Elapsed timer
 *   - Cached tokens (served from SessionBank, no compute)
 *
 * Idle state:
 *   - Last completed prefill summary (TTFT, prefill tok/s, cached/new)
 *   - 30-day historical mean prefill tok/s from prefill_history (LM Studio
 *     style baseline so users can sanity-check the live rate)
 *
 * Why this matters: during a long OpenCode prompt the decode TPS gauge
 * sits at zero (no decode tokens yet); this panel is what the user
 * watches so they know progress is happening.
 */
export function PrefillPanel() {
  const prefill = usePrimaryActivePrefill();
  const lastCompleted = useDashboardStore((s) => s.lastCompletedPrefill);
  const { data: historyPayload } = usePrefillHistory();

  // Smoothly tick the elapsed counter while active so the UI feels alive
  // even between chunk events (which arrive every ~2048 tokens of work).
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (!prefill.active) return;
    const handle = window.setInterval(() => setNow(performance.now()), 250);
    return () => window.clearInterval(handle);
  }, [prefill.active]);
  // Anchor the live elapsed so it doesn't reset when the request_id changes.
  const startRef = useRef<{ request_id: string; anchorMs: number; baseElapsed: number } | null>(
    null,
  );
  if (prefill.active) {
    if (
      !startRef.current ||
      startRef.current.request_id !== prefill.request_id
    ) {
      startRef.current = {
        request_id: prefill.request_id,
        anchorMs: now,
        baseElapsed: prefill.elapsed_s,
      };
    }
  } else if (startRef.current) {
    startRef.current = null;
  }
  const liveElapsed = prefill.active && startRef.current
    ? startRef.current.baseElapsed + (now - startRef.current.anchorMs) / 1000
    : prefill.active
      ? prefill.elapsed_s
      : 0;

  const historyMeanTokS = (() => {
    const rows = historyPayload?.history ?? [];
    const values = rows
      .map((r) => r.prefill_tok_s)
      .filter((v): v is number => typeof v === "number" && v > 0);
    if (values.length === 0) return null;
    return values.reduce((a, b) => a + b, 0) / values.length;
  })();

  if (prefill.active) {
    return <ActiveCard view={prefill} liveElapsed={liveElapsed} />;
  }

  return (
    <Card
      title="预填充（处理输入）"
      subtitle={
        lastCompleted
          ? `上次：${fmtNumber(lastCompleted.new_prefill_tokens ?? lastCompleted.tokens_total)} tokens · ${fmtSeconds(
              lastCompleted.elapsed_s,
            )} · ${fmtTokS(lastCompleted.prefill_tok_s)} tok/s`
          : historyMeanTokS != null
            ? `空闲 · 历史均值 ${fmtTokS(historyMeanTokS)} tok/s`
            : "空闲 · 暂无预填充样本"
      }
    >
      <div className="grid grid-cols-3 gap-3 text-xs">
        <IdleStat
          label="上次新增 token"
          value={fmtNumber(lastCompleted?.new_prefill_tokens ?? lastCompleted?.tokens_total)}
        />
        <IdleStat
          label="上次复用 token"
          value={fmtNumber(lastCompleted?.cached_tokens)}
          tone="cool"
        />
        <IdleStat
          label="上次预填充速度（token/s）"
          value={fmtTokS(lastCompleted?.prefill_tok_s)}
          tone="accent"
        />
      </div>
      <p className="text-xs text-[var(--text-muted)] mt-3 leading-relaxed">
        服务开始处理输入时，此处显示预填充进度、实时速度、预计剩余时间和已用时间。此阶段尚未开始生成输出，生成速度为零是正常现象。
      </p>
    </Card>
  );
}

function ActiveCard({
  view,
  liveElapsed,
}: {
  view: Extract<ReturnType<typeof usePrimaryActivePrefill>, { active: true }>;
  liveElapsed: number;
}) {
  const liveTokS =
    view.tokens_done > 0 && liveElapsed > 0
      ? view.tokens_done / liveElapsed
      : view.prefill_tok_s;
  const remaining = Math.max(0, view.tokens_total - view.tokens_done);
  const etaS =
    liveTokS && liveTokS > 0 && remaining > 0 ? remaining / liveTokS : null;
  const livePct = view.tokens_total > 0
    ? Math.min(100, (view.tokens_done / view.tokens_total) * 100)
    : 0;

  return (
    <Card
      title={
        <span className="flex items-center gap-2">
          <Loader2 className="size-4 text-[var(--accent-warm)] animate-spin" />
          <span>正在预填充</span>
        </span>
      }
      subtitle={
        <span>
          {fmtNumber(view.tokens_done)} / {fmtNumber(view.tokens_total)} tokens
          {view.session_id ? (
            <>
              {" · "}
              <span className="text-[var(--accent-cool)]">{truncateMiddle(view.session_id, 18)}</span>
            </>
          ) : null}
        </span>
      }
    >
      {/* Big progress bar — the main thing the user looks at */}
      <div className="relative h-6 w-full rounded-full bg-[var(--bg-elevated)] overflow-hidden border border-[var(--border-soft)]">
        <motion.div
          className="absolute inset-y-0 left-0"
          style={{ background: "var(--accent-warm)" }}
          initial={false}
          animate={{ width: `${livePct}%` }}
          transition={{ type: "spring", stiffness: 80, damping: 18, mass: 0.6 }}
        />
        <div className="absolute inset-0 grid place-items-center text-xs font-semibold tabular-nums text-[var(--text-primary)] mix-blend-difference">
          {livePct.toFixed(1)}%
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-xs">
        <Stat
          label="实时预填充速度（token/s）"
          value={fmtTokS(liveTokS)}
          tone="accent"
        />
        <Stat
          label="预计剩余时间"
          value={etaS != null ? fmtSeconds(etaS) : "计算中"}
          tone="warm"
        />
        <Stat label="已用时间" value={fmtSeconds(liveElapsed)} />
        <Stat
          label="缓存 / 总量"
          value={`${fmtNumber(view.cached_tokens)} / ${fmtNumber(view.tokens_total)}`}
          tone="cool"
        />
      </div>

      <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)] mt-3">
        请求 {truncateMiddle(view.request_id, 22)}
      </div>
    </Card>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "warm" | "accent" | "cool";
}) {
  const color =
    tone === "warm"
      ? "text-[var(--accent-warm)]"
      : tone === "accent"
        ? "text-[var(--accent)]"
        : tone === "cool"
          ? "text-[var(--accent-cool)]"
          : "text-[var(--text-primary)]";
  return (
    <div className="rounded-md border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-3 py-2">
      <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        {label}
      </div>
      <div className={`text-sm font-semibold tabular-nums ${color}`}>{value}</div>
    </div>
  );
}

function IdleStat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "accent" | "cool";
}) {
  const color =
    tone === "accent"
      ? "text-[var(--accent)]"
      : tone === "cool"
        ? "text-[var(--accent-cool)]"
        : "text-[var(--text-primary)]";
  return (
    <div className="rounded-md border border-dashed border-[var(--border-soft)] px-3 py-2">
      <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        {label}
      </div>
      <div className={`text-sm font-semibold tabular-nums ${color}`}>{value}</div>
    </div>
  );
}
