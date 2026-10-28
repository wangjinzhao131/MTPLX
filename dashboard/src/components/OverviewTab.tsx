import { BigNumber, Card } from "./Card";
import { PrefillPanel } from "./PrefillPanel";
import { TPSGauge } from "./TPSGauge";
import { TPSTimeSeries } from "./TPSTimeSeries";
import { TokensServedTile } from "./TokensServedTile";
import { fmtNumber, fmtSeconds, fmtTokS } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function OverviewTab() {
  const latest = useDashboardStore((s) => s.latest);
  const inFlight = useDashboardStore((s) => s.inFlight);
  const sessionBank = useDashboardStore((s) => s.sessionBank);
  const contextWindow = useDashboardStore((s) => s.contextWindow);

  const contextLen = latest?.context_len ?? 0;
  const contextPct = contextWindow ? Math.min(100, (contextLen / contextWindow) * 100) : 0;

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 lg:col-span-5">
        <TPSGauge />
      </div>
      <div className="col-span-12 lg:col-span-7">
        <TPSTimeSeries />
      </div>

      {/* Prefill panel: hero status while the model is chewing the prompt
       *  (decode TPS is meaningless during this window). Full-width so the
       *  progress bar is impossible to miss. */}
      <div className="col-span-12">
        <PrefillPanel />
      </div>

      <div className="col-span-12 lg:col-span-4">
        <TokensServedTile />
      </div>

      <div className="col-span-6 lg:col-span-4">
        <Card title="正在处理">
          <BigNumber
            value={fmtNumber(inFlight.length)}
            unit="个请求"
            tone={inFlight.length > 0 ? "accent" : "default"}
            caption={
              inFlight.length === 0
                ? "空闲 · 等待下一个请求"
                : `${inFlight.length} 个处理中 · 最久 ${fmtSeconds(
                    Math.max(...inFlight.map((h) => h.age_s)),
                  )}`
            }
          />
        </Card>
      </div>

      <div className="col-span-6 lg:col-span-4">
        <Card
          title="缓存与上下文"
          subtitle={
            sessionBank
              ? `${sessionBank.prefixes?.length ?? 0} / ${sessionBank.max_entries} 个缓存位`
              : "—"
          }
        >
          <BigNumber
            value={`${contextPct.toFixed(0)}%`}
            unit="上下文已用"
            tone={contextPct >= 75 ? "warm" : contextPct >= 95 ? "hot" : "cool"}
            caption={`${fmtNumber(contextLen)} / ${fmtNumber(contextWindow)} tokens`}
          />
        </Card>
      </div>

      <div className="col-span-12 lg:col-span-6">
        <Card title="最近请求" subtitle="来自最新请求指标">
          <div className="grid grid-cols-2 gap-3 text-sm">
            <Field label="生成速度（token/s）" value={fmtTokS(latest?.decode_tok_s)} highlight />
            <Field label="首 token 延迟" value={fmtSeconds(latest?.ttft_s)} />
            <Field label="预填充耗时" value={fmtSeconds(latest?.prompt_eval_time_s)} />
            <Field label="生成耗时" value={fmtSeconds(latest?.decode_elapsed_s)} />
            <Field
              label="预填充速度（token/s）"
              value={fmtTokS(latest?.prefill_tok_s)}
            />
            <Field
              label="已缓存"
              value={`${fmtNumber(latest?.cached_tokens)} / ${fmtNumber(latest?.prompt_tokens)}`}
            />
          </div>
        </Card>
      </div>

      <div className="col-span-12 lg:col-span-6">
        <Card title="会话" subtitle="来自最新指标快照">
          <div className="grid grid-cols-2 gap-3 text-sm">
            <Field
              label="会话 ID"
              value={latest?.session_id ? latest.session_id : "—"}
            />
            <Field
              label="缓存命中"
              value={latest?.session_cache_hit ? "是" : "否"}
              highlight={Boolean(latest?.session_cache_hit)}
            />
            <Field
              label="恢复方式"
              value={latest?.session_restore_mode ?? "—"}
            />
            <Field
              label="未命中原因"
              value={latest?.cache_miss_reason ?? "—"}
            />
            <Field label="MTP 深度" value={fmtNumber(latest?.mtp_depth)} />
            <Field label="验证次数" value={fmtNumber(latest?.verify_calls)} />
          </div>
        </Card>
      </div>
    </div>
  );
}

function Field({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div className="bg-[var(--bg-elevated)] rounded-md px-3 py-2 border border-[var(--border-soft)]">
      <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        {label}
      </div>
      <div
        className={
          "text-sm font-semibold tabular-nums " +
          (highlight ? "text-[var(--accent)]" : "text-[var(--text-primary)]")
        }
      >
        {value}
      </div>
    </div>
  );
}
