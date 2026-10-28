import { fmtNumber, fmtSeconds, fmtTokS } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function BottomStatsBar() {
  const latest = useDashboardStore((s) => s.latest);
  const lifetime = useDashboardStore((s) => s.lifetime);
  const liveTokS = useDashboardStore((s) => s.liveTokS);
  const completionTokens = latest?.completion_tokens ?? null;
  const ttftS = latest?.ttft_s ?? null;
  // Every other figure in this bar describes the last completed request, so
  // tok/s does too once decoding stops; the highlight marks a live reading.
  const isLive = typeof liveTokS === "number";
  const decodeTokS = isLive ? liveTokS : (latest?.decode_tok_s ?? null);
  const requestTokS = latest?.request_tok_s ?? null;
  const promptEvalS = latest?.prompt_eval_time_s ?? null;
  const decodeElapsedS = latest?.decode_elapsed_s ?? null;
  const totalRequests = lifetime?.requests_total ?? 0;
  return (
    <div className="px-4 lg:px-6 py-2 flex items-center justify-between gap-4 text-xs">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[var(--text-muted)] min-w-0">
        <Item label="输出 token" value={fmtNumber(completionTokens)} />
        <Item label="首 token 延迟" value={fmtSeconds(ttftS)} />
        <Item label="预填充耗时" value={fmtSeconds(promptEvalS)} />
        <Item label="生成耗时" value={fmtSeconds(decodeElapsedS)} />
        <Item
          label={isLive ? "实时 token/s" : "tok/s"}
          value={fmtTokS(decodeTokS)}
          highlight={isLive}
        />
        <Item label="请求平均 token/s" value={fmtTokS(requestTokS)} />
        <Item label="累计请求" value={fmtNumber(totalRequests)} />
      </div>
      <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)] hidden sm:block">
        MTPLX 实时监控
      </div>
    </div>
  );
}

function Item({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <span className="flex items-baseline gap-1.5 whitespace-nowrap">
      <span className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        {label}
      </span>
      <span
        className={
          "tabular-nums font-medium " +
          (highlight ? "text-[var(--accent)]" : "text-[var(--text-primary)]")
        }
      >
        {value}
      </span>
    </span>
  );
}
