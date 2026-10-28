import { BigNumber, Card } from "./Card";
import { fmtNumber } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function CumulativeCachedTokensTile() {
  const lifetime = useDashboardStore((s) => s.lifetime);
  const cached = lifetime?.cached_tokens_total ?? 0;
  const prompt = lifetime?.prompt_tokens_total ?? 0;
  const ratio = prompt > 0 ? (cached / prompt) * 100 : 0;
  return (
    <Card title="累计复用 token" subtitle="全部请求的缓存 token / 输入 token">
      <BigNumber
        value={fmtNumber(cached)}
        unit="token"
        tone="accent"
        caption={`${ratio.toFixed(1)}% / 共 ${fmtNumber(prompt)} 个输入 token`}
      />
    </Card>
  );
}

export function HitRateGauge() {
  const recent = useDashboardStore((s) => s.recent);
  const window = recent.slice(-32);
  const hits = window.filter((r) => r.session_cache_hit).length;
  const ratio = window.length > 0 ? (hits / window.length) * 100 : 0;
  const tone = ratio >= 70 ? "accent" : ratio >= 40 ? "warm" : "hot";
  return (
    <Card title="会话缓存命中率" subtitle={`最近 ${window.length} 个请求`}>
      <BigNumber
        value={`${ratio.toFixed(0)}%`}
        unit="命中"
        tone={tone}
        caption={`${hits} 次命中 / ${window.length} 个请求`}
      />
    </Card>
  );
}

export function ContextUtilizationBar() {
  const latest = useDashboardStore((s) => s.latest);
  const contextWindow = useDashboardStore((s) => s.contextWindow);
  const len = latest?.context_len ?? 0;
  const pct = contextWindow ? Math.min(100, (len / contextWindow) * 100) : 0;
  const band =
    pct >= 95 ? "hot" : pct >= 75 ? "warm" : pct >= 50 ? "cool" : "accent";
  const color =
    band === "hot"
      ? "var(--accent-hot)"
      : band === "warm"
        ? "var(--accent-warm)"
        : band === "cool"
          ? "var(--accent-cool)"
          : "var(--accent)";
  return (
    <Card title="上下文窗口使用率" subtitle={`${fmtNumber(len)} / ${fmtNumber(contextWindow ?? 0)} tokens`}>
      <div className="h-4 w-full rounded-full bg-[var(--bg-elevated)] overflow-hidden border border-[var(--border-soft)]">
        <div
          className="h-full transition-[width] duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <div className="flex justify-between mt-2 text-xs text-[var(--text-muted)] tabular-nums">
        <span>0</span>
        <span className="text-[var(--text-primary)] font-semibold">{pct.toFixed(0)}%</span>
        <span>{fmtNumber(contextWindow ?? 0)}</span>
      </div>
    </Card>
  );
}
