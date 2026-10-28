import { BigNumber, Card } from "./Card";
import { fmtNumber, fmtSeconds } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function TokensServedTile() {
  const lifetime = useDashboardStore((s) => s.lifetime);
  if (!lifetime) {
    return (
      <Card title="本次服务累计 token">
        <BigNumber value="—" caption="等待首个请求" />
      </Card>
    );
  }
  return (
    <Card title="本次服务累计 token">
      <BigNumber
        value={fmtNumber(lifetime.tokens_total)}
        unit="token"
        tone="accent"
        caption={
          <div className="space-y-1">
            <div>
              {fmtNumber(lifetime.requests_total)} 个请求 · 已运行 {fmtSeconds(lifetime.uptime_s)}
            </div>
            <div className="text-[var(--text-muted)]">
              输入：{fmtNumber(lifetime.prompt_tokens_total)} ·{" "}
              输出：{fmtNumber(lifetime.completion_tokens_total)} ·{" "}
              复用：{fmtNumber(lifetime.cached_tokens_total)}
            </div>
            {lifetime.cancelled_total > 0 ? (
              <div className="text-[var(--accent-warm)] text-xs">
                {fmtNumber(lifetime.cancelled_total)} 个已取消
              </div>
            ) : null}
          </div>
        }
      />
    </Card>
  );
}
