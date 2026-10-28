import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "./Card";
import { useDashboardStore } from "../state/store";

// Mirrors `CacheMissReason` enum (mtplx/session_bank.py:29-39).
const REASON_TOOLTIPS: Record<string, string> = {
  POLICY_MISMATCH:
    "请求间的采样或深度参数发生变化，缓存策略不再一致。",
  TEMPLATE_MISMATCH: "聊天模板发生变化（系统提示、角色标记），无法复用缓存。",
  MODEL_MISMATCH: "当前模型与创建缓存时的模型不同。",
  SNAPSHOT_DESYNC: "引擎状态与缓存快照不一致。",
  EVICTED: "按最近最少使用策略回收缓存，为新前缀腾出空间。",
  SESSION_BUSY: "另一个请求正在使用此会话的缓存。",
  NEW_SESSION: "该会话的首个请求，没有可复用的旧前缀。",
};

export function EvictionReasonHistogram() {
  const sessionBank = useDashboardStore((s) => s.sessionBank);
  const log = sessionBank?.eviction_log ?? [];
  const counts: Record<string, number> = {};
  for (const entry of log) {
    counts[entry.reason] = (counts[entry.reason] ?? 0) + 1;
  }
  const rows = Object.entries(counts)
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count);

  return (
    <Card
      title="缓存回收原因 · 最近 16 次"
      subtitle={
        sessionBank?.last_miss_reason
          ? `最近：${sessionBank.last_miss_reason}`
          : "尚无缓存回收"
      }
    >
      <div className="h-[220px]">
        {rows.length === 0 ? (
          <div className="h-full grid place-items-center text-[var(--text-muted)] text-sm">
            会话缓存稳定 · 未发生回收
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="reason"
                stroke="rgba(200,210,220,0.6)"
                tick={{ fontSize: 10 }}
                interval={0}
              />
              <YAxis stroke="rgba(200,210,220,0.6)" allowDecimals={false} />
              <Tooltip
                contentStyle={{
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border-soft)",
                  borderRadius: 8,
                  fontSize: 12,
                  maxWidth: 320,
                }}
                labelFormatter={(label) => (
                  <span className="text-[var(--text-primary)] font-semibold">
                    {String(label)}
                  </span>
                )}
                // Recharts' Formatter generics fight a literal "count" name;
                // returning unknown[] sidesteps the constraint and the runtime
                // shape is still a [label, name] tuple.
                formatter={((value: unknown, _name: unknown, item: { payload?: { reason?: string } }) => {
                  const reason = String(item?.payload?.reason ?? "");
                  const tooltip = REASON_TOOLTIPS[reason] ?? "缓存回收原因。";
                  return [`${value} · ${tooltip}`, "次数"];
                }) as never}
              />
              <Bar dataKey="count" fill="rgba(240,180,41,0.85)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </Card>
  );
}
