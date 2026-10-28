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
import { usePrefillHistory } from "../hooks/usePolling";
import { fmtSeconds } from "../lib/utils";

const BUCKETS = [
  { upper: 0.05, label: "<50ms" },
  { upper: 0.1, label: "50-100ms" },
  { upper: 0.25, label: "100-250ms" },
  { upper: 0.5, label: "250-500ms" },
  { upper: 1.0, label: "0.5-1s" },
  { upper: 2.0, label: "1-2s" },
  { upper: 5.0, label: "2-5s" },
  { upper: Infinity, label: ">5s" },
];

export function TTFTDistribution() {
  const { data } = usePrefillHistory();
  const rows = data?.history ?? [];
  const counts = BUCKETS.map((b) => ({ ...b, count: 0 }));
  rows.forEach((row) => {
    if (typeof row.ttft_s !== "number") return;
    const bucket = counts.find((b) => row.ttft_s! <= b.upper);
    if (bucket) bucket.count += 1;
  });
  const values = rows
    .map((r) => r.ttft_s)
    .filter((v): v is number => typeof v === "number")
    .sort((a, b) => a - b);
  const p50 = values[Math.floor(values.length * 0.5)] ?? null;
  const p95 = values[Math.floor(values.length * 0.95)] ?? null;
  const hasData = values.length > 0;

  return (
    <Card
      title="首 token 延迟分布（TTFT）"
      subtitle={
        hasData
          ? `p50 ${fmtSeconds(p50)} · p95 ${fmtSeconds(p95)} · n=${values.length}`
          : "暂无首 token 延迟样本"
      }
    >
      <div className="h-[200px]">
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={counts} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="label" stroke="rgba(200,210,220,0.6)" tick={{ fontSize: 10 }} />
              <YAxis stroke="rgba(200,210,220,0.6)" allowDecimals={false} />
              <Tooltip
                contentStyle={{
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border-soft)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
              <Bar dataKey="count" fill="rgba(155,118,233,0.85)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full grid place-items-center text-[var(--text-muted)] text-sm">
            完成请求后，将显示首 token 延迟分布。
          </div>
        )}
      </div>
    </Card>
  );
}
