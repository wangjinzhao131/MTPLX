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
import { fmtSeconds } from "../lib/utils";
import { useDashboardStore } from "../state/store";

type Slice = { key: string; label: string; color: string; description: string };

const SLICES: Slice[] = [
  { key: "verify_forward_time_s", label: "验证前向计算", color: "rgba(0,214,143,0.85)", description: "目标模型验证图的前向计算" },
  { key: "verify_logits_eval_time_s", label: "logits 求值", color: "rgba(79,182,243,0.85)", description: "对 MTP 草稿 token 进行 logits 求值" },
  { key: "verify_hidden_eval_time_s", label: "隐藏状态求值", color: "rgba(155,118,233,0.85)", description: "为后续缓存写入计算隐藏状态" },
  { key: "verify_target_distribution_time_s", label: "目标概率分布", color: "rgba(245,158,11,0.85)", description: "目标概率分布计算（概率比）" },
  { key: "verify_eval_unattributed_time_s", label: "未归类", color: "rgba(244,114,182,0.75)", description: "未归类的求值耗时，理想情况下接近零" },
  { key: "accept_time_s", label: "接受采样", color: "rgba(0,214,143,0.55)", description: "接受采样与残差修正" },
  { key: "repair_time_s", label: "修复", color: "rgba(239,68,68,0.85)", description: "拒绝后的修复计算（为 0 时可能延迟执行）" },
  { key: "snapshot_time_s", label: "快照", color: "rgba(200,210,220,0.45)", description: "缓存快照与恢复" },
  { key: "capture_commit_time_s", label: "捕获与提交", color: "rgba(0,214,143,0.35)", description: "捕获并提交验证状态的开销" },
  { key: "rollback_time_s", label: "回滚", color: "rgba(240,88,106,0.55)", description: "拒绝后的状态回滚" },
];

export function VerifyWaterfall() {
  const latest = useDashboardStore((s) => s.latest);

  const total = Number(latest?.verify_time_s ?? 0);
  const rows = SLICES.map((slice) => {
    const seconds = Number((latest as Record<string, unknown> | null)?.[slice.key] ?? 0) || 0;
    return {
      ...slice,
      seconds,
      pct: total > 0 ? (seconds / total) * 100 : 0,
    };
  });
  const hasData = rows.some((r) => r.seconds > 0);

  return (
    <Card
      title="验证周期耗时分解"
      subtitle={
        latest
          ? `验证总计 ${fmtSeconds(total)} · 目标模型前向 ${fmtSeconds(
              latest?.target_forward_time_s,
            )} · 草稿 ${fmtSeconds(latest?.draft_time_s)}`
          : "尚无完成的验证周期"
      }
    >
      <div className="h-[280px]">
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={rows}
              margin={{ top: 4, right: 30, left: 110, bottom: 0 }}
            >
              <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
              <XAxis
                type="number"
                stroke="rgba(200,210,220,0.6)"
                tickFormatter={(v) => `${(v * 1000).toFixed(0)}ms`}
              />
              <YAxis
                type="category"
                dataKey="label"
                stroke="rgba(200,210,220,0.7)"
                width={100}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border-soft)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: "var(--text-muted)" }}
                formatter={(value, _name, item) => {
                  const slice = SLICES.find((s) => s.label === item?.payload?.label);
                  if (typeof value !== "number") return [value, slice?.label ?? "—"];
                  return [
                    `${fmtSeconds(value)} · ${item?.payload?.pct?.toFixed(1) ?? "—"}%`,
                    slice?.description ?? slice?.label ?? "—",
                  ];
                }}
              />
              <Bar dataKey="seconds" radius={[0, 6, 6, 0]}>
                {rows.map((entry) => (
                  <Bar key={entry.key} dataKey="seconds" fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full grid place-items-center text-[var(--text-muted)] text-sm">
            完成一次生成后，将显示验证耗时分解。
          </div>
        )}
      </div>
    </Card>
  );
}
