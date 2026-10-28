import { Card } from "./Card";
import { fmtBytes } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function MemoryStackedBar() {
  const mem = useDashboardStore((s) => s.mem);
  const machine = useDashboardStore((s) => s.machine);
  const latest = useDashboardStore((s) => s.latest);

  const total = Number(machine?.unified_memory_bytes ?? 0);
  const active = Number(mem?.active_memory_bytes ?? 0);
  const cache = Number(mem?.cache_memory_bytes ?? 0);
  const peakLive = Number(mem?.peak_memory_bytes ?? 0);
  const peakRequest = Number(latest?.peak_memory_bytes ?? 0);
  const peak = Math.max(peakLive, peakRequest);

  const headroom = Math.max(0, total - active - cache);
  const denominator = total > 0 ? total : Math.max(active + cache + headroom, 1);

  const activePct = (active / denominator) * 100;
  const cachePct = (cache / denominator) * 100;
  const headroomPct = (headroom / denominator) * 100;
  const peakPct = total > 0 ? Math.min(100, (peak / total) * 100) : null;

  return (
    <Card
      title="MLX 内存占用"
      subtitle={
        total > 0
          ? `${fmtBytes(active + cache)} MLX 占用 · ${fmtBytes(headroom)} MLX 之外 · ${fmtBytes(total)} 总内存`
          : "MLX 实时内存快照"
      }
    >
      <div className="relative h-6 w-full rounded-full overflow-hidden border border-[var(--border-soft)] bg-[var(--bg-elevated)]">
        <div
          className="absolute inset-y-0 left-0 transition-[width] duration-500"
          style={{ width: `${activePct}%`, background: "var(--accent)" }}
        />
        <div
          className="absolute inset-y-0 transition-[width] duration-500"
          style={{
            left: `${activePct}%`,
            width: `${cachePct}%`,
            background: "var(--accent-cool)",
            opacity: 0.7,
          }}
        />
        <div
          className="absolute inset-y-0 transition-[width] duration-500"
          style={{
            left: `${activePct + cachePct}%`,
            width: `${headroomPct}%`,
            background: "rgba(255,255,255,0.06)",
          }}
        />
        {peakPct !== null && peakPct > 0 ? (
          <div
            className="absolute top-0 bottom-0 border-l-2 border-[var(--accent-warm)]"
            style={{ left: `${peakPct}%` }}
            title={`峰值 ${fmtBytes(peak)}`}
          />
        ) : null}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 text-xs">
        <Legend color="var(--accent)" label="正在使用" value={fmtBytes(active)} />
        <Legend color="var(--accent-cool)" label="缓存" value={fmtBytes(cache)} />
        <Legend color="var(--accent-warm)" label="峰值" value={fmtBytes(peak)} />
        <Legend color="rgba(255,255,255,0.15)" label="MLX 之外" value={fmtBytes(headroom)} />
      </div>
      <p className="text-xs text-[var(--text-muted)] mt-2">
        “MLX 之外”包含系统、其他应用及尚未使用的内存，不等于系统可用内存。
      </p>
      {!mem?.ok ? (
        <p className="text-xs text-[var(--text-muted)] mt-2">
          无法读取 MLX 内存： {mem?.error ?? "未知错误"}
        </p>
      ) : null}
    </Card>
  );
}

function Legend({
  color,
  label,
  value,
}: {
  color: string;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-2.5 h-2.5 rounded-sm" style={{ background: color }} />
      <span className="text-[var(--text-muted)] uppercase tracking-wider text-[10px]">{label}</span>
      <span className="ml-auto text-[var(--text-primary)] tabular-nums">{value}</span>
    </div>
  );
}
