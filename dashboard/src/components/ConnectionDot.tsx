import { useDashboardStore } from "../state/store";
import { cn } from "../lib/utils";

const COLORS: Record<string, string> = {
  open: "bg-emerald-400 shadow-[0_0_12px_rgb(74,222,128,0.6)]",
  connecting: "bg-amber-400 animate-pulse",
  reconnecting: "bg-amber-500 animate-pulse",
  failed: "bg-rose-500",
  unauthorized: "bg-rose-500",
  idle: "bg-slate-500",
};

const LABELS: Record<string, string> = {
  open: "实时连接",
  connecting: "连接中",
  reconnecting: "重新连接中",
  failed: "已离线",
  unauthorized: "需要登录",
  idle: "空闲",
};

export function ConnectionDot() {
  const state = useDashboardStore((s) => s.connection);
  return (
    <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
      <span className={cn("w-2 h-2 rounded-full", COLORS[state] ?? COLORS.idle)} />
      <span className="hidden sm:inline">{LABELS[state] ?? state}</span>
    </div>
  );
}
