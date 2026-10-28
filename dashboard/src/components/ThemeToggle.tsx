import { Palette } from "lucide-react";
import { useDashboardStore } from "../state/store";

export function ThemeToggle() {
  const theme = useDashboardStore((s) => s.theme);
  const cycleTheme = useDashboardStore((s) => s.cycleTheme);
  return (
    <button
      onClick={cycleTheme}
      title={`主题：${theme}（按 T 切换）`}
      className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)]"
    >
      <Palette className="size-4" />
      <span className="hidden lg:inline">{{ hippo: "河马", river: "河流", light: "明亮", mono: "单色" }[theme]}</span>
    </button>
  );
}
