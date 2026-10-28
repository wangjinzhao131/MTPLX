import { Volume2, VolumeX } from "lucide-react";
import { useDashboardStore } from "../state/store";

export function SoundToggle() {
  const enabled = useDashboardStore((s) => s.soundEnabled);
  const toggle = useDashboardStore((s) => s.toggleSound);
  return (
    <button
      onClick={toggle}
      title={enabled ? "关闭新纪录提示音（S）" : "开启新纪录提示音（S）"}
      className="text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)] inline-flex items-center"
    >
      {enabled ? <Volume2 className="size-4" /> : <VolumeX className="size-4" />}
    </button>
  );
}
