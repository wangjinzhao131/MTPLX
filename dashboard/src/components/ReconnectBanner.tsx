import { useDashboardStore } from "../state/store";

export function ReconnectBanner() {
  const connection = useDashboardStore((s) => s.connection);
  if (
    connection === "open" ||
    connection === "idle" ||
    connection === "connecting" ||
    // Sign-in has its own surface; this banner is for a server that is gone.
    connection === "unauthorized"
  ) {
    return null;
  }
  const message =
    connection === "failed"
      ? "与 MTPLX 的连接已断开，面板将继续尝试重连。"
      : "正在重新连接 MTPLX…";
  return (
    <div className="bg-amber-500/15 text-amber-300 text-xs px-4 py-1.5 text-center border-b border-amber-500/30">
      {message}
    </div>
  );
}
