import { BigNumber, Card } from "./Card";
import { HardwareBanner } from "./HardwareBanner";
import { MemoryStackedBar } from "./MemoryStackedBar";
import { fmtBytes } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function MemoryTab() {
  const mem = useDashboardStore((s) => s.mem);
  const latest = useDashboardStore((s) => s.latest);
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12">
        <HardwareBanner />
      </div>
      <div className="col-span-12">
        <MemoryStackedBar />
      </div>
      <div className="col-span-12 sm:col-span-6 lg:col-span-4">
        <Card title="活跃内存" subtitle="MLX 正在使用的内存">
          <BigNumber
            value={fmtBytes(mem?.active_memory_bytes ?? null)}
            tone="accent"
            caption="MLX 实时读数"
          />
        </Card>
      </div>
      <div className="col-span-12 sm:col-span-6 lg:col-span-4">
        <Card title="分配器缓存" subtitle="MLX 分配器保留的缓冲区">
          <BigNumber
            value={fmtBytes(mem?.cache_memory_bytes ?? null)}
            tone="cool"
            caption="可复用的内存缓冲区"
          />
        </Card>
      </div>
      <div className="col-span-12 sm:col-span-6 lg:col-span-4">
        <Card title="峰值内存" subtitle="当前进程启动以来的最高值">
          <BigNumber
            value={fmtBytes(
              Math.max(
                Number(mem?.peak_memory_bytes ?? 0),
                Number(latest?.peak_memory_bytes ?? 0),
              ) || null,
            )}
            tone="warm"
            caption="包含最近请求的峰值"
          />
        </Card>
      </div>
    </div>
  );
}
