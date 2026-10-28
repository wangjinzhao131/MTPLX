import { Card } from "./Card";
import { FanRingPanel } from "./FanRing";
import { useDashboardStore } from "../state/store";
import { relativeTime } from "../lib/utils";

export function ThermalTab() {
  const thermal = useDashboardStore((s) => s.thermal);
  const whenS = useDashboardStore((s) => s.thermalWhenS);
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 lg:col-span-7">
        <FanRingPanel />
      </div>
      <div className="col-span-12 lg:col-span-5">
        <Card title="温度与风扇快照" subtitle={whenS ? relativeTime(whenS) : "尚无采样"}>
          {thermal ? (
            <dl className="text-sm space-y-1">
              <Field label="状态" value={thermal.ok ? "正常" : "不可用"} />
              <Field label="最低转速（RPM）" value={String(thermal.min_rpm ?? "—")} />
              <Field label="最高转速（RPM）" value={String(thermal.max_rpm ?? "—")} />
              <Field label="风扇数量" value={String(thermal.fans?.length ?? 0)} />
            </dl>
          ) : (
            <p className="text-sm text-[var(--text-muted)]">
              温度与风扇采样默认关闭。启动 MTPLX 时添加 <code>--enable-thermal-poll</code> 可开启。
            </p>
          )}
        </Card>
      </div>
      <div className="col-span-12">
        <Card
          title="GPU 频率 · 尚未接入"
          subtitle="ThermalForge 不提供 GPU 频率，后续需接入 powermetrics"
        >
          <p className="text-sm text-[var(--text-muted)] leading-relaxed">
            ThermalForge 的 <code>status</code> 接口提供风扇转速和模式，但不提供 GPU 频率或温度压力。接入 <code>powermetrics</code> 前，此处不显示估算值。
          </p>
        </Card>
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <dt className="text-[var(--text-muted)]">{label}</dt>
      <dd className="text-[var(--text-primary)] tabular-nums">{value}</dd>
    </div>
  );
}
