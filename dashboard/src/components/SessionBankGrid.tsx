import { useMutation, useQueryClient } from "@tanstack/react-query";
import { HardDrive, Info, MemoryStick, Snowflake, TriangleAlert, X, Zap } from "lucide-react";
import type { ReactNode } from "react";
import { api } from "../lib/api";
import { fmtAge, followUpHits, ramOverview, ssdOverview } from "../lib/cacheOverview";
import { fmtBytes, fmtNumber, fmtSeconds, relativeTime, truncateMiddle } from "../lib/utils";
import { Card } from "./Card";
import { useDashboardStore } from "../state/store";
import type {
  ColdTierStats,
  MetricsLatest,
  SessionBank,
  SessionBankPrefix,
  SessionRow,
} from "../lib/types";

const RAM_EVICTION_NOTE =
  "缓存库超过容量预算或条目上限时，会优先移除闲置会话的内存快照；" +
  "超过空闲存活时间仍未访问的快照也会被移除。";
const SSD_EVICTION_NOTE =
  "写入将超过容量上限时，会优先删除最久未使用的快照。" +
  "SSD 快照没有存活时间限制，仅在需要空间时移除旧快照。";

export function SessionBankGrid() {
  const sessionBank = useDashboardStore((s) => s.sessionBank);
  const sessions = useDashboardStore((s) => s.sessions);
  const recent = useDashboardStore((s) => s.recent);
  const setSessionFilter = useDashboardStore((s) => s.setSessionFilter);
  const sessionFilter = useDashboardStore((s) => s.sessionFilter);
  const prefixes = sessionBank?.prefixes ?? [];
  const sessionByPrefix: Record<string, SessionRow> = (sessions?.sessions ?? []).reduce(
    (acc, row) => ((acc[row.session_id] = row), acc),
    {} as Record<string, SessionRow>,
  );

  const queryClient = useQueryClient();
  const clearMutation = useMutation({
    mutationFn: (sessionId: string) => api.postClearSession(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });

  return (
    <Card
      title="会话缓存库 · 缓存概览"
      subtitle={
        sessionBank?.last_miss_reason
          ? `RAM 与 SSD 缓存情况 · 最近未命中：${sessionBank.last_miss_reason}`
          : "RAM 与 SSD 缓存情况"
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <RamPanel bank={sessionBank} recent={recent} />
        <SsdPanel cold={sessionBank?.cold_tier} />
      </div>
      <div className="mt-4 text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        当前 RAM · 已用 {prefixes.length} / {sessionBank?.max_entries ?? "—"} 个缓存位
      </div>
      {prefixes.length > 0 ? (
        <div className="mt-2 grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-2">
          {prefixes.map((slot, idx) => (
            <SlotCard
              key={`${slot.session_id}:${slot.prefix_len}`}
              index={idx}
              slot={slot}
              session={sessionByPrefix[slot.session_id]}
              isFiltered={sessionFilter === slot.session_id}
              onClickSession={(sessionId) => setSessionFilter(sessionId)}
              onEvict={(sessionId) => clearMutation.mutate(sessionId)}
            />
          ))}
        </div>
      ) : (
        <p className="mt-2 text-xs text-[var(--text-muted)]">RAM 中暂无缓存。</p>
      )}
    </Card>
  );
}

function RamPanel({ bank, recent }: { bank: SessionBank | null; recent: MetricsLatest[] }) {
  const ram = ramOverview(bank);
  const hits = followUpHits(recent);
  const caption =
    ram.budgetBytes === null
      ? "未报告容量预算"
      : ram.configuredMaxBytes !== null && ram.configuredMaxBytes !== ram.budgetBytes
        ? `当前预算 ${fmtBytes(ram.budgetBytes)}（配置值 ${fmtBytes(ram.configuredMaxBytes)}）`
        : `预算 ${fmtBytes(ram.budgetBytes)}`;
  return (
    <TierPanel
      icon={<MemoryStick className="size-3.5 text-[var(--accent-warm)]" />}
      label="RAM · 热缓存"
      note={
        ram.idleTtlS !== null
          ? `${RAM_EVICTION_NOTE} 空闲存活时间：${fmtSeconds(ram.idleTtlS)}。`
          : RAM_EVICTION_NOTE
      }
      usedBytes={ram.usedBytes}
      caption={caption}
      fraction={ram.usedFraction}
    >
      <Stat label="快照数" value={fmtNumber(ram.snapshots)} />
      <Stat label="会话数" value={fmtNumber(ram.sessions)} />
      <Stat
        label="后续请求 RAM 命中"
        value={
          hits.ramFraction === null
            ? "—"
            : `${hits.ram} / ${hits.followUps} · ${(hits.ramFraction * 100).toFixed(0)}%`
        }
        title={
          `最近 ${recent.length} 个请求中，${hits.ram} 个后续请求从 RAM 找到前缀，` +
          `${hits.ssd} 个从 SSD 找到前缀，${hits.missed} 个未命中。首轮和已取消请求不计入。`
        }
      />
      <Stat
        label="SSD 命中 · 未命中"
        value={hits.followUps > 0 ? `${hits.ssd} · ${hits.missed}` : "—"}
        tone={hits.missed > 0 ? "warm" : undefined}
      />
    </TierPanel>
  );
}

function SsdPanel({ cold }: { cold: ColdTierStats | undefined }) {
  const ssd = ssdOverview(cold, Date.now() / 1000);
  if (!ssd.enabled) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border-soft)] bg-[var(--bg-elevated)] p-3 text-xs text-[var(--text-muted)]">
        <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest">
          <HardDrive className="size-3.5" /> SSD · 冷缓存
        </div>
        <p className="mt-2">SSD 缓存未开启：会话仅缓存在 RAM 中。</p>
      </div>
    );
  }
  const dedupe =
    ssd.logicalBytes > ssd.physicalBytes
      ? ` · 去重前 ${fmtBytes(ssd.logicalBytes)}`
      : "";
  return (
    <TierPanel
      icon={<HardDrive className="size-3.5 text-[var(--accent-cool)]" />}
      label="SSD · 冷缓存"
      note={SSD_EVICTION_NOTE}
      usedBytes={ssd.physicalBytes}
      caption={
        ssd.capBytes === null ? `磁盘占用${dedupe}` : `上限 ${fmtBytes(ssd.capBytes)}${dedupe}`
      }
      fraction={ssd.usedFraction}
    >
      <Stat label="快照数" value={fmtNumber(ssd.snapshots)} />
      <Stat
        label="会话数"
        value={
          ssd.sessions === null
            ? "—"
            : ssd.snapshotsWithoutSession > 0
              ? `${fmtNumber(ssd.sessions)} + ${fmtNumber(ssd.snapshotsWithoutSession)} 个无会话标识快照`
              : fmtNumber(ssd.sessions)
        }
        title={
          ssd.snapshotsWithoutSession > 0
            ? `${ssd.snapshotsWithoutSession} 个快照写入时没有会话 ID。`
            : undefined
        }
      />
      <Stat
        label="待写入"
        value={
          ssd.writePending
            ? `${fmtNumber(ssd.writerQueueDepth)} · ${fmtBytes(ssd.writerBacklogBytes)}`
            : "无"
        }
        tone={ssd.writePending ? "warm" : undefined}
        icon={ssd.writePending ? <TriangleAlert className="size-3" /> : undefined}
        title="已接受写入但尚未落盘的快照。在写入完成前，相应会话仅缓存在 RAM 中。"
      />
      <Stat
        label="恢复成功 · 失败"
        value={`${fmtNumber(ssd.restoreHits)} · ${fmtNumber(ssd.restoreFailures)}`}
        tone={ssd.restoreFailures > 0 ? "hot" : undefined}
      />
      <Stat label="最旧 · 最新" value={`${fmtAge(ssd.oldestAgeS)} · ${fmtAge(ssd.newestAgeS)}`} />
      <Stat
        label="从未恢复"
        value={
          ssd.neverRestored === null
            ? "—"
            : `${fmtNumber(ssd.neverRestored)} / ${fmtNumber(ssd.snapshots)}`
        }
      />
    </TierPanel>
  );
}

function TierPanel({
  icon,
  label,
  note,
  usedBytes,
  caption,
  fraction,
  children,
}: {
  icon: ReactNode;
  label: string;
  note: string;
  usedBytes: number;
  caption: string;
  fraction: number | null;
  children: ReactNode;
}) {
  const pct = fraction === null ? 0 : fraction * 100;
  const color =
    pct >= 95 ? "var(--accent-hot)" : pct >= 80 ? "var(--accent-warm)" : "var(--accent)";
  return (
    <div className="rounded-lg border border-[var(--border-soft)] bg-[var(--bg-elevated)] p-3">
      <div className="flex items-center justify-between">
        <span className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
          {icon}
          {label}
        </span>
        <span
          title={note}
          className="text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-help"
        >
          <Info className="size-3.5" />
        </span>
      </div>
      <div className="mt-2 flex flex-wrap items-baseline gap-x-2">
        <span className="text-2xl font-semibold tabular-nums text-[var(--text-primary)]">
          {fmtBytes(usedBytes)}
        </span>
        <span className="text-xs text-[var(--text-muted)]">{caption}</span>
      </div>
      <div className="mt-2 h-2 w-full rounded-full bg-[var(--bg-card)] overflow-hidden border border-[var(--border-soft)]">
        <div
          className="h-full transition-[width] duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">{children}</dl>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
  icon,
  title,
}: {
  label: string;
  value: string;
  tone?: "warm" | "hot";
  icon?: ReactNode;
  title?: string;
}) {
  const color =
    tone === "hot"
      ? "text-[var(--accent-hot)]"
      : tone === "warm"
        ? "text-[var(--accent-warm)]"
        : "text-[var(--text-primary)]";
  return (
    <div className="flex items-baseline justify-between gap-2" title={title}>
      <dt className="text-[var(--text-muted)] uppercase tracking-wider text-[9px]">{label}</dt>
      <dd className={`inline-flex items-center gap-1 tabular-nums ${color}`}>
        {icon}
        {value}
      </dd>
    </div>
  );
}

function SlotCard({
  index,
  slot,
  session,
  isFiltered,
  onClickSession,
  onEvict,
}: {
  index: number;
  slot: SessionBankPrefix;
  session: SessionRow | undefined;
  isFiltered: boolean;
  onClickSession: (sessionId: string) => void;
  onEvict: (sessionId: string) => void;
}) {
  const ageS = Date.now() / 1000 - slot.last_access_s;
  const inFlight = Boolean(session?.in_flight);
  const hot = ageS < 30;
  return (
    <button
      type="button"
      onClick={() => onClickSession(slot.session_id)}
      className={
        "group relative text-left rounded-lg border bg-[var(--bg-elevated)] p-2.5 transition-colors " +
        (isFiltered
          ? "border-[var(--accent)] shadow-[0_0_0_1px_var(--accent)]"
          : hot
            ? "border-[var(--accent)]/40 hover:border-[var(--accent)]"
            : "border-[var(--border-soft)] hover:border-[var(--text-muted)]")
      }
    >
      <div className="flex items-center justify-between pr-5">
        <span className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
          缓存位 {index + 1}
        </span>
        {inFlight ? (
          <span className="inline-flex items-center gap-1 text-[10px] text-[var(--accent)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
            正在处理
          </span>
        ) : hot ? (
          <Zap className="size-3 text-[var(--accent-warm)]" />
        ) : (
          <Snowflake className="size-3 text-[var(--accent-cool)]" />
        )}
      </div>
      <div className="text-xs font-mono text-[var(--text-primary)] mt-1 truncate">
        {truncateMiddle(slot.session_id, 24)}
      </div>
      <dl className="mt-1.5 grid grid-cols-2 gap-x-2 gap-y-0.5 text-[11px]">
        <Tag label="前缀长度" value={fmtNumber(slot.prefix_len)} />
        <Tag label="命中次数" value={fmtNumber(slot.hits)} />
        <Tag label="占用" value={fmtBytes(slot.nbytes)} />
        <Tag label="最近访问" value={relativeTime(slot.last_access_s)} />
      </dl>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onEvict(slot.session_id);
        }}
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 text-[var(--text-muted)] hover:text-[var(--accent-hot)] transition-opacity"
        title="移除此缓存项"
      >
        <X className="size-3.5" />
      </button>
    </button>
  );
}

function Tag({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-1">
      <dt className="text-[var(--text-muted)] uppercase tracking-wider text-[9px]">{label}</dt>
      <dd className="text-[var(--text-primary)] tabular-nums">{value}</dd>
    </div>
  );
}
