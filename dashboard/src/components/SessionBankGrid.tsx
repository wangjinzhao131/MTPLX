import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Snowflake, X, Zap } from "lucide-react";
import { api } from "../lib/api";
import { fmtBytes, fmtNumber, relativeTime, truncateMiddle } from "../lib/utils";
import { Card } from "./Card";
import { useDashboardStore } from "../state/store";
import type { SessionBankPrefix, SessionRow } from "../lib/types";

export function SessionBankGrid() {
  const sessionBank = useDashboardStore((s) => s.sessionBank);
  const sessions = useDashboardStore((s) => s.sessions);
  const setSessionFilter = useDashboardStore((s) => s.setSessionFilter);
  const sessionFilter = useDashboardStore((s) => s.sessionFilter);
  const max = sessionBank?.max_entries ?? 8;
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

  const slots = Array.from({ length: max }, (_, i) => prefixes[i] ?? null);
  const totalBytes = sessionBank?.total_nbytes ?? 0;

  return (
    <Card
      title="会话缓存库 · 已缓存前缀"
      subtitle={`${prefixes.length} / ${max} 个缓存位 · 共 ${fmtBytes(totalBytes)}${
        sessionBank?.last_miss_reason
          ? ` · 最近未命中：${sessionBank.last_miss_reason}`
          : ""
      }`}
    >
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {slots.map((slot, idx) => (
          <SlotCard
            key={idx}
            index={idx}
            slot={slot}
            session={slot ? sessionByPrefix[slot.session_id] : undefined}
            isFiltered={Boolean(slot && sessionFilter === slot.session_id)}
            onClickSession={(sessionId) => setSessionFilter(sessionId)}
            onEvict={(sessionId) => clearMutation.mutate(sessionId)}
          />
        ))}
      </div>
    </Card>
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
  slot: SessionBankPrefix | null;
  session: SessionRow | undefined;
  isFiltered: boolean;
  onClickSession: (sessionId: string) => void;
  onEvict: (sessionId: string) => void;
}) {
  if (!slot) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border-soft)] bg-[var(--bg-elevated)] aspect-square p-3 grid place-items-center text-[var(--text-muted)] text-xs">
        缓存位 {index + 1} · 空
      </div>
    );
  }
  const ageS = Date.now() / 1000 - slot.last_access_s;
  const inFlight = Boolean(session?.in_flight);
  const hot = ageS < 30;
  return (
    <button
      type="button"
      onClick={() => onClickSession(slot.session_id)}
      className={
        "group relative text-left rounded-lg border bg-[var(--bg-elevated)] p-3 transition-colors " +
        (isFiltered
          ? "border-[var(--accent)] shadow-[0_0_0_1px_var(--accent)]"
          : hot
            ? "border-[var(--accent)]/40 hover:border-[var(--accent)]"
            : "border-[var(--border-soft)] hover:border-[var(--text-muted)]")
      }
    >
      <div className="flex items-center justify-between">
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
      <dl className="mt-2 grid grid-cols-2 gap-x-2 gap-y-1 text-[11px]">
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
