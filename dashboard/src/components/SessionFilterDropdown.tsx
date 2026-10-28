import { useDashboardStore, useFilteredSessionIds } from "../state/store";
import { truncateMiddle } from "../lib/utils";

export function SessionFilterDropdown() {
  const ids = useFilteredSessionIds();
  const value = useDashboardStore((s) => s.sessionFilter) ?? "";
  const setValue = useDashboardStore((s) => s.setSessionFilter);
  return (
    <label className="hidden md:flex items-center gap-2 text-xs text-[var(--text-muted)]">
      <span>会话</span>
      <select
        value={value}
        onChange={(e) => setValue(e.target.value || null)}
        className="bg-[var(--bg-card)] border border-[var(--border-soft)] rounded px-2 py-1 text-xs text-[var(--text-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
      >
        <option value="">全部会话</option>
        {ids.map((id) => (
          <option key={id} value={id}>
            {truncateMiddle(id, 28)}
          </option>
        ))}
      </select>
    </label>
  );
}
