import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Copy, Trash } from "lucide-react";
import { api } from "../lib/api";
import type { MutableSettings } from "../lib/types";
import { Card } from "./Card";
import { useDashboardStore } from "../state/store";

type MutableKey = keyof MutableSettings;

const MUTABLE_SETTINGS_KEYS: MutableKey[] = [
  "depth",
  "temperature",
  "top_p",
  "top_k",
  "presence_penalty",
  "max_response_tokens",
  "stream_interval",
  "enable_thinking",
  "reasoning_parser",
];

const DEBOUNCE_MS = 250;
const CHANGED_ON_SERVER_HINT_MS = 6000;

export function ControlsSidebar() {
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 lg:col-span-7">
        <DefaultsCard />
      </div>
      <div className="col-span-12 lg:col-span-5">
        <RestartRequiredCard />
        <div className="mt-4">
          <AdminActions />
        </div>
      </div>
    </div>
  );
}

function DefaultsCard() {
  const settings = useDashboardStore((s) => s.settings);
  // `draft` is what the controls show. Every key the user is not editing
  // follows the server on each snapshot; only keys in `dirty` (edited here,
  // not yet applied) are ever sent. Posting a full diff against the live
  // settings used to write a stale draft back over values changed from the
  // native app, the CLI or another tab.
  const [draft, setDraft] = useState<MutableSettings | null>(settings);
  const [dirty, setDirty] = useState<ReadonlySet<MutableKey>>(() => new Set());
  const [changedOnServer, setChangedOnServer] = useState<MutableKey[]>([]);
  const draftRef = useRef(draft);
  draftRef.current = draft;
  const mergeSettings = useDashboardStore((s) => s.mergeSettings);
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: (payload: Partial<MutableSettings>) => api.postSettings(payload),
    onSuccess: (res) => {
      // The response carries the server's settings after the write (with
      // any clamping); adopt them now rather than waiting for the next
      // snapshot, so the controls never snap back to the pre-write values.
      const written: Partial<MutableSettings> = {};
      MUTABLE_SETTINGS_KEYS.forEach((key) => {
        if (key in res) (written as Record<string, unknown>)[key] = res[key];
      });
      mergeSettings(written);
      queryClient.invalidateQueries({ queryKey: ["snapshot"] });
    },
  });
  const [lastApplied, setLastApplied] = useState<Partial<MutableSettings> | null>(null);

  useEffect(() => {
    if (!settings) return;
    const shown = draftRef.current;
    if (!shown) {
      setDraft(settings);
      return;
    }
    // A key the user is not editing whose server value differs from what
    // the controls show moved underneath us: follow the server and say so.
    const moved = MUTABLE_SETTINGS_KEYS.filter(
      (key) => !dirty.has(key) && shown[key] !== settings[key],
    );
    if (moved.length === 0) return;
    setDraft((prev) => {
      if (!prev) return settings;
      const next = { ...prev };
      moved.forEach((key) => {
        (next as Record<string, unknown>)[key] = settings[key];
      });
      return next;
    });
    setChangedOnServer(moved);
  }, [settings, dirty]);

  useEffect(() => {
    if (changedOnServer.length === 0) return;
    const handle = window.setTimeout(() => setChangedOnServer([]), CHANGED_ON_SERVER_HINT_MS);
    return () => window.clearTimeout(handle);
  }, [changedOnServer]);

  const edit = <K extends MutableKey>(key: K, value: MutableSettings[K]) => {
    setDraft((prev) => (prev ? { ...prev, [key]: value } : prev));
    setDirty((prev) => {
      if (prev.has(key)) return prev;
      const next = new Set(prev);
      next.add(key);
      return next;
    });
  };

  useEffect(() => {
    if (!draft || !settings || dirty.size === 0) return;
    // Send only the keys the user moved, and only the mutable ones: the
    // draft carries the full settings payload, whose informational keys
    // would trip the server's all-or-nothing unknown_settings 400.
    const payload: Partial<MutableSettings> = {};
    dirty.forEach((key) => {
      if (draft[key] !== settings[key]) {
        (payload as Record<string, unknown>)[key] = draft[key];
      }
    });
    const keys = Object.keys(payload) as MutableKey[];
    if (keys.length === 0) {
      // Every edit landed back on the server's value; nothing is pending.
      setDirty(new Set());
      return;
    }
    const handle = window.setTimeout(() => {
      mutation.mutate(payload, {
        onSuccess: (res) => {
          setLastApplied(res.applied ?? payload);
          // A key stays dirty if the user moved it again while the request
          // was in flight; the next debounce sends the newer value.
          setDirty((prev) => {
            const next = new Set(prev);
            keys.forEach((key) => {
              if (draftRef.current?.[key] === payload[key]) next.delete(key);
            });
            return next;
          });
        },
      });
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft]);

  if (!draft) {
    return (
      <Card title="默认参数" subtitle="正在加载服务设置…">
        <div className="text-sm text-[var(--text-muted)]">
          收到首个服务快照后将显示设置。
        </div>
      </Card>
    );
  }

  return (
    <Card
      title="默认参数"
      subtitle="每次聊天请求使用的服务端默认参数"
      action={
        mutation.isPending ? (
          <span className="text-xs text-[var(--text-muted)]">正在应用…</span>
        ) : lastApplied ? (
          <span className="text-xs text-[var(--text-muted)]">
            已应用 · {Object.keys(lastApplied).join(", ")}
          </span>
        ) : undefined
      }
    >
      <div className="space-y-4">
        <NumberField
          label="推测深度"
          value={draft.depth}
          min={0}
          max={5}
          onChange={(v) => edit("depth", v)}
        />
        <NumberField
          label="采样温度"
          value={draft.temperature}
          min={0}
          max={2}
          step={0.05}
          onChange={(v) => edit("temperature", v)}
        />
        <NumberField
          label="核采样（top_p）"
          value={draft.top_p}
          min={0}
          max={1}
          step={0.01}
          onChange={(v) => edit("top_p", v)}
        />
        <NumberField
          label="候选数（top_k）"
          value={draft.top_k}
          min={0}
          max={2000}
          step={1}
          onChange={(v) => edit("top_k", v)}
        />
        <NumberField
          label="重复惩罚"
          value={draft.presence_penalty ?? 0}
          min={0}
          max={2}
          step={0.05}
          onChange={(v) => edit("presence_penalty", v)}
          description="0 不施加惩罚（适合编程）；0.5–1.5 可减少重复。"
        />
        <NumberField
          label="流式输出间隔"
          value={draft.stream_interval}
          min={1}
          max={32}
          step={1}
          onChange={(v) => edit("stream_interval", v)}
        />
        <ToggleField
          label="开启思考"
          value={draft.enable_thinking}
          onChange={(v) => edit("enable_thinking", v)}
          description="开启后，请求默认包含思考内容。"
        />
        <SelectField
          label="思考内容解析器"
          value={draft.reasoning_parser}
          options={["qwen3", "poolside_v1", "none"]}
          onChange={(v) => edit("reasoning_parser", v)}
        />
        {changedOnServer.length > 0 ? (
          <div className="text-xs text-[var(--accent-warm)]">
            服务端已更新：{changedOnServer.join(", ")}
          </div>
        ) : null}
        {mutation.isError ? (
          <div className="text-xs text-[var(--accent-hot)]">
            {String((mutation.error as Error).message)}
          </div>
        ) : null}
      </div>
    </Card>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  description,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (next: number) => void;
  description?: string;
}) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between text-xs text-[var(--text-muted)]">
        <span>{label}</span>
        <span className="tabular-nums text-[var(--text-primary)]">{Number(value).toFixed(step < 1 ? 2 : 0)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full mt-1 accent-[var(--accent)]"
      />
      {description ? (
        <div className="mt-0.5 text-xs text-[var(--text-muted)]">{description}</div>
      ) : null}
    </label>
  );
}

function ToggleField({
  label,
  value,
  onChange,
  description,
}: {
  label: string;
  value: boolean;
  onChange: (next: boolean) => void;
  description?: string;
}) {
  return (
    <label className="flex items-start justify-between gap-3">
      <div>
        <div className="text-sm text-[var(--text-primary)]">{label}</div>
        {description ? (
          <div className="text-xs text-[var(--text-muted)]">{description}</div>
        ) : null}
      </div>
      <button
        type="button"
        onClick={() => onChange(!value)}
        className={`h-5 w-9 rounded-full transition-colors relative shrink-0 ${
          value ? "bg-[var(--accent)]" : "bg-[var(--border-soft)]"
        }`}
        aria-pressed={value}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
            value ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </button>
    </label>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (next: string) => void;
}) {
  return (
    <label className="block">
      <div className="text-xs text-[var(--text-muted)]">{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full bg-[var(--bg-elevated)] border border-[var(--border-soft)] rounded px-2 py-1.5 text-sm text-[var(--text-primary)]"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}

function RestartRequiredCard() {
  const modelId = useDashboardStore((s) => s.modelId);
  const profileName = useDashboardStore((s) => s.profileName);
  const restartCommand = `mtplx serve --model ${modelId ?? "<model>"} --profile ${profileName ?? "<profile>"} --port 8000`;
  const copy = () => {
    if (typeof navigator !== "undefined") {
      navigator.clipboard?.writeText(restartCommand);
    }
  };
  return (
    <Card
      title="需要重启的设置"
      subtitle="配置方案、模型、MTP、地址和端口只能在启动时修改"
    >
      <p className="text-xs text-[var(--text-muted)] mb-3">
        这些参数需要重新加载模型后生效，不能通过在线设置修改。可复制下方命令，在需要时手动重启服务。
      </p>
      <div className="rounded border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-3 py-2 font-mono text-xs text-[var(--text-primary)] overflow-x-auto">
        {restartCommand}
      </div>
      <button
        type="button"
        onClick={copy}
        className="mt-3 inline-flex items-center gap-1.5 text-xs text-[var(--accent-cool)] hover:text-[var(--accent)]"
      >
        <Copy className="size-3.5" />
        复制重启命令
      </button>
    </Card>
  );
}

function AdminActions() {
  const queryClient = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const clearAll = useMutation({
    mutationFn: () => api.postClearCache(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      setConfirming(false);
    },
  });
  return (
    <Card title="管理操作" subtitle="整个会话缓存库的操作">
      <button
        type="button"
        onClick={() => setConfirming(true)}
        className="inline-flex items-center gap-2 text-sm text-[var(--accent-hot)] hover:bg-[var(--accent-hot)]/10 rounded px-3 py-2 transition-colors"
      >
        <Trash className="size-4" />
        清空所有会话缓存
      </button>
      {confirming ? (
        <div className="mt-3 p-3 rounded-md border border-[var(--accent-hot)]/40 bg-[var(--accent-hot)]/5 text-sm text-[var(--text-primary)]">
          <p>
            确定清空所有缓存前缀？缓存重新建立前，后续请求需要完整预填充。
          </p>
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              onClick={() => clearAll.mutate()}
              disabled={clearAll.isPending}
              className="text-xs px-3 py-1 rounded bg-[var(--accent-hot)] text-white disabled:opacity-50"
            >
              {clearAll.isPending ? "正在清空…" : "确认清空缓存"}
            </button>
            <button
              type="button"
              onClick={() => setConfirming(false)}
              className="text-xs px-3 py-1 rounded border border-[var(--border-soft)] text-[var(--text-muted)]"
            >
              取消
            </button>
          </div>
        </div>
      ) : null}
    </Card>
  );
}
