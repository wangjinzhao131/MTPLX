import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Keyboard, X } from "lucide-react";

const SHORTCUTS: { key: string; label: string }[] = [
  { key: "?", label: "显示或隐藏快捷键面板" },
  { key: "t", label: "切换主题（河马 → 河流 → 明亮 → 单色）" },
  { key: "g", label: "切换到下一个页面" },
  { key: "space", label: "暂停或恢复实时更新" },
  { key: "s", label: "开启或关闭新纪录提示音" },
  { key: "Esc", label: "关闭浮层" },
];

export function KeyboardShortcutsOverlay() {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "?" && !e.metaKey && !e.ctrlKey) {
        const target = e.target as HTMLElement | null;
        if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
        e.preventDefault();
        setOpen((v) => !v);
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title="键盘快捷键（?）"
        className="fixed bottom-16 right-4 z-30 inline-flex items-center justify-center rounded-full p-2 bg-[var(--bg-card)] border border-[var(--border-soft)] text-[var(--text-muted)] hover:text-[var(--text-primary)] shadow"
      >
        <Keyboard className="size-4" />
      </button>
      <AnimatePresence>
        {open ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/60 grid place-items-center p-4"
            onClick={() => setOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.96, y: 8 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.96, y: 8 }}
              className="bg-[var(--bg-card)] border border-[var(--border-soft)] rounded-2xl p-6 max-w-md w-full"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-base font-semibold text-[var(--text-primary)]">
                  键盘快捷键
                </h2>
                <button
                  type="button"
                  onClick={() => setOpen(false)}
                  className="text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                >
                  <X className="size-4" />
                </button>
              </div>
              <dl className="space-y-2 text-sm">
                {SHORTCUTS.map((s) => (
                  <div key={s.key} className="flex items-center justify-between gap-4">
                    <dt className="font-mono text-[var(--accent)] bg-[var(--bg-elevated)] px-2 py-0.5 rounded border border-[var(--border-soft)]">
                      {s.key}
                    </dt>
                    <dd className="text-[var(--text-muted)] text-right">{s.label}</dd>
                  </div>
                ))}
              </dl>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </>
  );
}
