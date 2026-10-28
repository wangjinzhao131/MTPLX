import { AnimatePresence, motion } from "motion/react";
import { Trophy } from "lucide-react";
import { useNewMaxTPSDetector } from "../hooks/useNewMaxTPSDetector";
import { fmtTokS } from "../lib/utils";

export function NewMaxTPSToast() {
  const { newMaxBanner } = useNewMaxTPSDetector();
  return (
    <div className="fixed top-16 right-4 z-50 pointer-events-none">
      <AnimatePresence>
        {newMaxBanner ? (
          <motion.div
            key={`${newMaxBanner.when_s}-${newMaxBanner.tok_s}`}
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 280, damping: 22 }}
            className="rounded-xl border border-[var(--accent)]/30 bg-[var(--bg-card)] shadow-[0_12px_40px_rgba(0,214,143,0.25)] px-4 py-3 flex items-center gap-3"
          >
            <Trophy className="size-5 text-[var(--accent)]" />
            <div className="leading-tight">
              <div className="text-sm font-semibold text-[var(--text-primary)]">
                刷新历史最高速度
              </div>
              <div className="text-xs text-[var(--text-muted)] tabular-nums">
                {fmtTokS(newMaxBanner.tok_s)} tok/s
              </div>
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
