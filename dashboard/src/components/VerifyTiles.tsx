import { BigNumber, Card } from "./Card";
import { depthTotal, fmtNumber, fmtTokS } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function VerifyRatioTile() {
  const latest = useDashboardStore((s) => s.latest);
  const drafted = depthTotal(latest?.drafted_tokens, latest?.drafted_by_depth);
  const verifies = latest?.verify_calls ?? 0;
  const ratio = verifies > 0 ? drafted / verifies : null;
  return (
    <Card title="每次验证的草稿 token 数" subtitle="数值越高，通常速度越快">
      <BigNumber
        value={ratio === null ? "—" : ratio.toFixed(2)}
        unit="tok/call"
        tone={typeof ratio === "number" && ratio >= 3 ? "accent" : "default"}
        caption={`${fmtNumber(drafted)} 个草稿 token · ${fmtNumber(verifies)} 次验证`}
      />
    </Card>
  );
}

export function CorrectionBonusTile() {
  const latest = useDashboardStore((s) => s.latest);
  const correction = latest?.correction_tokens ?? 0;
  const bonus = latest?.bonus_tokens ?? 0;
  return (
    <Card title="修正与额外 token" subtitle="丢弃后的修正与额外生成">
      <div className="flex items-baseline gap-6">
        <div>
          <div className="text-2xl font-semibold text-[var(--accent-hot)] tabular-nums">
            {fmtNumber(correction)}
          </div>
          <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
            修正
          </div>
        </div>
        <div>
          <div className="text-2xl font-semibold text-[var(--accent)] tabular-nums">
            {fmtNumber(bonus)}
          </div>
          <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
            额外生成
          </div>
        </div>
      </div>
      <div className="text-xs text-[var(--text-muted)] mt-3">
        额外 token：该深度接受数超过草稿数；修正 token：拒绝草稿后的残差采样。
      </div>
    </Card>
  );
}

export function ServerTokSTile() {
  const latest = useDashboardStore((s) => s.latest);
  const requestTokS = latest?.request_tok_s ?? null;
  const decodeTokS = latest?.decode_tok_s ?? null;
  return (
    <Card title="生成速度与请求平均速度" subtitle="生成速度不包含预填充耗时">
      <div className="flex items-baseline gap-6">
        <div>
          <div className="text-2xl font-semibold text-[var(--accent)] tabular-nums">
            {fmtTokS(decodeTokS)}
          </div>
          <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
            生成速度（token/s）
          </div>
        </div>
        <div>
          <div className="text-2xl font-semibold text-[var(--accent-cool)] tabular-nums">
            {fmtTokS(requestTokS)}
          </div>
          <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
            请求平均速度（token/s）
          </div>
        </div>
      </div>
    </Card>
  );
}
