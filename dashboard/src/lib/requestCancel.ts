import type { MetricsLatest } from "./types";
import { fmtNumber, fmtSeconds } from "./utils";

export type Cancellation = {
  /** Badge text, e.g. "cancelled · client · 3m 0s". */
  label: string;
  /** Why and after how long, in a sentence, plus the cache outcome. */
  tooltip: string;
  /** The cache outcome the request still had, e.g. "hit 92,636"; null when unknown. */
  cacheHint: string | null;
};

// Short names for the badge; the tooltip carries the raw reason. The keys are
// the server's cancellation reasons (the cancel origins in openai.py).
const REASON_LABELS: Record<string, string> = {
  client_disconnected: "客户端",
  client_disconnect: "客户端",
  stream_cancelled: "流式响应",
  stream_teardown: "流式响应",
  stream_stall: "流停滞",
  stop_sequence: "停止序列",
  early_tool_finish: "工具调用",
  hidden_tool_guard: "工具保护",
  post_endpoint: "取消接口",
};

const REASON_SENTENCES: Record<string, string> = {
  client_disconnected: "客户端关闭了连接",
  client_disconnect: "客户端关闭了连接",
  stream_cancelled: "流式响应被取消",
  stream_teardown: "流式响应被关闭",
  stream_stall: "流式响应停滞",
  stop_sequence: "停止序列结束了回答",
  early_tool_finish: "服务端提前结束了工具调用",
  hidden_tool_guard: "隐藏工具流保护机制停止了请求",
  post_endpoint: "POST /v1/mtplx/cancel 停止了请求",
};

/**
 * The cache outcome a cancelled row still reports; null when the server did
 * not know it yet. Older servers wrote `session_cache_hit: false` and no
 * `cached_tokens` on every cancelled row, so without a token count the
 * outcome is unknown as well.
 */
export function cancelledCacheHint(row: MetricsLatest): string | null {
  if (row.cached_tokens === null || row.cached_tokens === undefined) return null;
  if (row.session_cache_hit === true) {
    const source = row.cache_source && row.cache_source !== "ram" ? ` ${row.cache_source}` : "";
    return `命中${source} ${fmtNumber(row.cached_tokens)}`;
  }
  if (row.session_cache_hit === false) {
    return row.cache_miss_reason ? `未命中：${row.cache_miss_reason}` : "未命中";
  }
  return null;
}

/**
 * Describe a request that was cancelled before it finished, or null for one
 * that ran to the end. Its row would otherwise fall back to the MISS badge,
 * whatever the cache did.
 */
export function cancellationOf(row: MetricsLatest): Cancellation | null {
  if (!row.request_cancelled) return null;
  const reason = row.cancellation_reason ?? "";
  const elapsed = row.cancellation_elapsed_s ?? row.request_elapsed_s;
  const parts = ["已取消"];
  if (reason) parts.push(REASON_LABELS[reason] ?? reason);
  if (typeof elapsed === "number" && Number.isFinite(elapsed)) parts.push(fmtSeconds(elapsed));
  const why = reason ? (REASON_SENTENCES[reason] ?? reason) : "原因未知";
  const after =
    typeof elapsed === "number" && Number.isFinite(elapsed) ? `（经过 ${fmtSeconds(elapsed)}）` : "";
  const cacheHint = cancelledCacheHint(row);
  const cacheLine =
    cacheHint === null
      ? "缓存结果未知：请求在提示词读取完成前就已结束。"
      : `缓存：${cacheHint}。`;
  return {
    label: parts.join(" · "),
    tooltip: `已取消${after}：${why}${reason && REASON_SENTENCES[reason] ? `（${reason}）` : ""}。\n${cacheLine}\n不计入命中率。`,
    cacheHint,
  };
}

/** Rows that count toward a cache hit rate: requests that ran to the end. */
export function countsTowardHitRate(row: MetricsLatest): boolean {
  return !row.request_cancelled;
}
