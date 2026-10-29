import type { MetricsLatest } from "./types";

// Spans of the server's TTFT breakdown (mtplx/server/request_spans.py) in
// which a request waited for other work instead of doing its own.
export const WAIT_SPANS = [
  ["response_tail_wait_s", "上一轮提交"],
  ["postcommit_wait_s", "提交后处理"],
  ["scheduler_queue_s", "调度队列"],
  ["lock_wait_s", "生成锁"],
] as const;

export type RequestWait = {
  total_s: number;
  parts: { label: string; seconds: number }[];
};

// The time before a request's own work started. Read from the spans rather
// than `lock_wait_time_s`, which counts only part of it and, where it carries
// the queue wait, would count that span twice.
export function requestWait(row: MetricsLatest): RequestWait | null {
  const spans = row.ttft_spans?.exclusive_s;
  if (!spans) return null;
  const parts = WAIT_SPANS.flatMap(([key, label]) => {
    const seconds = spans[key];
    return typeof seconds === "number" && seconds > 0 ? [{ label, seconds }] : [];
  });
  return { total_s: parts.reduce((sum, part) => sum + part.seconds, 0), parts };
}
