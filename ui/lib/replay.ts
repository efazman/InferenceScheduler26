import type { ReplayState, ReqView, SchedEvent } from "./types";

// Display-only threshold used when a request has no size label (real prompts): requests whose
// PREDICTED length is at least this are drawn as "long". Matches scheduler.config.LONG_TOKEN_THRESHOLD.
export const LONG_TOKEN_THRESHOLD = 300;

export function sortEvents(events: SchedEvent[]): SchedEvent[] {
  return [...events].sort((a, b) => a.timestamp_ms - b.timestamp_ms || a.seq - b.seq);
}

export function sizeOf(r: Pick<ReqView, "sizeClass" | "predicted" | "actual">): "short" | "long" {
  if (r.sizeClass === "short" || r.sizeClass === "long") return r.sizeClass;
  const tokens = r.actual ?? r.predicted ?? 0;
  return tokens >= LONG_TOKEN_THRESHOLD ? "long" : "short";
}

/** Rebuild scheduler state at time t by applying every event with timestamp <= t.
 *
 * The queue order comes from the engine's latest queue_snapshot. For the adaptive policy the
 * order also depends on the clock (requests become overdue while a job runs, with no event), so
 * when `adaptiveMaxWaitMs` is given the documented promotion rule is re-applied at t: overdue
 * requests first, oldest arrival first; everything else keeps the engine's (SEJF) order. */
export function replay(events: SchedEvent[], t: number, adaptiveMaxWaitMs?: number): ReplayState {
  const reqs = new Map<string, ReqView>();
  const completed: ReqView[] = [];
  const failed: ReqView[] = [];
  let order: string[] = [];
  let running: string | null = null;

  for (const e of events) {
    if (e.timestamp_ms > t) break;
    const id = e.request_id ?? "";
    const r = reqs.get(id);
    switch (e.event_type) {
      case "request_arrived":
        reqs.set(id, { id, sizeClass: e.data?.size_class ?? null, arrival: e.timestamp_ms, state: "arriving" });
        break;
      case "cost_predicted":
        if (r) r.predicted = e.predicted_output_tokens;
        break;
      case "request_enqueued":
        if (r) Object.assign(r, { state: "queued", enqueue: e.timestamp_ms });
        break;
      case "queue_snapshot":
        order = e.data?.order ?? order;
        break;
      case "request_selected":
        if (r) r.selectedBy = e.data?.reason;
        break;
      case "inference_started":
        if (r) Object.assign(r, { state: "running", start: e.timestamp_ms });
        running = id;
        break;
      case "inference_completed":
        if (r) {
          Object.assign(r, { state: "completed", end: e.timestamp_ms, actual: e.actual_output_tokens });
          completed.push(r);
        }
        if (running === id) running = null;
        break;
      case "request_failed":
        if (r) {
          Object.assign(r, { state: "failed", end: e.timestamp_ms, error: e.error });
          failed.push(r);
        }
        if (running === id) running = null;
        break;
    }
  }

  const queued = new Set([...reqs.values()].filter((r) => r.state === "queued").map((r) => r.id));
  const queue = order.filter((id) => queued.has(id)).map((id) => reqs.get(id)!);
  // Logs written without queue snapshots: fall back to enqueue order.
  for (const r of [...reqs.values()].filter((r) => r.state === "queued" && !order.includes(r.id)).sort(
    (a, b) => (a.enqueue ?? 0) - (b.enqueue ?? 0),
  )) {
    queue.push(r);
  }
  if (adaptiveMaxWaitMs != null) {
    const overdue = queue.filter((r) => t - r.arrival >= adaptiveMaxWaitMs)
      .sort((a, b) => a.arrival - b.arrival || (a.enqueue ?? 0) - (b.enqueue ?? 0) || a.id.localeCompare(b.id));
    const rest = queue.filter((r) => t - r.arrival < adaptiveMaxWaitMs);
    queue.splice(0, queue.length, ...overdue, ...rest);
  }
  return { now: t, running: running ? reqs.get(running) ?? null : null, queue, completed, failed, arrived: reqs.size };
}

export function percentile(values: number[], q: number): number | null {
  if (!values.length) return null;
  const xs = [...values].sort((a, b) => a - b);
  const pos = ((xs.length - 1) * q) / 100;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo);
}

export interface LiveMetrics {
  completed: number;
  meanLatency: number | null;
  p50: number | null;
  p95: number | null;
  throughputPerMin: number | null;
  maxWait: number | null; // includes requests still waiting right now
  starved: number;
}

/** Metrics over what has happened by state.now (same definitions as scheduler/metrics.py). */
export function liveMetrics(s: ReplayState, firstArrival: number, starvationMs: number): LiveMetrics {
  const lat = s.completed.map((r) => (r.end ?? 0) - r.arrival);
  const started = [...s.completed, ...(s.running ? [s.running] : [])].map((r) => (r.start ?? 0) - r.arrival);
  const waiting = s.queue.map((r) => s.now - r.arrival);
  const waits = [...started, ...waiting];
  const elapsedMin = (s.now - firstArrival) / 60000;
  return {
    completed: s.completed.length,
    meanLatency: lat.length ? lat.reduce((a, b) => a + b, 0) / lat.length : null,
    p50: percentile(lat, 50),
    p95: percentile(lat, 95),
    throughputPerMin: elapsedMin > 0 ? s.completed.length / elapsedMin : null,
    maxWait: waits.length ? Math.max(...waits) : null,
    starved: waits.filter((w) => w > starvationMs).length,
  };
}

export interface Interval {
  id: string;
  start: number;
  end: number;
  size: "short" | "long";
  ok: boolean;
}

/** Service intervals (inference_started -> completed/failed) for the execution timeline. */
export function intervals(events: SchedEvent[]): Interval[] {
  const starts = new Map<string, number>();
  const sizes = new Map<string, ReqView>();
  const out: Interval[] = [];
  for (const e of events) {
    const id = e.request_id ?? "";
    if (e.event_type === "request_arrived")
      sizes.set(id, { id, sizeClass: e.data?.size_class ?? null, arrival: 0, state: "arriving" });
    if (e.event_type === "cost_predicted" && sizes.get(id)) sizes.get(id)!.predicted = e.predicted_output_tokens;
    if (e.event_type === "inference_started") starts.set(id, e.timestamp_ms);
    if ((e.event_type === "inference_completed" || e.event_type === "request_failed") && starts.has(id)) {
      const view = sizes.get(id);
      if (view && e.actual_output_tokens != null) view.actual = e.actual_output_tokens;
      out.push({ id, start: starts.get(id)!, end: e.timestamp_ms, size: view ? sizeOf(view) : "short",
                 ok: e.event_type === "inference_completed" });
    }
  }
  return out;
}

export function runEnd(events: SchedEvent[]): number {
  return events.length ? events[events.length - 1].timestamp_ms : 0;
}

export const fmtSec = (ms: number | null | undefined, digits = 1) =>
  ms == null ? "–" : `${(ms / 1000).toFixed(digits)} s`;
