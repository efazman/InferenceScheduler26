// View-model layer: the stable shapes components render, decoupled from simulator- or
// backend-specific fields. Simulated bundles and real llama.cpp runs map into the same shapes.

import type { RunSummary, SchedEvent } from "./types";
import { LONG_TOKEN_THRESHOLD } from "./replay";

export type Source = "simulated" | "mock" | "real" | "unknown";

export interface RunSummaryView {
  policy: string;
  source: Source;
  n: number;
  failed: number;
  meanLatencyMs: number | null;
  p50LatencyMs: number | null;
  p95LatencyMs: number | null;
  p99LatencyMs: number | null;
  p99Reliable: boolean; // only show p99 when the sample is large enough (>= 100 completed)
  meanWaitMs: number | null;
  maxWaitMs: number | null;
  shortMeanLatencyMs: number | null;
  longMeanLatencyMs: number | null;
  longMaxWaitMs: number | null;
  starvationCount: number | null;
  throughputRpm: number | null; // sanity check only: K=1 reordering should not change it
  maxWaitThresholdMs: number | null;
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export function toSummaryView(s: RunSummary | Record<string, unknown> | null | undefined, policy: string,
                              source: Source): RunSummaryView | null {
  if (!s) return null;
  const r = s as Record<string, unknown>;
  const rps = num(r.throughput_rps);
  return {
    policy, source,
    n: num(r.n_completed) ?? 0, failed: num(r.n_failed) ?? 0,
    meanLatencyMs: num(r.mean_latency_ms), p50LatencyMs: num(r.p50_latency_ms),
    p95LatencyMs: num(r.p95_latency_ms), p99LatencyMs: num(r.p99_latency_ms), p99Reliable: r.p99_reliable === true,
    meanWaitMs: num(r.mean_queue_wait_ms), maxWaitMs: num(r.max_queue_wait_ms),
    shortMeanLatencyMs: num(r.short_mean_latency_ms), longMeanLatencyMs: num(r.long_mean_latency_ms),
    longMaxWaitMs: num(r.long_max_queue_wait_ms), starvationCount: num(r.starvation_count),
    throughputRpm: rps == null ? null : rps * 60, maxWaitThresholdMs: num(r.starvation_threshold_ms),
  };
}

export function sourceOf(measurement: unknown, backend?: unknown, simulated?: unknown): Source {
  if (measurement === "real" || measurement === "mock" || measurement === "simulated") return measurement;
  if (simulated === true) return "simulated";
  if (backend === "mock") return "mock";
  if (backend === "llamacpp") return "real";
  return "unknown"; // never silently assume real
}

// ------------------------------------------------------------------------- adaptive threshold

export interface MaxWaitView {
  mode: "static" | "derived" | "unknown";
  maxWaitMs: number | null; // the threshold actually applied
  simulation?: boolean; // static threshold of a simulated bundle
  configuredMs?: number | null; // explicit value before mock time compression (if different)
  timeScale?: number | null;
  multiplier?: number | null;
  medianServiceMs?: number | null;
  sourceFile?: string | null;
  samples?: number | null;
}

/** From a recorded run's run_started event data (python: scheduler/max_wait.py MaxWaitConfig). */
export function maxWaitFromRun(started: Record<string, any> | undefined): MaxWaitView {
  const mw = started?.max_wait;
  const applied = num(mw?.effective_max_wait_ms) ?? num(started?.policy?.max_wait_ms) ?? num(started?.starvation_threshold_ms);
  if (mw?.mode === "derived")
    return { mode: "derived", maxWaitMs: applied ?? num(mw.max_wait_ms), multiplier: num(mw.multiplier),
             medianServiceMs: num(mw.median_service_ms), sourceFile: mw.source ?? null, samples: num(mw.n_samples) };
  if (mw?.mode === "explicit" || applied != null) {
    const configured = num(mw?.max_wait_ms);
    return { mode: "static", maxWaitMs: applied ?? configured, timeScale: num(started?.time_scale),
             configuredMs: configured != null && applied != null && Math.abs(configured - applied) > 1e-6 ? configured : null };
  }
  return { mode: "unknown", maxWaitMs: null };
}

export function staticMaxWait(ms: number): MaxWaitView {
  return { mode: "static", maxWaitMs: ms, simulation: true };
}

const sec = (ms: number | null | undefined, d = 1) => (ms == null ? "?" : `${(ms / 1000).toFixed(d)} s`);

export function formatMaxWait(m: MaxWaitView): { main: string; detail: string } {
  if (m.mode === "derived")
    return { main: `MAX_WAIT = ${sec(m.maxWaitMs)}`,
             detail: `${m.multiplier ?? "?"} × ${sec(m.medianServiceMs, 2)} median model service time` +
                     (m.samples ? ` (n = ${m.samples})` : "") };
  if (m.mode === "static") {
    if (m.simulation) return { main: `MAX_WAIT = ${sec(m.maxWaitMs)}`, detail: "static threshold (simulation setting)" };
    if (m.configuredMs != null)
      return { main: `MAX_WAIT = ${sec(m.maxWaitMs)}`,
               detail: `explicit ${sec(m.configuredMs)} on a ${m.timeScale ? Math.round(1 / m.timeScale) : "?"}× compressed mock clock` };
    return { main: `MAX_WAIT = ${sec(m.maxWaitMs)}`, detail: "explicit threshold" };
  }
  return { main: "MAX_WAIT unknown", detail: "this run did not record its threshold" };
}

// ------------------------------------------------------------------------- story moments

export interface StoryMoments {
  hol: number | null; // FIFO is about to run a long request while short ones wait behind it
  promote: number | null; // adaptive is about to promote an overdue request
  end: number;
}

/** Find the demo's key instants in the loaded runs (computed, never hard-coded). */
export function storyMoments(byPolicy: Record<string, SchedEvent[]>, sizeOf: (id: string) => "short" | "long"): StoryMoments {
  const end = Math.max(0, ...Object.values(byPolicy).map((ev) => (ev.length ? ev[ev.length - 1].timestamp_ms : 0)));
  let hol: number | null = null;
  const fifo = byPolicy.fifo ?? [];
  for (const e of fifo) {
    if (e.event_type !== "request_selected" || !e.request_id || sizeOf(e.request_id) !== "long") continue;
    // the queue just before this decision: last snapshot before it
    const snap = [...fifo].reverse().find((x) => x.event_type === "queue_snapshot" && x.seq < e.seq);
    const waiting = (snap?.data?.order ?? []).filter((id: string) => id !== e.request_id && sizeOf(id) === "short");
    if (waiting.length >= 2) {
      hol = Math.max(0, e.timestamp_ms - 1);
      break;
    }
  }
  const promoted = (byPolicy.adaptive ?? []).find((e) => e.event_type === "request_selected" && e.data?.reason === "overdue");
  return { hol, promote: promoted ? Math.max(0, promoted.timestamp_ms - 1) : null, end };
}

export function predictedSize(predicted: number | null | undefined): "short" | "long" {
  return (predicted ?? 0) >= LONG_TOKEN_THRESHOLD ? "long" : "short";
}

// ------------------------------------------------------------------------- recorded-run helpers

/** Short/long per request from the event log: workload label if recorded, else predicted length. */
export function sizesFromEvents(events: SchedEvent[]): (id: string) => "short" | "long" {
  const label = new Map<string, string | null>();
  const predicted = new Map<string, number | null | undefined>();
  for (const e of events) {
    if (!e.request_id) continue;
    if (e.event_type === "request_arrived") label.set(e.request_id, e.data?.size_class ?? null);
    if (e.event_type === "cost_predicted") predicted.set(e.request_id, e.predicted_output_tokens);
  }
  return (id) => {
    const l = label.get(id);
    return l === "short" || l === "long" ? l : predictedSize(predicted.get(id));
  };
}

/** The instant the queue was deepest: a good default view for a single recorded run. */
export function busiestMoment(events: SchedEvent[]): number {
  let best = { depth: -1, t: 0 };
  for (const e of events)
    if (e.event_type === "request_selected" && (e.queue_depth ?? 0) > best.depth)
      best = { depth: e.queue_depth ?? 0, t: Math.max(0, e.timestamp_ms - 1) };
  return best.t;
}
