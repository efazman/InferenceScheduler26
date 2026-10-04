// Frontend logic tests (node:test + tsx). Run: npm test
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { replay, sortEvents } from "../lib/replay";
import type { SchedEvent, SimBundle } from "../lib/types";
import {
  busiestMoment, formatMaxWait, maxWaitFromRun, sizesFromEvents, sourceOf, staticMaxWait, storyMoments, toSummaryView,
} from "../lib/view";

const bundle = (name: string): SimBundle => JSON.parse(readFileSync(new URL(`../public/sim/${name}.json`, import.meta.url), "utf-8"));
const WORKLOADS = ["head_of_line", "mostly_short", "balanced", "mostly_long", "bursty"];

test("source never silently defaults to real", () => {
  assert.equal(sourceOf("real"), "real");
  assert.equal(sourceOf(undefined, "mock"), "mock");
  assert.equal(sourceOf(undefined, undefined, true), "simulated");
  assert.equal(sourceOf(undefined, "llamacpp"), "real");
  assert.equal(sourceOf(undefined, undefined), "unknown");
  assert.equal(sourceOf("garbage", "something-else"), "unknown");
});

test("summary view maps the backend summary and tolerates missing fields", () => {
  const b = bundle("head_of_line");
  const v = toSummaryView(b.runs.fifo.summary, "fifo", "simulated")!;
  assert.equal(v.meanLatencyMs, b.runs.fifo.summary.mean_latency_ms);
  assert.equal(v.throughputRpm, (b.runs.fifo.summary.throughput_rps as number) * 60);
  assert.equal(v.p99Reliable, false); // 10 requests: p99 must not look meaningful
  const partial = toSummaryView({ n_completed: 3, mean_latency_ms: 1200 }, "sejf", "real")!;
  assert.equal(partial.p95LatencyMs, null);
  assert.equal(partial.starvationCount, null);
  assert.equal(toSummaryView(null, "x", "real"), null);
});

test("MAX_WAIT display: derived, explicit, compressed mock, simulation, unknown", () => {
  const derived = maxWaitFromRun({ max_wait: { mode: "derived", max_wait_ms: 18390, multiplier: 3, median_service_ms: 6130,
    n_samples: 2000, effective_max_wait_ms: 18390 }, policy: { max_wait_ms: 18390 } });
  const f = formatMaxWait(derived);
  assert.equal(f.main, "MAX_WAIT = 18.4 s");
  assert.match(f.detail, /3 × 6\.13 s median model service time \(n = 2000\)/);
  const mock = formatMaxWait(maxWaitFromRun({ max_wait: { mode: "explicit", max_wait_ms: 15000, effective_max_wait_ms: 3000 },
    time_scale: 0.2 }));
  assert.equal(mock.main, "MAX_WAIT = 3.0 s");
  assert.match(mock.detail, /explicit 15\.0 s on a 5× compressed mock clock/);
  assert.match(formatMaxWait(staticMaxWait(15000)).detail, /simulation setting/);
  assert.equal(formatMaxWait(maxWaitFromRun(undefined)).main, "MAX_WAIT unknown");
});

test("story moments are found in the data, not hard-coded", () => {
  const b = bundle("head_of_line");
  const byPolicy = Object.fromEntries(Object.entries(b.runs).map(([p, r]) => [p, sortEvents(r.events)]));
  const sizes = new Map(b.requests.map((r) => [r.request_id, r.size_class]));
  const m = storyMoments(byPolicy, (id) => (sizes.get(id) === "long" ? "long" : "short"));
  assert.ok(m.hol != null && m.promote != null && m.end > m.promote!);
  // at the head-of-line moment, FIFO's next request is long while SEJF's is short
  const fifo = replay(byPolicy.fifo, m.hol!), sejf = replay(byPolicy.sejf, m.hol!);
  assert.equal(sizes.get(fifo.queue[0].id), "long");
  assert.equal(sizes.get(sejf.queue[0].id), "short");
  // at the promotion moment, Adaptive's next request is overdue
  const ad = replay(byPolicy.adaptive, m.promote!, b.max_wait_ms);
  assert.ok(m.promote! - ad.queue[0].arrival >= b.max_wait_ms);
});

test("displayed queue head equals the engine's actual choice at every decision", () => {
  let checked = 0;
  for (const w of WORKLOADS) {
    const b = bundle(w);
    for (const policy of ["fifo", "sejf", "adaptive"]) {
      const ev = sortEvents(b.runs[policy].events);
      for (const e of ev.filter((x) => x.event_type === "request_selected")) {
        const before = ev.filter((x) => x.seq < e.seq);
        const shown = replay(before, e.timestamp_ms, policy === "adaptive" ? b.max_wait_ms : undefined).queue[0]?.id;
        assert.equal(shown, e.request_id, `${w}/${policy} at ${e.timestamp_ms}`);
        checked++;
      }
    }
  }
  assert.ok(checked > 400);
});

test("recorded-run helpers", () => {
  const ev: SchedEvent[] = [
    { event_type: "request_arrived", timestamp_ms: 0, seq: 1, request_id: "a", data: { size_class: "long" } },
    { event_type: "cost_predicted", timestamp_ms: 0, seq: 2, request_id: "b", predicted_output_tokens: 40 },
    { event_type: "cost_predicted", timestamp_ms: 0, seq: 3, request_id: "c", predicted_output_tokens: 900 },
    { event_type: "request_selected", timestamp_ms: 100, seq: 4, request_id: "a", queue_depth: 1 },
    { event_type: "request_selected", timestamp_ms: 900, seq: 5, request_id: "b", queue_depth: 4 },
  ];
  const size = sizesFromEvents(ev);
  assert.deepEqual([size("a"), size("b"), size("c")], ["long", "short", "long"]);
  assert.equal(busiestMoment(ev), 899);
});
