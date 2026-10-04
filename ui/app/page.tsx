"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { QueueStrip } from "@/components/QueueStrip";
import { CompletedTable, MetricCards, QueueTable, RunningCard, SummaryTable } from "@/components/Panels";
import { Timeline } from "@/components/Timeline";
import { intervals, liveMetrics, replay, runEnd, sortEvents } from "@/lib/replay";
import type { PolicyRun, RecordedRunInfo, RunSummary, SchedEvent, SimBundle, SimIndex } from "@/lib/types";
import { POLICIES, POLICY_LABEL } from "@/lib/types";

const SPEEDS = [1, 5, 10, 25, 50];

/** Plays simulated/recorded time forward at `speed` x real time. */
function usePlayback(tEnd: number, initialSpeed = 10) {
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(initialSpeed);
  const last = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = (now: number) => {
      const dt = last.current == null ? 0 : now - last.current;
      last.current = now;
      setT((prev) => {
        const next = Math.min(prev + dt * speed, tEnd);
        if (next >= tEnd) setPlaying(false);
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      last.current = null;
    };
  }, [playing, speed, tEnd]);
  return { t, setT, playing, setPlaying, speed, setSpeed };
}

function Controls({ pb, tEnd, label }: { pb: ReturnType<typeof usePlayback>; tEnd: number; label: string }) {
  return (
    <div className="controls">
      <button className="btn primary" onClick={() => {
        if (pb.t >= tEnd) pb.setT(0);
        pb.setPlaying(!pb.playing);
      }}>{pb.playing ? "Pause" : pb.t >= tEnd && tEnd > 0 ? "Replay" : "Play"}</button>
      <button className="btn" onClick={() => { pb.setPlaying(false); pb.setT(0); }}>Reset</button>
      <input type="range" min={0} max={Math.max(tEnd, 1)} step={10} value={pb.t} className="scrub"
             onChange={(e) => { pb.setPlaying(false); pb.setT(Number(e.target.value)); }} aria-label="Time" />
      <span className="clock mono">{(pb.t / 1000).toFixed(1)} / {(tEnd / 1000).toFixed(1)} s {label}</span>
      <label className="speed">
        Speed
        <select value={pb.speed} onChange={(e) => pb.setSpeed(Number(e.target.value))}>
          {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
        </select>
      </label>
    </div>
  );
}

function PolicyTabs({ value, onChange, options }: { value: string; onChange: (p: string) => void; options: readonly string[] }) {
  return (
    <div className="seg" role="tablist" aria-label="Scheduling policy">
      {options.map((p) => (
        <button key={p} role="tab" aria-selected={value === p} className={value === p ? "on" : ""}
                onClick={() => onChange(p)}>{POLICY_LABEL[p] ?? p}</button>
      ))}
    </div>
  );
}

function maxPredicted(events: SchedEvent[]): number {
  return events.reduce((m, e) => (e.event_type === "cost_predicted" ? Math.max(m, e.predicted_output_tokens ?? 0) : m), 1);
}

function firstArrival(events: SchedEvent[]): number {
  return events.find((e) => e.event_type === "request_arrived")?.timestamp_ms ?? 0;
}

/** Shared detail view for one policy run at time t. */
function Detail({ run, t, maxWaitMs }: { run: PolicyRun; t: number; maxWaitMs: number }) {
  const state = useMemo(() => replay(run.events, t, run.policy === "adaptive" ? maxWaitMs : undefined),
    [run.events, run.policy, t, maxWaitMs]);
  const m = liveMetrics(state, firstArrival(run.events), maxWaitMs);
  return (
    <>
      <MetricCards m={m} starvationMs={maxWaitMs} />
      <div className="grid2">
        <div className="col">
          <RunningCard state={state} />
          <QueueTable state={state} maxWaitMs={maxWaitMs} />
        </div>
        <CompletedTable state={state} />
      </div>
    </>
  );
}

function SimulationView() {
  const [index, setIndex] = useState<SimIndex | null>(null);
  const [workload, setWorkload] = useState("head_of_line");
  const [bundle, setBundle] = useState<SimBundle | null>(null);
  const [policy, setPolicy] = useState("adaptive");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/sim/index.json").then((r) => r.json()).then(setIndex)
      .catch(() => setError("No simulated runs found. Run `python -m scheduler export-ui` from the repo root."));
  }, []);
  useEffect(() => {
    setBundle(null);
    fetch(`/sim/${workload}.json`).then((r) => r.json()).then(setBundle).catch(() => setError(`Missing /sim/${workload}.json`));
  }, [workload]);

  const runs: PolicyRun[] = useMemo(() => (bundle ? POLICIES.filter((p) => bundle.runs[p]).map((p) => ({
    policy: p, events: sortEvents(bundle.runs[p].events), summary: bundle.runs[p].summary,
  })) : []), [bundle]);
  const tEnd = Math.max(0, ...runs.map((r) => runEnd(r.events)));
  const pb = usePlayback(tEnd, 10);
  const deepLink = useRef<{ workload: string; t: number } | null>(null);
  useEffect(() => { // deep link: ?workload=head_of_line&policy=sejf&t=12000 (ms)
    const q = new URLSearchParams(window.location.search);
    const w = q.get("workload") ?? "head_of_line";
    if (q.get("workload")) setWorkload(w);
    if (q.get("policy")) setPolicy(q.get("policy")!);
    if (q.get("t")) deepLink.current = { workload: w, t: Number(q.get("t")) };
  }, []);
  useEffect(() => { // new workload loaded: rewind, unless a deep link targets this workload
    pb.setPlaying(false);
    const link = deepLink.current;
    if (bundle && link && bundle.workload.name === link.workload) {
      pb.setT(link.t);
      deepLink.current = null;
    } else {
      pb.setT(0);
    }
  }, [bundle]); // eslint-disable-line react-hooks/exhaustive-deps

  if (error) return <p className="error">{error}</p>;
  if (!index || !bundle) return <p className="muted">Loading simulated runs…</p>;
  const maxWaitMs = bundle.max_wait_ms;
  const maxTok = Math.max(...runs.map((r) => maxPredicted(r.events)));
  const selected = runs.find((r) => r.policy === policy) ?? runs[0];
  const summaries = Object.fromEntries(runs.map((r) => [r.policy, r.summary!])) as Record<string, RunSummary>;

  return (
    <>
      <div className="banner">{bundle.warning}</div>
      <div className="toolbar">
        <label>
          Workload
          <select value={workload} onChange={(e) => setWorkload(e.target.value)}>
            {index.workloads.map((w) => <option key={w.name} value={w.name}>{w.name} ({w.n})</option>)}
          </select>
        </label>
        <span className="muted small">{bundle.workload.description}</span>
        <span className="spacer" />
        <span className="muted small">Adaptive max wait: {maxWaitMs / 1000}s</span>
      </div>
      <Controls pb={pb} tEnd={tEnd} label="simulated" />

      <section className="card">
        <h3>Same requests, same moment, three policies <span className="muted small">running request, then the queue in the order each policy would serve it. Width = predicted tokens</span></h3>
        {runs.map((r) => (
          <QueueStrip key={r.policy} policy={r.policy}
                      state={replay(r.events, pb.t, r.policy === "adaptive" ? maxWaitMs : undefined)} maxWaitMs={maxWaitMs}
                      maxTokens={maxTok} selected={r.policy === policy} onSelect={() => setPolicy(r.policy)} />
        ))}
        <div className="legend small">
          <span><i className="sw short" /> short</span><span><i className="sw long" /> long</span>
          <span><i className="sw overdue" /> overdue (waited ≥ {maxWaitMs / 1000}s)</span>
          <span><i className="sw run" /> running</span>
        </div>
      </section>

      <section className="card">
        <h3>Execution order over time</h3>
        <Timeline rows={runs.map((r) => ({ policy: r.policy, intervals: intervals(r.events) }))}
                  tEnd={tEnd} t={pb.t} selected={policy} />
      </section>

      <div className="toolbar">
        <h2>Policy detail</h2>
        <PolicyTabs value={policy} onChange={setPolicy} options={runs.map((r) => r.policy)} />
      </div>
      <Detail run={selected} t={pb.t} maxWaitMs={maxWaitMs} />

      <section className="card">
        <h3>End-of-run comparison <span className="muted small">best value per row highlighted</span></h3>
        <SummaryTable summaries={summaries} />
      </section>
    </>
  );
}

function RecordedView() {
  const [runsDir, setRunsDir] = useState("");
  const [list, setList] = useState<RecordedRunInfo[]>([]);
  const [name, setName] = useState<string | null>(null);
  const [data, setData] = useState<{ events: SchedEvent[]; summary: RunSummary | null; complete: boolean } | null>(null);
  const [follow, setFollow] = useState(true);

  useEffect(() => {
    const load = () => fetch("/api/runs").then((r) => r.json()).then((j) => {
      setRunsDir(j.runs_dir);
      setList(j.runs);
      setName((n) => n ?? j.runs[0]?.name ?? null);
    });
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    if (!name) return;
    let stop = false;
    const load = () => fetch(`/api/runs/${name}`).then((r) => r.json()).then((j) => {
      if (stop) return;
      setData({ events: j.events ?? [], summary: j.summary, complete: j.complete });
      if (!j.complete) setTimeout(load, 1000); // live run: poll until run_completed
    });
    load();
    return () => { stop = true; };
  }, [name]);

  const events = data?.events ?? [];
  const tEnd = runEnd(events);
  const pb = usePlayback(tEnd, 1);
  const startWall = (events.find((e) => e.event_type === "run_started") as any)?.wall_time;
  const [, force] = useState(0);
  useEffect(() => { // live mode: advance the clock with wall time so waits keep growing between events
    if (!data || data.complete || !follow) return;
    const id = setInterval(() => force((x) => x + 1), 250);
    return () => clearInterval(id);
  }, [data, follow]);

  const info = list.find((r) => r.name === name);
  if (!list.length)
    return (
      <div className="card">
        <h3>No recorded runs yet</h3>
        <p>Runs are read from <code>{runsDir || "scheduler_runs/"}</code>. Create one from the repo root:</p>
        <pre>python -m scheduler run --backend mock --workload head_of_line --n 60 --policy adaptive --speedup 5</pre>
        <p className="muted">Later on the RTX 3060 Ti: <code>--backend llamacpp --predictor distilbert:&lt;artifacts&gt; --prompts-file &lt;prompts.jsonl&gt;</code></p>
      </div>
    );
  const live = data != null && !data.complete;
  const t = live && follow && startWall ? Math.max(tEnd, Date.now() - Date.parse(startWall)) : pb.t;
  const maxWaitMs = (events.find((e) => e.event_type === "run_started")?.data?.policy?.max_wait_ms as number) ??
    data?.summary?.starvation_threshold_ms ?? 15000;
  const policy = info?.policy ?? events[0]?.scheduler_policy ?? "?";
  const run: PolicyRun = { policy, events, summary: data?.summary ?? null };

  return (
    <>
      {info?.warning && <div className="banner">{info.warning}</div>}
      <div className="toolbar">
        <label>
          Run
          <select value={name ?? ""} onChange={(e) => setName(e.target.value)}>
            {list.map((r) => <option key={r.name} value={r.name}>{r.name}{r.complete ? "" : " (live)"}</option>)}
          </select>
        </label>
        <span className="muted small">policy {POLICY_LABEL[policy] ?? policy} · backend {info?.backend} · predictor {info?.predictor}</span>
        <span className="spacer" />
        {live && <label className="small"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow live</label>}
        {live && <span className="badge live">live</span>}
      </div>
      {!(live && follow) && <Controls pb={pb} tEnd={tEnd} label="recorded" />}
      <section className="card">
        <h3>Queue right now</h3>
        <QueueStrip policy={policy} state={replay(events, t, policy === "adaptive" ? maxWaitMs : undefined)} maxWaitMs={maxWaitMs} maxTokens={maxPredicted(events)} selected />
      </section>
      <section className="card">
        <h3>Execution order over time</h3>
        <Timeline rows={[{ policy, intervals: intervals(events) }]} tEnd={Math.max(tEnd, t)} t={t} selected={policy} />
      </section>
      <Detail run={run} t={t} maxWaitMs={maxWaitMs} />
      {data?.summary && (
        <section className="card"><h3>End-of-run summary</h3><SummaryTable summaries={{ [policy]: data.summary }} /></section>
      )}
    </>
  );
}

export default function Page() {
  const [mode, setMode] = useState<"sim" | "runs">("sim");
  useEffect(() => { // deep link: ?mode=runs opens the recorded / live view
    if (new URLSearchParams(window.location.search).get("mode") === "runs") setMode("runs");
  }, []);
  return (
    <main>
      <header className="top">
        <div>
          <h1>Adaptive LLM Scheduler</h1>
          <p className="muted">One GPU, one request at a time (K = 1, non-preemptive). The predictor estimates each
            request&apos;s output length; the policy decides who runs next.</p>
        </div>
        <div className="seg" role="tablist" aria-label="Data source">
          <button role="tab" aria-selected={mode === "sim"} className={mode === "sim" ? "on" : ""} onClick={() => setMode("sim")}>Simulation</button>
          <button role="tab" aria-selected={mode === "runs"} className={mode === "runs" ? "on" : ""} onClick={() => setMode("runs")}>Recorded / live runs</button>
        </div>
      </header>
      {mode === "sim" ? <SimulationView /> : <RecordedView />}
    </main>
  );
}
