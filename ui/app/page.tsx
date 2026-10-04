"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { SourceBadge, RunStateBadge } from "@/components/Badges";
import { CompletedTable, MetricCards, QueueTable, RunningCard, SummaryTable } from "@/components/Panels";
import { Legend, PolicyLane } from "@/components/QueueStrip";
import { Timeline } from "@/components/Timeline";
import { fmtSec, intervals, liveMetrics, replay, runEnd, sortEvents } from "@/lib/replay";
import type { PolicyRun, RecordedRunInfo, RunSummary, SchedEvent, SimBundle, SimIndex } from "@/lib/types";
import { POLICIES, POLICY_LABEL } from "@/lib/types";
import {
  busiestMoment, formatMaxWait, maxWaitFromRun, type MaxWaitView, type Source, sizesFromEvents, sourceOf, staticMaxWait,
  storyMoments, toSummaryView,
} from "@/lib/view";

const SPEEDS = [1, 5, 10, 25, 50];
const PRESET_LABEL: Record<string, string> = {
  head_of_line: "Head-of-line blocking", mostly_short: "Mostly short", balanced: "Balanced",
  mostly_long: "Mostly long", bursty: "Bursty",
};
// Head-of-line first: it's the default and the clearest demo; anything unknown keeps its order after.
const presetRank = (n: string) => { const i = Object.keys(PRESET_LABEL).indexOf(n); return i < 0 ? 99 : i; };
const presetOrder = (names: string[]) => [...names].sort((a, b) => presetRank(a) - presetRank(b));
const LIVE_STALE_MS = 30_000; // an incomplete run whose log hasn't changed for this long is not "live"

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
type Playback = ReturnType<typeof usePlayback>;

function Controls({ pb, tEnd, label, children }: { pb: Playback; tEnd: number; label: string; children?: React.ReactNode }) {
  return (
    <div className="controls">
      <button className="btn primary" onClick={() => {
        if (pb.t >= tEnd) pb.setT(0);
        pb.setPlaying(!pb.playing);
      }}>{pb.playing ? "Pause" : pb.t >= tEnd && tEnd > 0 ? "Replay" : "Play"}</button>
      <button className="btn" onClick={() => { pb.setPlaying(false); pb.setT(0); }}>Reset</button>
      {children}
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

function Seg({ value, onChange, options, label, ariaLabel }: {
  value: string; onChange: (v: string) => void; options: readonly string[]; label?: (v: string) => string; ariaLabel: string;
}) {
  return (
    <div className="seg" role="tablist" aria-label={ariaLabel}>
      {options.map((o) => (
        <button key={o} role="tab" aria-selected={value === o} className={value === o ? "on" : ""}
                onClick={() => onChange(o)}>{label ? label(o) : o}</button>
      ))}
    </div>
  );
}

function MaxWaitChip({ mw }: { mw: MaxWaitView }) {
  const f = formatMaxWait(mw);
  return <span className="chip" title="Adaptive fairness threshold: a request that has waited this long is served before shorter jobs">
    <b>{f.main}</b> <span className="muted">{f.detail}</span></span>;
}

function Explainer({ source, mw }: { source: Source; mw: MaxWaitView }) {
  return (
    <section className="explainer" aria-label="What is happening">
      <div className="explainer-badges"><SourceBadge source={source} large /><MaxWaitChip mw={mw} /></div>
      <ol>
        <li>Each policy receives the <b>same request arrivals</b>.</li>
        <li>The predictor estimates each request&apos;s output length <b>before</b> inference.</li>
        <li>The scheduler picks which request runs next on <b>one GPU</b> (one at a time, no preemption).</li>
        <li><b>Adaptive</b> favours short jobs but promotes any request that has waited past MAX_WAIT.</li>
      </ol>
    </section>
  );
}

const maxPredicted = (events: SchedEvent[]) =>
  events.reduce((m, e) => (e.event_type === "cost_predicted" ? Math.max(m, e.predicted_output_tokens ?? 0) : m), 1);
const firstArrival = (events: SchedEvent[]) => events.find((e) => e.event_type === "request_arrived")?.timestamp_ms ?? 0;
const nRequests = (events: SchedEvent[]) => events.filter((e) => e.event_type === "request_arrived").length;

/** The three policies, same instant, stacked in a fixed order. */
function Lanes({ runs, t, maxWaitMs, selected, onSelect }: {
  runs: PolicyRun[]; t: number; maxWaitMs: number; selected?: string; onSelect?: (p: string) => void;
}) {
  const maxTok = Math.max(1, ...runs.map((r) => maxPredicted(r.events)));
  return (
    <div className="lanes">
      {runs.map((r) => {
        const state = replay(r.events, t, r.policy === "adaptive" ? maxWaitMs : undefined);
        return <PolicyLane key={r.policy} policy={r.policy} state={state} maxWaitMs={maxWaitMs} maxTokens={maxTok}
                           metrics={liveMetrics(state, firstArrival(r.events), maxWaitMs)} total={nRequests(r.events)}
                           selected={r.policy === selected} onSelect={onSelect ? () => onSelect(r.policy) : undefined} />;
      })}
      <Legend maxWaitMs={maxWaitMs} />
    </div>
  );
}

/** Detail panels for one policy run at time t. */
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

function StoryButtons({ moments, pb }: { moments: ReturnType<typeof storyMoments>; pb: Playback }) {
  const go = (t: number | null) => () => { if (t != null) { pb.setPlaying(false); pb.setT(t); } };
  return (
    <span className="story" aria-label="Jump to a key moment">
      <button className="btn story-btn" disabled={moments.hol == null} onClick={go(moments.hol)}
              title="FIFO is about to run a long request while short ones wait">1 · Head-of-line</button>
      <button className="btn story-btn" disabled={moments.promote == null} onClick={go(moments.promote)}
              title={moments.promote == null ? "No request exceeded MAX_WAIT in this workload" : "Adaptive is about to promote an overdue request"}>
        2 · Overdue promotion</button>
      <button className="btn story-btn" onClick={go(moments.end)} title="Final state and the full comparison">3 · Results</button>
    </span>
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
    fetch(`/sim/${workload}.json`).then((r) => r.json()).then(setBundle)
      .catch(() => setError(`Simulated workload "${workload}" is missing. Run \`python -m scheduler export-ui\`.`));
  }, [workload]);

  const runs: PolicyRun[] = useMemo(() => (bundle ? POLICIES.filter((p) => bundle.runs[p]).map((p) => ({
    policy: p, events: sortEvents(bundle.runs[p].events), summary: bundle.runs[p].summary,
  })) : []), [bundle]);
  const sizes = useMemo(() => new Map(bundle?.requests.map((r) => [r.request_id, r.size_class]) ?? []), [bundle]);
  const moments = useMemo(() => storyMoments(Object.fromEntries(runs.map((r) => [r.policy, r.events])),
    (id) => (sizes.get(id) === "long" ? "long" : "short")), [runs, sizes]);
  const tEnd = moments.end;
  const pb = usePlayback(tEnd, 10);
  const deepLink = useRef<{ workload: string; t: number } | null>(null);
  useEffect(() => { // deep link: ?workload=head_of_line&policy=sejf&t=12000 (ms)
    const q = new URLSearchParams(window.location.search);
    const w = q.get("workload") ?? "head_of_line";
    if (q.get("workload")) setWorkload(w);
    if (q.get("policy")) setPolicy(q.get("policy")!);
    if (q.get("t")) deepLink.current = { workload: w, t: Number(q.get("t")) };
  }, []);
  useEffect(() => { // new workload: open on its head-of-line moment so the story is visible immediately
    pb.setPlaying(false);
    const link = deepLink.current;
    if (bundle && link && bundle.workload.name === link.workload) {
      pb.setT(link.t);
      deepLink.current = null;
    } else {
      pb.setT(moments.hol ?? 0);
    }
  }, [bundle, moments.hol]); // eslint-disable-line react-hooks/exhaustive-deps

  if (error) return <div className="card empty"><b>Simulation data unavailable.</b> <span className="muted">{error}</span></div>;
  if (!index || !bundle) return <div className="card empty muted">Loading simulated runs…</div>;
  const maxWaitMs = bundle.max_wait_ms;
  const selected = runs.find((r) => r.policy === policy) ?? runs[0];
  const views = runs.map((r) => toSummaryView(r.summary, r.policy, "simulated")).filter((v) => v != null);

  return (
    <>
      <Explainer source="simulated" mw={staticMaxWait(maxWaitMs)} />
      <div className="toolbar">
        <Seg ariaLabel="Workload preset" value={workload} onChange={setWorkload}
             options={presetOrder(index.workloads.map((w) => w.name))} label={(n) => PRESET_LABEL[n] ?? n} />
        <span className="muted small">{bundle.workload.description} · {nRequests(runs[0]?.events ?? [])} requests</span>
      </div>
      <Controls pb={pb} tEnd={tEnd} label="simulated"><StoryButtons moments={moments} pb={pb} /></Controls>

      <section className="card">
        <h3>Same requests, same moment, three policies <SourceBadge source="simulated" /></h3>
        <Lanes runs={runs} t={pb.t} maxWaitMs={maxWaitMs} selected={policy} onSelect={setPolicy} />
      </section>

      <section className="card">
        <h3>Who held the GPU, and when <span className="muted small">each block is one request · width = service time</span></h3>
        <Timeline rows={runs.map((r) => ({ policy: r.policy, intervals: intervals(r.events) }))}
                  tEnd={tEnd} t={pb.t} selected={policy} />
      </section>

      <section className="card">
        <h3>End-of-run comparison <SourceBadge source="simulated" /> <span className="muted small">same workload, all three policies</span></h3>
        <SummaryTable views={views} />
      </section>

      <div className="toolbar">
        <h2>Policy detail</h2>
        <Seg ariaLabel="Policy" value={policy} onChange={setPolicy} options={runs.map((r) => r.policy)}
             label={(p) => POLICY_LABEL[p] ?? p} />
      </div>
      <Detail run={selected} t={pb.t} maxWaitMs={maxWaitMs} />
    </>
  );
}

// ----------------------------------------------------------------------------- recorded / live

type RunData = { events: SchedEvent[]; summary: RunSummary | null; complete: boolean };

function useRunData(name: string | null): RunData | null {
  const [data, setData] = useState<RunData | null>(null);
  useEffect(() => {
    setData(null);
    if (!name) return;
    let stop = false;
    const load = () => fetch(`/api/runs/${name}`).then((r) => r.json()).then((j) => {
      if (stop) return;
      setData({ events: j.events ?? [], summary: j.summary, complete: j.complete });
      if (!j.complete) setTimeout(load, 1000); // live run: poll until run_completed
    }).catch(() => { if (!stop) setTimeout(load, 3000); });
    load();
    return () => { stop = true; };
  }, [name]);
  return data;
}

function runState(info: RecordedRunInfo | undefined, data: RunData | null): "live" | "recorded" | "stopped" {
  if (!info || data?.complete || info.complete) return "recorded";
  return Date.now() - info.updated_ms < LIVE_STALE_MS ? "live" : "stopped";
}

function servedBy(info?: RecordedRunInfo): string {
  if (!info) return "";
  if (info.backend === "mock") return "Served by the MOCK backend (no model)";
  const g = info.generation;
  if (info.backend === "llamacpp")
    return `Served by ${g?.model_name ?? "Llama"}${g?.quantization ? ` ${g.quantization}` : ""} via llama.cpp` +
      (g?.max_new_tokens ? ` · max ${g.max_new_tokens} new tokens` : "");
  return `Backend: ${info.backend ?? "unknown"}`;
}

function predictorLine(spec?: string | null): string {
  if (!spec) return "Predictor not connected.";
  if (spec === "mock") return "Predictor: mock stand-in (not the trained model)";
  const i = spec.indexOf(":");
  const kind = i < 0 ? spec : spec.slice(0, i);
  const dir = i < 0 ? "" : spec.slice(i + 1);
  return `Predictor: ${kind === "distilbert" ? "DistilBERT" : kind === "baseline" ? "prompt-length baseline" : kind}` +
    (dir ? ` (${dir.split(/[\\/]/).filter(Boolean).pop()})` : "");
}

/** Completed runs of one manifest under different policies: the measured version of the lanes view. */
function ManifestComparison({ runs, maxWaitMs, source }: { runs: RecordedRunInfo[]; maxWaitMs: number; source: Source }) {
  const [loaded, setLoaded] = useState<Record<string, SchedEvent[]>>({});
  useEffect(() => {
    let stop = false;
    Promise.all(runs.map((r) => fetch(`/api/runs/${r.name}`).then((x) => x.json()).then((j) => [r.name, j.events] as const)))
      .then((pairs) => { if (!stop) setLoaded(Object.fromEntries(pairs)); }).catch(() => undefined);
    return () => { stop = true; };
  }, [runs.map((r) => r.name).join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  const ordered = [...runs].sort((a, b) => POLICIES.indexOf(a.policy as any) - POLICIES.indexOf(b.policy as any));
  const ready = ordered.every((r) => loaded[r.name]);
  const policyRuns: PolicyRun[] = ready ? ordered.map((r) => ({ policy: r.policy ?? "?", events: loaded[r.name], summary: r.summary })) : [];
  const tEnd = ready ? Math.max(...policyRuns.map((r) => runEnd(r.events))) : 0;
  const moments = useMemo(() => storyMoments(Object.fromEntries(policyRuns.map((r) => [r.policy, r.events])),
    sizesFromEvents(policyRuns[0]?.events ?? [])), [ready]); // eslint-disable-line react-hooks/exhaustive-deps
  const pb = usePlayback(tEnd, 5);
  useEffect(() => { if (ready) pb.setT(moments.hol ?? moments.end); }, [ready]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!ready) return <div className="card empty muted">Loading runs of this manifest…</div>;
  const views = policyRuns.map((r) => toSummaryView(r.summary, r.policy, source)).filter((v) => v != null);
  return (
    <section className="card">
      <h3>Same manifest, {ordered.length} policies <SourceBadge source={source} />
        <span className="muted small">manifest {ordered[0].manifest_id} · identical requests and arrival times</span></h3>
      <Controls pb={pb} tEnd={tEnd} label={source === "real" ? "measured" : "recorded"}>
        <StoryButtons moments={moments} pb={pb} /></Controls>
      <Lanes runs={policyRuns} t={pb.t} maxWaitMs={maxWaitMs} />
      <h3 className="sub">Who held the GPU, and when</h3>
      <Timeline rows={policyRuns.map((r) => ({ policy: r.policy, intervals: intervals(r.events) }))} tEnd={tEnd} t={pb.t} />
      <h3 className="sub">End-of-run comparison <SourceBadge source={source} /></h3>
      <SummaryTable views={views} />
    </section>
  );
}

type TigerState = { status: "loading" | "connected" | "not_configured" | "unavailable"; table_exists?: boolean;
  runs: { run_id: string; policy: string | null; measurement: string | null; completed: number;
          mean_latency_ms: number | null; p95_latency_ms: number | null; max_wait_ms: number | null; started: string | null }[] };

function TigerHistory() {
  const [s, setS] = useState<TigerState>({ status: "loading", runs: [] });
  useEffect(() => {
    let stop = false;
    const load = () => fetch("/api/tiger").then((r) => r.json()).then((j) => { if (!stop) setS(j); })
      .catch(() => { if (!stop) setS({ status: "unavailable", runs: [] }); });
    load();
    const id = setInterval(load, 15000);
    return () => { stop = true; clearInterval(id); };
  }, []);
  return (
    <section className="card">
      <h3>Run history in Tiger Data
        {s.status === "connected" && <span className="state state-rec">● connected</span>}
        <span className="muted small">optional database copy; every view above works from local files</span></h3>
      {s.status === "loading" && <p className="muted">Checking Tiger Data…</p>}
      {s.status === "not_configured" && <p className="muted">Database not configured: showing local results only.</p>}
      {s.status === "unavailable" && <p className="muted">Database unavailable: showing local results.</p>}
      {s.status === "connected" && !s.runs.length && (
        <p className="muted">{s.table_exists === false ? "Connected, but the scheduler_events table hasn't been created yet."
          : "Connected. No runs stored yet: real runs appear here once uploaded (run --tiger or tiger-import)."}</p>)}
      {s.status === "connected" && s.runs.length > 0 && (
        <table className="tbl">
          <thead><tr><th>Run</th><th>Source</th><th>Policy</th><th className="num">Completed</th>
            <th className="num">Mean latency</th><th className="num">p95</th><th className="num">Max wait</th></tr></thead>
          <tbody>{s.runs.map((r) => (
            <tr key={r.run_id}><td className="mono">{r.run_id}</td><td><SourceBadge source={sourceOf(r.measurement)} /></td>
              <td>{POLICY_LABEL[r.policy ?? ""] ?? r.policy}</td><td className="num">{r.completed}</td>
              <td className="num">{fmtSec(r.mean_latency_ms)}</td><td className="num">{fmtSec(r.p95_latency_ms)}</td>
              <td className="num">{fmtSec(r.max_wait_ms)}</td></tr>))}</tbody>
        </table>
      )}
    </section>
  );
}

function RecordedView() {
  const [runsDir, setRunsDir] = useState("");
  const [list, setList] = useState<RecordedRunInfo[] | null>(null);
  const [name, setName] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);

  useEffect(() => {
    const load = () => fetch("/api/runs").then((r) => r.json()).then((j) => {
      setRunsDir(j.runs_dir);
      setList(j.runs);
      const wanted = new URLSearchParams(window.location.search).get("run"); // deep link: ?mode=runs&run=<name>
      setName((n) => n ?? (j.runs.some((r: RecordedRunInfo) => r.name === wanted) ? wanted : j.runs[0]?.name) ?? null);
    }).catch(() => setList((l) => l ?? []));
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);
  const data = useRunData(name);

  const events = data?.events ?? [];
  const tEnd = runEnd(events);
  const pb = usePlayback(tEnd, 1);
  const startedEv = events.find((e) => e.event_type === "run_started") as any;
  const [, force] = useState(0);
  const info = list?.find((r) => r.name === name);
  const state = runState(info, data);
  const opened = useRef<string | null>(null);
  useEffect(() => { // finished or stopped run: open on the moment its queue was deepest (once per run)
    if (data && state !== "live" && name && opened.current !== name) { opened.current = name; pb.setT(busiestMoment(events)); }
  }, [data, name, state]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { // live: advance the clock with wall time so waits keep growing between events
    if (state !== "live" || !follow) return;
    const id = setInterval(() => force((x) => x + 1), 250);
    return () => clearInterval(id);
  }, [state, follow]);

  if (list == null) return <div className="card empty muted">Looking for recorded runs…</div>;
  const anyReal = list.some((r) => r.measurement === "real");
  const realEmpty = (
    <div className="card empty">
      <span className="state state-stopped">NO REAL RUNS YET</span> Real GPU measurements will appear here after
      integration (runbook: <code>docs/RUNBOOK_REAL_EXPERIMENT.md</code>).{list.length ? " Every run listed below is MOCK." : ""}
    </div>
  );
  if (!list.length)
    return (
      <>
        {realEmpty}
        <div className="card">
          <h3>No recorded runs <span className="muted small">reading {runsDir || "scheduler_runs/"}</span></h3>
          <p className="muted">To try this view without a GPU, record a mock run from the repo root:</p>
          <pre>python -m scheduler run --backend mock --workload head_of_line --n 60 --policy adaptive --speedup 5</pre>
        </div>
        <TigerHistory />
      </>
    );

  const t = state === "live" && follow && startedEv?.wall_time ? Math.max(tEnd, Date.now() - Date.parse(startedEv.wall_time)) : pb.t;
  const mw = maxWaitFromRun(startedEv?.data ?? info?.max_wait_info ?? undefined);
  const maxWaitMs = mw.maxWaitMs ?? 15000;
  const source = sourceOf(info?.measurement, info?.backend);
  const policy = info?.policy ?? events[0]?.scheduler_policy ?? "?";
  const run: PolicyRun = { policy, events, summary: data?.summary ?? null };
  const siblings = info?.manifest_id ? list.filter((r) => r.complete && r.manifest_id === info.manifest_id) : [];
  const comparable = siblings.length >= 2 && new Set(siblings.map((r) => r.policy)).size === siblings.length;
  const view = toSummaryView(data?.summary, policy, source);

  return (
    <>
      {!anyReal && realEmpty}
      <div className="toolbar">
        <label>
          Run
          <select value={name ?? ""} onChange={(e) => setName(e.target.value)}>
            {list.map((r) => <option key={r.name} value={r.name}>
              {r.name} · {(r.measurement ?? "unknown").toUpperCase()}{r.complete ? "" : " · incomplete"}</option>)}
          </select>
        </label>
        <SourceBadge source={source} large />
        <RunStateBadge state={state} />
        {state === "live" && <label className="small"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow live</label>}
      </div>
      <div className="runinfo">
        <span>{servedBy(info)}</span><span>{predictorLine(info?.predictor)}</span>
        <span>Policy: <b>{POLICY_LABEL[policy] ?? policy}</b></span><MaxWaitChip mw={mw} />
      </div>
      {info?.warning && <div className="banner">{info.warning}</div>}
      {state === "stopped" && <div className="banner">This run stopped before finishing (no new events for over
        {` ${LIVE_STALE_MS / 1000}`} s). It&apos;s shown as recorded up to its last event; nothing here is live.</div>}
      {comparable && <ManifestComparison runs={siblings} maxWaitMs={maxWaitMs} source={source} />}
      <div className="toolbar"><h2>This run</h2></div>
      {!(state === "live" && follow) && <Controls pb={pb} tEnd={tEnd} label={state === "live" ? "so far" : "recorded"} />}
      <section className="card">
        <h3>Queue at this moment <SourceBadge source={source} /></h3>
        <Lanes runs={[run]} t={t} maxWaitMs={maxWaitMs} />
      </section>
      <section className="card">
        <h3>Who held the GPU, and when</h3>
        <Timeline rows={[{ policy, intervals: intervals(events) }]} tEnd={Math.max(tEnd, t)} t={t} selected={policy} />
      </section>
      <Detail run={run} t={t} maxWaitMs={maxWaitMs} />
      {view && (
        <section className="card"><h3>End-of-run summary <SourceBadge source={source} /></h3><SummaryTable views={[view]} /></section>
      )}
      <TigerHistory />
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
          <p className="muted">FIFO vs shortest-estimated-job-first (SEJF) vs Adaptive scheduling for a single-GPU Llama 3.1 8B server.</p>
        </div>
        <Seg ariaLabel="Data source" value={mode} onChange={(v) => setMode(v as "sim" | "runs")}
             options={["sim", "runs"]} label={(v) => (v === "sim" ? "Simulation" : "Recorded / live runs")} />
      </header>
      {mode === "sim" ? <SimulationView /> : <RecordedView />}
    </main>
  );
}
