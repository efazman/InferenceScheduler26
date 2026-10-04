import type { ReplayState, RunSummary } from "@/lib/types";
import { POLICY_LABEL } from "@/lib/types";
import { fmtSec, type LiveMetrics, sizeOf } from "@/lib/replay";

export function SizeBadge({ size }: { size: "short" | "long" }) {
  return <span className={`badge ${size}`}>{size}</span>;
}

export function RunningCard({ state }: { state: ReplayState }) {
  const r = state.running;
  return (
    <section className="card">
      <h3>Running now <span className="muted small">K = 1, non-preemptive</span></h3>
      {r ? (
        <div className="running">
          <div className="running-id">{r.id}</div>
          <SizeBadge size={sizeOf(r)} />
          <div className="kv"><span>Predicted</span><b>{Math.round(r.predicted ?? 0)} tok</b></div>
          <div className="kv"><span>Waited</span><b>{fmtSec((r.start ?? 0) - r.arrival)}</b></div>
          <div className="kv"><span>Running for</span><b>{fmtSec(state.now - (r.start ?? state.now))}</b></div>
          <div className="kv"><span>Chosen because</span><b>{r.selectedBy?.replace("_", " ") ?? "–"}</b></div>
        </div>
      ) : (
        <p className="muted">Backend idle.</p>
      )}
    </section>
  );
}

export function QueueTable({ state, maxWaitMs }: { state: ReplayState; maxWaitMs: number }) {
  return (
    <section className="card">
      <h3>Queue <span className="muted small">in service order</span></h3>
      {state.queue.length === 0 ? (
        <p className="muted">Empty.</p>
      ) : (
        <table className="tbl">
          <thead>
            <tr><th>#</th><th>Request</th><th>Size</th><th className="num">Predicted</th><th className="num">Waiting</th><th>State</th></tr>
          </thead>
          <tbody>
            {state.queue.map((r, i) => {
              const wait = state.now - r.arrival;
              const overdue = wait >= maxWaitMs;
              return (
                <tr key={r.id} className={overdue ? "overdue-row" : undefined}>
                  <td>{i + 1}</td>
                  <td className="mono">{r.id}</td>
                  <td><SizeBadge size={sizeOf(r)} /></td>
                  <td className="num">{Math.round(r.predicted ?? 0)}</td>
                  <td className="num">{fmtSec(wait)}</td>
                  <td>{overdue ? <span className="badge overdue">overdue</span> : "queued"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}

export function MetricCards({ m, starvationMs }: { m: LiveMetrics; starvationMs: number }) {
  const cards: [string, string, string?][] = [
    ["p50 latency", fmtSec(m.p50)],
    ["p95 latency", fmtSec(m.p95)],
    ["Mean latency", fmtSec(m.meanLatency)],
    ["Throughput", m.throughputPerMin == null ? "–" : `${m.throughputPerMin.toFixed(1)}/min`],
    ["Max wait", fmtSec(m.maxWait), `${m.starved} over ${starvationMs / 1000}s`],
  ];
  return (
    <div className="metrics">
      {cards.map(([label, value, sub]) => (
        <div className="metric" key={label}>
          <div className="metric-label">{label}</div>
          <div className="metric-value">{value}</div>
          {sub && <div className="metric-sub">{sub}</div>}
        </div>
      ))}
      <div className="metric-note muted small">{m.completed} completed so far</div>
    </div>
  );
}

export function CompletedTable({ state }: { state: ReplayState }) {
  const rows = [...state.completed, ...state.failed].sort((a, b) => (b.end ?? 0) - (a.end ?? 0)).slice(0, 12);
  return (
    <section className="card">
      <h3>Completed <span className="muted small">latest first</span></h3>
      {rows.length === 0 ? (
        <p className="muted">Nothing finished yet.</p>
      ) : (
        <table className="tbl">
          <thead>
            <tr><th>Request</th><th>Size</th><th className="num">Predicted</th><th className="num">Actual</th>
              <th className="num">Wait</th><th className="num">Latency</th><th>Chosen by</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className={r.state === "failed" ? "overdue-row" : undefined}>
                <td className="mono">{r.id}</td>
                <td><SizeBadge size={sizeOf(r)} /></td>
                <td className="num">{Math.round(r.predicted ?? 0)}</td>
                <td className="num">{r.state === "failed" ? "failed" : r.actual}</td>
                <td className="num">{fmtSec((r.start ?? r.end ?? 0) - r.arrival)}</td>
                <td className="num">{fmtSec((r.end ?? 0) - r.arrival)}</td>
                <td>{r.selectedBy?.replace("_", " ") ?? "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

const SUMMARY_ROWS: [string, keyof RunSummary, "min" | "max", (v: number) => string][] = [
  ["Mean latency", "mean_latency_ms", "min", (v) => fmtSec(v)],
  ["p50 latency", "p50_latency_ms", "min", (v) => fmtSec(v)],
  ["p95 latency", "p95_latency_ms", "min", (v) => fmtSec(v)],
  ["Mean queue wait", "mean_queue_wait_ms", "min", (v) => fmtSec(v)],
  ["Max queue wait", "max_queue_wait_ms", "min", (v) => fmtSec(v)],
  ["Throughput", "throughput_rps", "max", (v) => `${(v * 60).toFixed(1)}/min`],
  ["Short-request latency", "short_mean_latency_ms", "min", (v) => fmtSec(v)],
  ["Long-request latency", "long_mean_latency_ms", "min", (v) => fmtSec(v)],
  ["Long-request max wait", "long_max_queue_wait_ms", "min", (v) => fmtSec(v)],
  ["Starved", "starvation_count", "min", (v) => `${v}`],
];

export function SummaryTable({ summaries }: { summaries: Record<string, RunSummary> }) {
  const policies = Object.keys(summaries);
  return (
    <table className="tbl summary">
      <thead>
        <tr><th>End of run</th>{policies.map((p) => <th key={p} className="num">{POLICY_LABEL[p] ?? p}</th>)}</tr>
      </thead>
      <tbody>
        {SUMMARY_ROWS.map(([label, key, better, fmt]) => {
          const vals = policies.map((p) => summaries[p][key] as number | null);
          const nums = vals.filter((v): v is number => v != null);
          const best = nums.length ? (better === "min" ? Math.min(...nums) : Math.max(...nums)) : null;
          const tie = nums.every((v) => Math.abs(v - (best ?? v)) < 1e-9);
          return (
            <tr key={key as string}>
              <td>{label}</td>
              {vals.map((v, i) => (
                <td key={policies[i]} className={`num${!tie && v === best ? " best" : ""}`}>{v == null ? "–" : fmt(v)}</td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
