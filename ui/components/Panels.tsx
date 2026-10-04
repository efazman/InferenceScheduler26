import type { ReplayState } from "@/lib/types";
import type { RunSummaryView } from "@/lib/view";
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
        <p className="muted">GPU idle.</p>
      )}
    </section>
  );
}

export function QueueTable({ state, maxWaitMs }: { state: ReplayState; maxWaitMs: number }) {
  return (
    <section className="card">
      <h3>Queue <span className="muted small">in service order</span></h3>
      {state.queue.length === 0 ? (
        <p className="muted">Queue empty.</p>
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
  // Latency and fairness are the point of K=1 scheduling; throughput is only a sanity check.
  const cards: [string, string, string?][] = [
    ["Mean latency", fmtSec(m.meanLatency)],
    ["p50 latency", fmtSec(m.p50)],
    ["p95 latency", fmtSec(m.p95)],
    ["Short-request latency", fmtSec(m.shortMean)],
    ["Max wait", fmtSec(m.maxWait), `${m.starved} over ${(starvationMs / 1000).toFixed(1)}s`],
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
      <div className="metric-note muted small">
        {m.completed} completed so far · throughput (sanity check) {m.throughputPerMin == null ? "–" : `${m.throughputPerMin.toFixed(1)}/min`}
      </div>
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

type Row = [label: string, key: keyof RunSummaryView, better: "min" | "max" | null, fmt: (v: number) => string];
const SUMMARY_ROWS: Row[] = [
  ["Mean latency", "meanLatencyMs", "min", (v) => fmtSec(v)],
  ["p50 latency", "p50LatencyMs", "min", (v) => fmtSec(v)],
  ["p95 latency", "p95LatencyMs", "min", (v) => fmtSec(v)],
  ["p99 latency", "p99LatencyMs", "min", (v) => fmtSec(v)],
  ["Mean queue wait", "meanWaitMs", "min", (v) => fmtSec(v)],
  ["Max queue wait", "maxWaitMs", "min", (v) => fmtSec(v)],
  ["Short-request latency", "shortMeanLatencyMs", "min", (v) => fmtSec(v)],
  ["Long-request latency", "longMeanLatencyMs", "min", (v) => fmtSec(v)],
  ["Long-request max wait", "longMaxWaitMs", "min", (v) => fmtSec(v)],
  ["Waited over MAX_WAIT", "starvationCount", "min", (v) => `${v}`],
  ["Failed requests", "failed", "min", (v) => `${v}`],
  // K=1 reordering moves waiting around; it should not change capacity, so no winner here
  ["Throughput (sanity check)", "throughputRpm", null, (v) => `${v.toFixed(1)}/min`],
];

/** End-of-run comparison. Best value per row is computed from the loaded results, never assumed. */
export function SummaryTable({ views }: { views: RunSummaryView[] }) {
  const p99ok = views.length > 0 && views.every((v) => v.p99Reliable);
  const anyFailed = views.some((v) => v.failed > 0);
  const rows = SUMMARY_ROWS.filter(([, key]) => (key !== "p99LatencyMs" || p99ok) && (key !== "failed" || anyFailed));
  return (
    <>
      <table className="tbl summary">
        <thead>
          <tr><th>Metric</th>{views.map((v) => <th key={v.policy} className="num">{POLICY_LABEL[v.policy] ?? v.policy}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map(([label, key, better, fmt]) => {
            const vals = views.map((v) => v[key] as number | null);
            const nums = vals.filter((x): x is number => x != null);
            const best = better && nums.length ? (better === "min" ? Math.min(...nums) : Math.max(...nums)) : null;
            const tie = best == null || nums.every((x) => Math.abs(x - best) <= 1e-6 * Math.max(1, Math.abs(x)));
            return (
              <tr key={key} className={key === "throughputRpm" ? "secondary" : undefined}>
                <td>{label}</td>
                {vals.map((x, i) => {
                  const win = !tie && x != null && Math.abs(x - (best as number)) <= 1e-6 * Math.max(1, Math.abs(x));
                  return <td key={views[i].policy} className={`num${win ? " best" : ""}`}>
                    {x == null ? "–" : fmt(x)}{win && <span className="best-mark" aria-label="best"> ✓</span>}
                  </td>;
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small table-note">
        ✓ = best in row. {views[0] ? `${views[0].n} requests per policy. ` : ""}
        {p99ok ? "" : "p99 hidden: fewer than 100 completed requests per policy. "}
        Lower is better except throughput, which K = 1 reordering should not change.
      </p>
    </>
  );
}
