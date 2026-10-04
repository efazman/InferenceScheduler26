import type { ReplayState, ReqView } from "@/lib/types";
import { POLICY_LABEL } from "@/lib/types";
import { fmtSec, type LiveMetrics, sizeOf } from "@/lib/replay";

const shortId = (id: string) => id.replace(/^r0*(?=\d)/, "r");

export const POLICY_RULE: Record<string, string> = {
  fifo: "arrival order",
  sejf: "shortest predicted first",
  adaptive: "shortest first · overdue promoted",
};

function Block({ r, now, maxWaitMs, maxTokens, running, next, promotes }: {
  r: ReqView; now: number; maxWaitMs: number; maxTokens: number; running?: boolean; next?: boolean; promotes?: boolean;
}) {
  const size = sizeOf(r);
  const tokens = Math.round(r.predicted ?? 0);
  const waited = (running ? (r.start ?? now) : now) - r.arrival;
  const overdue = !running && waited >= maxWaitMs;
  // width tracks predicted tokens; the floor keeps id, tag and wait readable on short jobs
  const width = running ? 150 : Math.max(overdue ? 132 : 104, Math.min(220, (tokens / Math.max(maxTokens, 1)) * 220));
  const ran = running ? now - (r.start ?? now) : 0;
  const tag = overdue ? (promotes ? "⚠ OVERDUE ↑" : "⚠ OVERDUE") : next ? "NEXT" : null;
  const title = `${r.id} · ${size} · predicted ${tokens} tokens · ${running ? `waited ${fmtSec(waited)}, running ${fmtSec(ran)}`
    : `waiting ${fmtSec(waited)}`}${overdue ? ` · max wait exceeded${promotes ? ": fairness boost, served before shorter jobs" : " (no fairness rule in this policy)"}` : ""}`;
  return (
    <div className={`qblock ${size}${running ? " on-gpu" : ""}${overdue ? " overdue" : ""}${next ? " next" : ""}`}
         style={{ width }} title={title} aria-label={title}>
      <span className="qtop">
        <span className="qid">{running ? "▶ " : ""}{shortId(r.id)}</span>
        {tag && <span className={`qtag${overdue ? " qtag-over" : ""}`}>{tag}</span>}
      </span>
      <span className="qmeta">{tokens} tok · {running ? `${fmtSec(ran)} run` : fmtSec(waited)}</span>
    </div>
  );
}

/** One policy: who holds the GPU, the queue in service order, and its numbers so far. */
export function PolicyLane({ policy, state, maxWaitMs, maxTokens, metrics, total, selected, onSelect }: {
  policy: string; state: ReplayState; maxWaitMs: number; maxTokens: number; metrics?: LiveMetrics; total?: number;
  selected?: boolean; onSelect?: () => void;
}) {
  const promotes = policy === "adaptive";
  return (
    <div className={`lane${selected ? " selected" : ""}`} onClick={onSelect} role={onSelect ? "button" : undefined}
         aria-label={`${POLICY_LABEL[policy] ?? policy} lane`}>
      <div className="lane-head">
        <span className="lane-name">{POLICY_LABEL[policy] ?? policy}</span>
        <span className="lane-rule">{POLICY_RULE[policy] ?? ""}</span>
        {metrics && (
          <span className="lane-stats">
            <span><b>{metrics.completed}</b>{total ? `/${total}` : ""} done</span>
            <span>mean <b>{fmtSec(metrics.meanLatency)}</b></span>
            <span>max wait <b>{fmtSec(metrics.maxWait)}</b></span>
            <span className={metrics.starved ? "warn" : undefined}>{metrics.starved} over max wait</span>
          </span>
        )}
      </div>
      <div className="lane-body">
        <div className="slot">
          <div className="slot-label">ON GPU</div>
          {state.running ? (
            <Block r={state.running} now={state.now} maxWaitMs={maxWaitMs} maxTokens={maxTokens} running />
          ) : (
            <div className="qblock idle">GPU idle</div>
          )}
        </div>
        <div className="queue">
          <div className="slot-label">QUEUE · {state.queue.length} waiting · next first</div>
          <div className="queue-row">
            {state.queue.length === 0 && <span className="muted small">Queue empty.</span>}
            {state.queue.map((r, i) => (
              <Block key={r.id} r={r} now={state.now} maxWaitMs={maxWaitMs} maxTokens={maxTokens} next={i === 0}
                     promotes={promotes} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export function Legend({ maxWaitMs }: { maxWaitMs: number }) {
  return (
    <div className="legend small" aria-label="Legend">
      <span><i className="sw short" /> short (predicted)</span>
      <span><i className="sw long" /> long (predicted)</span>
      <span>block width = predicted tokens</span>
      <span><i className="sw run" /> ▶ on GPU</span>
      <span><i className="sw overdue" /> ⚠ OVERDUE: waited ≥ MAX_WAIT ({(maxWaitMs / 1000).toFixed(1)} s) · ↑ Adaptive moves it ahead of shorter jobs</span>
    </div>
  );
}
