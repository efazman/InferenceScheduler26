import type { ReplayState, ReqView } from "@/lib/types";
import { POLICY_LABEL } from "@/lib/types";
import { sizeOf } from "@/lib/replay";

const shortId = (id: string) => id.replace(/^r0*(?=\d)/, "r");

function Block({ r, now, maxWaitMs, maxTokens, running }: {
  r: ReqView; now: number; maxWaitMs: number; maxTokens: number; running?: boolean;
}) {
  const size = sizeOf(r);
  const tokens = r.predicted ?? 0;
  const width = Math.max(56, Math.min(180, (tokens / Math.max(maxTokens, 1)) * 180));
  const waited = (running ? (r.start ?? now) : now) - r.arrival;
  const overdue = !running && waited >= maxWaitMs;
  return (
    <div
      className={`qblock ${size}${running ? " running" : ""}${overdue ? " overdue" : ""}`}
      style={{ width }}
      title={`${r.id} · predicted ${Math.round(tokens)} tokens · waited ${(waited / 1000).toFixed(1)} s${
        overdue ? " · OVERDUE" : ""}`}
    >
      <span className="qid">{shortId(r.id)}</span>
      <span className="qtok">{Math.round(tokens)}</span>
    </div>
  );
}

export function QueueStrip({ policy, state, maxWaitMs, maxTokens, selected, onSelect }: {
  policy: string; state: ReplayState; maxWaitMs: number; maxTokens: number; selected?: boolean;
  onSelect?: () => void;
}) {
  return (
    <div className={`strip${selected ? " selected" : ""}`} onClick={onSelect} role={onSelect ? "button" : undefined}>
      <div className="strip-label">
        <strong>{POLICY_LABEL[policy] ?? policy}</strong>
        <span className="muted">{state.queue.length} queued</span>
      </div>
      <div className="strip-row">
        <div className="strip-running">
          {state.running ? (
            <Block r={state.running} now={state.now} maxWaitMs={maxWaitMs} maxTokens={maxTokens} running />
          ) : (
            <div className="qblock idle">idle</div>
          )}
        </div>
        <div className="strip-arrow" aria-hidden>←</div>
        <div className="strip-queue">
          {state.queue.length === 0 && <span className="muted small">queue empty</span>}
          {state.queue.map((r) => (
            <Block key={r.id} r={r} now={state.now} maxWaitMs={maxWaitMs} maxTokens={maxTokens} />
          ))}
        </div>
      </div>
    </div>
  );
}
