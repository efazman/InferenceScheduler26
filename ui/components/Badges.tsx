import type { Source } from "@/lib/view";

const SOURCE: Record<Source, { label: string; title: string }> = {
  simulated: { label: "SIMULATED", title: "Synthetic workload, mock predictor, mock backend. Not a measurement." },
  mock: { label: "MOCK", title: "Wall-clock run on the mock backend. Synthetic lengths; not a measurement." },
  real: { label: "REAL", title: "Measured: real predictor and Llama 3.1 8B served by llama.cpp on the GPU." },
  unknown: { label: "UNKNOWN SOURCE", title: "This run did not record where its numbers came from. Do not treat as real." },
};

export function SourceBadge({ source, large }: { source: Source; large?: boolean }) {
  const s = SOURCE[source];
  return <span className={`src src-${source}${large ? " src-lg" : ""}`} title={s.title}>{s.label}</span>;
}

export function RunStateBadge({ state }: { state: "live" | "recorded" | "stopped" }) {
  if (state === "live") return <span className="state state-live" title="Events are still arriving">● LIVE</span>;
  if (state === "stopped")
    return <span className="state state-stopped" title="Run ended without run_completed (stopped or crashed)">■ INCOMPLETE</span>;
  return <span className="state state-rec" title="Finished run, replayed from its event log">RECORDED</span>;
}
