import type { Interval } from "@/lib/replay";
import { POLICY_LABEL } from "@/lib/types";

const W = 1000;
const LEFT = 84;
const ROW = 34;
const TOP = 8;

function niceStep(spanMs: number): number {
  const target = spanMs / 8;
  const steps = [1000, 2000, 5000, 10000, 15000, 30000, 60000, 120000, 300000, 600000];
  return steps.find((s) => s >= target) ?? steps[steps.length - 1];
}

/** Execution timeline: one row per policy, a block per request while it held the backend. */
export function Timeline({ rows, tEnd, t, selected }: {
  rows: { policy: string; intervals: Interval[] }[]; tEnd: number; t: number; selected?: string;
}) {
  const span = Math.max(tEnd, 1);
  const x = (ms: number) => LEFT + (ms / span) * (W - LEFT - 12);
  const h = TOP + rows.length * ROW + 26;
  const step = niceStep(span);
  const ticks = Array.from({ length: Math.floor(span / step) + 1 }, (_, i) => i * step);
  return (
    <svg viewBox={`0 0 ${W} ${h}`} className="timeline" role="img"
         aria-label="Execution order over time for each policy">
      {ticks.map((ms) => (
        <g key={ms}>
          <line x1={x(ms)} x2={x(ms)} y1={TOP} y2={h - 22} className="grid" />
          <text x={x(ms)} y={h - 8} textAnchor="middle" className="tick">{Math.round(ms / 1000)}s</text>
        </g>
      ))}
      {rows.map((row, i) => {
        const y = TOP + i * ROW;
        return (
          <g key={row.policy}>
            <text x={LEFT - 10} y={y + ROW / 2 + 4} textAnchor="end"
                  className={`row-label${row.policy === selected ? " sel" : ""}`}>
              {POLICY_LABEL[row.policy] ?? row.policy}
            </text>
            {row.intervals.filter((iv) => iv.start <= t).map((iv) => {
              const x0 = x(iv.start);
              const w = Math.max(1.5, x(Math.min(iv.end, t)) - x0 - 1);
              return (
                <g key={iv.id}>
                  <rect x={x0} y={y + 5} width={w} height={ROW - 10} rx={3}
                        className={`blk ${iv.ok ? iv.size : "failed"}`}>
                    <title>{`${iv.id} (${iv.size}) ${(iv.start / 1000).toFixed(1)}–${(iv.end / 1000).toFixed(1)} s`}</title>
                  </rect>
                  {w > 34 && (
                    <text x={x0 + w / 2} y={y + ROW / 2 + 4} textAnchor="middle" className="blk-label">
                      {iv.id.replace(/^r0*(?=\d)/, "r")}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        );
      })}
      <line x1={x(Math.min(t, span))} x2={x(Math.min(t, span))} y1={TOP - 4} y2={h - 22} className="cursor" />
    </svg>
  );
}
