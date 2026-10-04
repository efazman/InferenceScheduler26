// Server-only helpers for recorded runs written by `python -m scheduler run` (events.jsonl).
import { promises as fs } from "fs";
import path from "path";
import type { RecordedRunInfo, SchedEvent } from "./types";

export const RUNS_DIR = process.env.SCHEDULER_RUNS_DIR ?? path.resolve(/*turbopackIgnore: true*/ process.cwd(), "..", "scheduler_runs");
const SAFE_NAME = /^[\w.\-]+$/;

export function runPath(name: string): string | null {
  return SAFE_NAME.test(name) ? path.join(/*turbopackIgnore: true*/ RUNS_DIR, name) : null;
}

export async function readEvents(dir: string): Promise<SchedEvent[]> {
  const text = await fs.readFile(path.join(dir, "events.jsonl"), "utf-8");
  const events: SchedEvent[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      events.push(JSON.parse(line));
    } catch {
      // partially written last line of a live run
    }
  }
  return events.sort((a, b) => a.timestamp_ms - b.timestamp_ms || a.seq - b.seq);
}

export async function listRuns(): Promise<RecordedRunInfo[]> {
  let names: string[] = [];
  try {
    names = await fs.readdir(RUNS_DIR);
  } catch {
    return [];
  }
  const out: RecordedRunInfo[] = [];
  for (const name of names.sort().reverse()) {
    const dir = runPath(name);
    if (!dir) continue;
    try {
      await fs.access(path.join(dir, "events.jsonl"));
    } catch {
      continue;
    }
    let meta: Record<string, any> = {};
    try {
      meta = JSON.parse(await fs.readFile(path.join(dir, "summary.json"), "utf-8"));
    } catch {
      // still running: summary.json is written at the end
    }
    const first = (await readEvents(dir))[0];
    out.push({
      name,
      policy: meta.policy ?? first?.scheduler_policy ?? null,
      backend: meta.backend ?? (first as any)?.backend ?? null,
      predictor: meta.predictor ?? (first as any)?.predictor ?? null,
      workload: meta.workload ?? (first as any)?.workload_name ?? null,
      complete: Boolean(meta.summary),
      warning: meta.warning ?? null,
    });
  }
  return out;
}
