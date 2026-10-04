# Scheduler dashboard (Next.js)

```bash
cd ui
npm install
npm run dev        # http://localhost:3000   (or: npm run build && npm start)
```

Requires Node 18.18+ (developed on Node 24, Next 16, React 19).

## Two data sources, one renderer

| Tab | Data | How it's produced |
| --- | --- | --- |
| **Simulation** | `public/sim/<workload>.json`: all three policies on the same synthetic workload | `python -m scheduler export-ui` (bundles are committed, so the UI works without Python) |
| **Recorded / live runs** | `../scheduler_runs/<run>/events.jsonl` through `GET /api/runs` and `GET /api/runs/<run>` | `python -m scheduler run ...` (mock now, llama.cpp later). Polled every second until `run_completed`. |

Set `SCHEDULER_RUNS_DIR` to read runs from another directory.

`lib/replay.ts` rebuilds the scheduler state at any instant from the event log, so simulated and
real runs render identically. The queue order comes from the engine's `queue_snapshot` events.
For Adaptive, the overdue promotion is reapplied at the displayed time, because requests become
overdue between events. That display logic was checked against all 420 engine decisions in the
bundled runs.

## Views

- **Simulation** opens on the head-of-line preset at the head-of-line moment. It has a 4-line
  explainer with a source badge and MAX_WAIT; workload presets; story buttons (**1 · Head-of-line**,
  **2 · Overdue promotion**, **3 · Results**, each jumping to a moment computed from the loaded data);
  and three **policy lanes**. Each lane shows what's ON GPU, then the QUEUE in that policy's service
  order. Blocks are sized by predicted tokens, the next request is marked NEXT, and overdue requests
  carry the text `⚠ OVERDUE` (`↑` = Adaptive moves them ahead). Each lane also shows done / mean /
  max wait / over-max counts. Below sit the GPU timeline, the end-of-run comparison (✓ = best per
  row, computed; p99 only with ≥100 requests; throughput as a sanity check), and per-policy detail.
- **Recorded / live runs** reads `scheduler_runs/`. Each run is badged **REAL**, **MOCK** or
  **UNKNOWN SOURCE** (never assumed real), and **LIVE**, **RECORDED** or **INCOMPLETE** (an
  unfinished run whose log has stopped updating is never shown as live). It also shows which backend
  and model served the run, which predictor was used, and the run's own MAX_WAIT (`3 × median
  service time` for real runs). Runs of the same manifest render as side-by-side lanes. **Run
  history in Tiger Data** is read-only and optional: it reports connected, not configured or
  unavailable, and every other view works from local files.

## Data shape

Components render `lib/view.ts` shapes (`RunSummaryView`, `MaxWaitView`, `Source`), mapped from the
backend's `summary.json` / `run_completed` summary by `toSummaryView`. Simulated bundles and real
runs populate the same shape.

## Tests

```bash
npm test           # node:test + tsx: view-model logic, story moments, display order == engine choice
npm run typecheck
npm run build
```

## Deep links

`/?workload=head_of_line&policy=sejf&t=9000` opens a workload at simulated time t (ms).
`/?mode=runs` opens the recorded / live view; add `&run=<name>` to pick a run. The demo script is in
[`../docs/DEMO_FLOW.md`](../docs/DEMO_FLOW.md).
