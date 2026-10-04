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

- **Same requests, same moment, three policies:** the running request, then the queue in each
  policy's service order. Block width is predicted tokens, short and long are colour-coded, and
  overdue requests are red. This is where FIFO's `LONG | SHORT | SHORT` turns into SEJF's
  `SHORT | SHORT | LONG`.
- **Execution order over time:** which request held the GPU when, for each policy.
- **Policy detail:** live p50, p95, mean latency, throughput and max wait (including requests
  still waiting), plus the running request, the queue with wait times, and recent completions.
- **End-of-run comparison:** final metrics per policy, with the best value per row highlighted.

## Deep links

`/?workload=head_of_line&policy=sejf&t=9000` opens a workload at simulated time t (ms).
`/?mode=runs` opens the recorded / live view.
