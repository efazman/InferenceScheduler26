# Scheduler, simulator, metrics and dashboard

This is branch `feature/scheduler-ui-parallel`, built on a Mac while the RTX 3060 Ti generated
labels. Everything here runs **without** the GPU, llama.cpp or the final predictor. Those plug in
later without code changes (see *Integration* below).

```
prompt ──▶ Predictor ──▶ queue ──▶ SchedulerPolicy ──▶ InferenceBackend ──▶ EventSink ──▶ JSONL / Tiger Data ──▶ dashboard
           mock │ distilbert │ baseline   fifo │ sejf │ adaptive   mock │ llamacpp
```

| Module | Role |
| --- | --- |
| `models.py` | `Request` dataclass. Queue wait, service time and end-to-end latency are derived properties. |
| `predictors.py` | `Predictor` protocol: `predict(prompt) -> {"expected_output_tokens", "uncertainty", ...}`. Includes `MockPredictor`, `DistilBertPredictor`, `LinearBaselinePredictor` and `load_predictor(spec)`. `ml.predictor.LengthPredictor` satisfies the protocol as-is. |
| `backends.py` | `InferenceBackend` protocol: `generate(prompt, seed) -> {"text", "output_tokens", "latency_ms"}`. Includes `MockInferenceBackend` and `LlamaCppBackend`, which wraps the label generator's own client and decoding settings from `datagen/config.py`. |
| `policies.py` | `SchedulerPolicy.order(queue, now)` and `choose_next`. Includes FIFO, SEJF and Adaptive. Policies never call the backend. |
| `engine.py` | `SchedulerEngine(predictor, backend, policy, sink, clock)`: K = 1, non-preemptive. |
| `clock.py` | `VirtualClock` (simulation) and `WallClock` (real runs). The same engine loop runs with either. |
| `events.py` | Flat event schema plus sinks: `LocalJsonlEventSink`, `InMemoryEventSink`, `FanoutSink`. |
| `tigerdata.py` | Optional Tiger Data copy of the events (`TigerDataEventSink`, schema, import). Credentials come from the environment or a gitignored `.env`. |
| `metrics.py` | Percentiles, queue wait, throughput, starvation count, short/long split. |
| `workloads.py` | Five seeded synthetic workloads, plus `workload_from_prompts` for real prompts. |
| `simulator.py` | Engine + `VirtualClock` + mock predictor/backend. Service = 60 ms + 13.3 ms/token. |
| `manifest.py` | Experiment manifest: the fixed arrival trace (requests, prompts, arrivals, seeds) every policy replays. |
| `max_wait.py` | Explicit or derived (multiplier × median service time) adaptive threshold. |
| `__main__.py` | CLI: `simulate`, `compare`, `sweep`, `export-ui`, `prompts-from-split`, `make-manifest`, `measure-service`, `run`, `report`. |
| `../ui/` | Next.js dashboard (see `ui/README.md`). |

## The three policies

At every decision point (the backend is idle and the queue isn't empty), with `wait = now − arrival_time`:

| Policy | Picks |
| --- | --- |
| FIFO | Lowest `(enqueue_time, request_id)` |
| SEJF | Lowest `(predicted_tokens, enqueue_time, request_id)` |
| **Adaptive** | If any request has `wait ≥ MAX_WAIT`: the **oldest** of those (`arrival_time, enqueue_time, request_id`). Otherwise the same as SEJF. |

Where `MAX_WAIT` comes from:
- **Simulation and controlled experiments** use an explicit static value: 15 s by default
  (`DEFAULT_MAX_WAIT_MS`), or `--max-wait-ms`.
- **Real runs** use `MAX_WAIT = multiplier × measured median service time`, with the multiplier
  defaulting to 3.0 (`DEFAULT_MAX_WAIT_MULTIPLIER`). Pass `--median-service-from <runs.jsonl>` or
  `--median-service-ms`. A real llama.cpp run refuses to start without one of these, and the
  resolved value and its source are recorded in the run and shown in the dashboard.

The guarantee: once a request is overdue, only the job already running and requests that became
overdue before it can still delay it. New short traffic can't. That's a bound on starvation, not
a hard cap at `MAX_WAIT`, because with K = 1 and no preemption nothing can start while another
job runs. **The threshold has to be tuned to load.** See [`docs/SIM_RESULTS.md`](../docs/SIM_RESULTS.md):
at high load with 15 s, nearly everything is overdue and Adaptive behaves like FIFO.

## Events

Each line of `events.jsonl` is one event with these fields: `event_type, timestamp_ms, seq,
run_id, workload_name, scheduler_policy, backend, predictor, simulated, request_id, queue_depth,
predicted_output_tokens, uncertainty, actual_output_tokens, queue_wait_ms, service_time_ms,
end_to_end_latency_ms, success, error, data, wall_time`.

The event types are `run_started`, `request_arrived`, `cost_predicted`, `request_enqueued`,
`request_selected` (`data.reason` is `overdue` or `shortest_estimate`), `inference_started`,
`inference_completed`, `request_failed`, `queue_snapshot` (`data.order` is the policy's full
service order after every queue change) and `run_completed` (`data.summary`).

`timestamp_ms` is engine time: virtual in simulation, ms since run start in real runs. In real
runs, arrivals during a job are written when the job ends, with their true arrival timestamp, so
sort by `(timestamp_ms, seq)`. `read_events()` and the dashboard both do.

## Running it (from the repo root, with `.venv` active)

```bash
python -m pytest                                   # whole repo, ML + datagen + scheduler
python -m pytest tests/test_scheduler_*.py         # scheduler only

python -m scheduler simulate --workload head_of_line --policy sejf --events-out /tmp/ev.jsonl
python -m scheduler compare                        # FIFO vs SEJF vs Adaptive, n = 60 per workload
python -m scheduler compare --n 1000 --out docs/sim_results_n1000.md
python -m scheduler sweep                          # Adaptive threshold x load
python -m scheduler export-ui                      # refresh ui/public/sim/*.json for the dashboard

# Wall-clock run with the realtime mock backend (no GPU). Writes scheduler_runs/<name>/,
# which the dashboard's "Recorded / live runs" tab shows live.
python -m scheduler run --backend mock --workload head_of_line --n 60 --policy adaptive --speedup 5

cd ui && npm install && npm run dev                # http://localhost:3000
```

The workloads are `mostly_short` (80/20), `balanced` (50/50), `mostly_long` (20/80),
`head_of_line` and `bursty`. All are seeded (`--seed`, default 42), and the arrival rate comes
from `--load` (default 0.95). Their lengths are synthetic, and every output is labelled
SIMULATED or MOCK.

## Real experiment and merge

- **Runbook**, from finished labels to the recorded demo, with exact commands:
  [`docs/RUNBOOK_REAL_EXPERIMENT.md`](../docs/RUNBOOK_REAL_EXPERIMENT.md)
- **Merge plan**, covering file ownership, conflict notes and GPU-machine commands:
  [`docs/MERGE_PLAN.md`](../docs/MERGE_PLAN.md)

The real-experiment pieces:

| Piece | Where |
| --- | --- |
| Fixed arrival trace that every policy replays (`make-manifest`, `run --manifest`) | `scheduler/manifest.py` |
| `MAX_WAIT = 3.0 x median service time` (`measure-service`, `run --median-service-from`) | `scheduler/max_wait.py` |
| Like-for-like comparison and checks (`report`) | `scheduler/__main__.py` |
| Held-out prompts from a trained predictor (`prompts-from-split`) | `scheduler/__main__.py` |
| Tiger Data: non-blocking sink (`run --tiger`), `tiger-check`, `tiger-init`, `tiger-import`; schema | `scheduler/tigerdata.py`, `scheduler/tigerdata_schema.sql` |
