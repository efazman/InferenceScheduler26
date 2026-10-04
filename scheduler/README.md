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
| `events.py` | Flat event schema plus sinks: `LocalJsonlEventSink`, `InMemoryEventSink`, `FanoutSink`, and `TigerDataEventSink` (placeholder). |
| `metrics.py` | Percentiles, queue wait, throughput, starvation count, short/long split. |
| `workloads.py` | Five seeded synthetic workloads, plus `workload_from_prompts` for real prompts. |
| `simulator.py` | Engine + `VirtualClock` + mock predictor/backend. Service = 60 ms + 13.3 ms/token. |
| `__main__.py` | CLI: `simulate`, `compare`, `sweep`, `export-ui`, `prompts-from-split`, `run`. |
| `../ui/` | Next.js dashboard (see `ui/README.md`). |

## The three policies

At every decision point (the backend is idle and the queue isn't empty), with `wait = now − arrival_time`:

| Policy | Picks |
| --- | --- |
| FIFO | Lowest `(enqueue_time, request_id)` |
| SEJF | Lowest `(predicted_tokens, enqueue_time, request_id)` |
| **Adaptive** | If any request has `wait ≥ MAX_WAIT`: the **oldest** of those (`arrival_time, enqueue_time, request_id`). Otherwise the same as SEJF. |

`MAX_WAIT` defaults to 15 s (`scheduler/config.py: DEFAULT_MAX_WAIT_MS`). Override it with
`--max-wait-ms` on every CLI command.

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

## Integration with the RTX 3060 Ti branch

**Do not run any of this while label generation is still live** (STATUS.md, "nothing heavy on
this box").

1. **Merge.** On the 3060 Ti machine, once generation and training are done:
   ```powershell
   git fetch origin
   git checkout master; git pull
   git merge --no-ff origin/feature/scheduler-ui-parallel
   .\.venv\Scripts\python.exe -m pytest
   ```
   This branch adds only new files, apart from `.gitignore` (two appended lines). See
   *Expected conflicts*.
2. **Plug in the winning predictor.** Nothing changes in code. Pass a spec:
   `--predictor distilbert:artifacts\<winning run dir>` or `--predictor baseline:artifacts\<dir>`,
   using the directory `ml.compare_runs` picked. The baseline's prompt-token counter is read from
   its `baseline.json`. A `llamacpp:` counter needs llama-server up, which it will be.
3. **Point at llama-server.** Use the server STATUS.md already uses: `-c 4096 -ngl 99 -np 1`,
   where one slot means our queue order is the execution order. `LlamaCppBackend` reads
   `LLM_BACKEND_URL` (default `http://127.0.0.1:8080`), or pass `--url`. Decoding settings come
   from `datagen/config.py`, so real runs generate exactly the way the labels were produced.
4. **Build a real workload from held-out prompts**, so the predictor is never scored on prompts it
   trained on:
   ```powershell
   .\.venv\Scripts\python.exe -m scheduler prompts-from-split --artifacts artifacts\<winner> `
       --split test --out data\scheduler\test_prompts.jsonl
   ```
5. **Run every policy on the same prompts and arrival schedule.** Same `--seed`, so arrival times
   and per-request generation seeds match:
   ```powershell
   foreach ($p in "fifo","sejf","adaptive") {
     .\.venv\Scripts\python.exe -m scheduler run --backend llamacpp --predictor distilbert:artifacts\<winner> `
       --prompts-file data\scheduler\test_prompts.jsonl --policy $p --mean-interarrival-ms 7000 `
       --run-name real-$p
   }
   ```
   Pick `--mean-interarrival-ms` for the load you want: mean service time ÷ load. Mean service
   time is about 6 s at the 1024 cap (STATUS.md), so 7000 ms gives roughly 0.85 load. Set
   `--max-wait-ms` from `docs/SIM_RESULTS.md`. Each run writes
   `scheduler_runs/real-<policy>/{events.jsonl, summary.json}`, fsynced per event.
6. **Connect Tiger Data.** Implement `TigerDataEventSink` in `scheduler/events.py` (the planned
   schema is in its docstring; credentials come from `TIGER_DATA_DSN`), then in `cmd_run` wrap the
   sink as `FanoutSink(LocalJsonlEventSink(...), TigerDataEventSink())`. The local JSONL stays the
   source of truth, and a database outage can't break a run.
7. **Demo.** Run `cd ui && npm run dev`. The *Simulation* tab shows the reordering story on
   synthetic workloads. The *Recorded / live runs* tab shows the real runs from step 5 (live while
   they execute, if started during the demo). For a real-data side-by-side, use
   `python -m scheduler compare`-style tables built from the three `summary.json` files.

## Expected merge conflicts

- `.gitignore`: this branch appends `scheduler_runs/`. If master also appended lines, keep both.
- Nothing else. This branch doesn't touch `ml/`, `datagen/`, `scripts/`, `STATUS.md`,
  `ORCHESTRATOR_UPDATE.md`, `requirements.txt` or any data or artifact directory. It only
  *imports* `datagen.backends`, `datagen.config` and `ml.*`. If master renames
  `OpenAICompatBackend`, `LengthPredictor.load` or `PromptLengthBaseline.load`, the adapters in
  `scheduler/backends.py` and `scheduler/predictors.py` are the only places to update, and their
  tests (`test_llamacpp_*`, `test_real_predictor_adapters_*`) will catch it.
