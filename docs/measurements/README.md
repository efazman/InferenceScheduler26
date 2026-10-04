# Archived real measurements (RTX 3060 Ti)

Every measurement this project reports, committed to the repository so the results outlive the
Tiger Cloud free trial, the local `scheduler_runs/` directory, and the machine itself.

These are **real measurements**, not simulations: a real DistilBERT predictor and real
Llama 3.1 8B Q4_K_M generations on one RTX 3060 Ti. Every record carries `"simulated": false`
and `"measurement": "real"`.

## Provenance

| | |
| --- | --- |
| Machine | NVIDIA RTX 3060 Ti 8 GB, driver 591.86, Windows 11 |
| Model | Meta Llama 3.1 8B Instruct, **Q4_K_M** GGUF (`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`, sha256 `7b064f58…33557c`) |
| Runtime | llama.cpp `llama-server` build **b11381**, commit **836d57176**, CUDA 12.4, `-c 4096 -ngl 99 -np 1` |
| Concurrency | **K = 1** (one slot), so queue order equals execution order |
| Decoding | temperature 0.7, top_p 0.9, `max_new_tokens` 1024, system prompt `"You are a helpful assistant."` (sha256 `75357d68…`) |
| Predictor | `distilbert:artifacts/maxlen_512`, 2.8–3.1 ms per call on this GPU |
| Manifest | `dc37a5a86f129b1e` — 112 held-out test-split prompts, 7,000 ms mean inter-arrival, seed 42 |
| Derived MAX_WAIT | 14,079 ms = 3.0 × median service 4,693 ms, from 3,001 real label generations |
| Failures | **0** across all four runs |

## The results

| Metric | FIFO | SEJF | Adaptive 3× | Adaptive 5× |
| --- | --- | --- | --- | --- |
| Mean latency (s) | 16.65 | **13.36** | 15.75 | 16.34 |
| p50 latency (s) | 12.27 | **8.78** | 10.62 | 10.58 |
| p95 latency (s) | 48.70 | 51.65 | **44.18** | 46.20 |
| p99 latency (s) | 57.76 | 80.56 | **51.80** | 53.71 |
| Mean queue wait (s) | 10.81 | **7.66** | 10.13 | 10.62 |
| Max queue wait (s) | 54.75 | 72.17 | **48.78** | 50.82 |
| Short-request mean latency (s) | 14.02 | **7.03** | 13.66 | 14.23 |
| Long-request mean latency (s) | 18.23 | 17.45 | **17.11** | 17.70 |
| Long-request max wait (s) | 54.32 | 72.17 | **48.37** | 50.34 |
| Throughput (req/min) | 7.94 | 7.94 | 7.95 | 7.92 |
| Starved, own threshold | 28 | 18 | 35 | 25 |

Short/long split: 42 short / 70 long, by workload label else actual output ≥ 300 tokens.
Thresholds: 14.08 s for the first three runs, 23.47 s for Adaptive 5×.

**Starvation counts at each run's own threshold are not comparable.** Recomputed from the raw
queue waits at a common threshold:

| Starved at | FIFO | SEJF | Adaptive 3× | Adaptive 5× |
| --- | --- | --- | --- | --- |
| > 14.08 s | 28 | **18** | 35 | 34 |
| > 23.47 s | 19 | **9** | 20 | 25 |

Adaptive's selection reasons, which rule out "it degenerated to FIFO":

| Run | chose by shortest-estimate | chose because overdue |
| --- | --- | --- |
| Adaptive 3× | 77 / 112 (**69%**) | 35 / 112 (31%) |
| Adaptive 5× | 87 / 112 (**78%**) | 25 / 112 (22%) |

**Throughput is flat at ~7.9 req/min across all four runs. No throughput improvement is claimed** —
at K=1 with no preemption, reordering cannot create decode capacity. It is reported only as a
sanity check that no run stalled.

## Predictor selection (held-out, 112 prompts)

| | DistilBERT | Prompt-length baseline |
| --- | --- | --- |
| Test MAE | **193.1 tokens** | 241.5 tokens |
| p90 absolute error | 415.2 | 483.9 |
| Underprediction rate | 0.420 | 0.473 |
| Severe underprediction | **0.089** | 0.143 |
| Inference latency | 2.82 ms (0.06% of service) | — |

Input-length experiment (only `max_length` varied, everything else held fixed):

| max_length | MAE | Severe underprediction | Mean signed error |
| --- | --- | --- | --- |
| 128 | 204.78 | **0.045** | +43.2 |
| 256 | 193.61 | 0.080 | +5.0 |
| 512 | **193.14** | 0.089 | +12.8 |

Learning curve at `max_length` 512 — still improving at 750 prompts:

| Prompts | n_train | n_test | DistilBERT MAE | Baseline MAE |
| --- | --- | --- | --- | --- |
| 250 | 175 | 38 | 246.7 | 280.1 |
| 500 | 350 | 75 | 217.2 | 269.0 |
| 750 | 525 | 112 | **193.1** | 241.5 |

## Label dataset measured on this machine

| | |
| --- | --- |
| Source | LMSYS-Chat-1M → 1,000,000 rows read |
| After filtering | 316,816 clean prompts (English, single-turn, deduped, moderation-filtered) |
| Sampled | 750 prompts, category-balanced, seed 42 |
| Generations | **3,000** (750 × 4 seeds), 0 failures |
| Decode speed | ~75.7 tok/s median, full GPU offload |
| Median service time | **4,693 ms** (p10 565 ms, p90 13,592 ms) |
| Output tokens | median 407, p90 918 — heavy-tailed |
| Target | p90 of the 4 runs per prompt (conservative: underprediction is the harmful direction) |
| Right-censored | **296 / 3,000 runs (9.87%)** hit the 1024 cap, across 100 of 750 prompts |

## Known caveat: not seed-reproducible

`llama-server` reuses its KV/prompt cache between requests by default, so the same prompt and seed
is **not** reproducible. A request recorded at 1024 tokens / `finish_reason: length` in the base run
deterministically returns 539 tokens / `stop` when re-issued with `"cache_prompt": false`, which is
also deterministic across token caps.

Consequences, recorded honestly rather than worked around:

- The four generations per prompt are four *real* samples, so the p90 target is valid — but this
  dataset must not be described as seed-reproducible.
- A planned pass to un-censor the 1024-capped labels at a 2048 cap was **abandoned**: a
  cache-affected generation cannot be re-run to recover its true length (42% of re-runs came back
  *shorter* than the cap they had exceeded). Its 14 partial records were quarantined and never
  reached a label set.
- A reproducible dataset would need `"cache_prompt": false` on every request and a fresh run.

## Files

```
docs/measurements/
    real-fifo/        events.jsonl, summary.json
    real-sejf/        events.jsonl, summary.json
    real-adaptive/    events.jsonl, summary.json   (MAX_WAIT 3x = 14.08 s)
    real-adaptive-5x/ events.jsonl, summary.json   (MAX_WAIT 5x = 23.47 s)
```

`events.jsonl` is one fsynced JSON object per event: `run_started`, `request_arrived`,
`cost_predicted`, `request_enqueued`, `queue_snapshot`, `request_selected`, `inference_started`,
`inference_completed`. Each carries `predicted_output_tokens`, `actual_output_tokens`,
`queue_wait_ms`, `service_time_ms`, `end_to_end_latency_ms`, `uncertainty` and `queue_depth`.

**`manifest.json` is deliberately excluded.** It embeds LMSYS prompt text, which this repository
does not redistribute. These files carry only `prompt_chars` and `prompt_tokens` counts.
`scripts/check_archive_safe.py` enforces that — it walks every value in the archive and fails on
any long string, text-shaped field name, or stray manifest:

```bash
python scripts/check_archive_safe.py     # walks ~82k values; must print OK
```

Regenerate the manifest deterministically (same `manifest_id`) with:

```bash
python -m scheduler prompts-from-split --artifacts artifacts/maxlen_512 --split test \
    --out data/scheduler/test_prompts.jsonl
python -m scheduler make-manifest --prompts-file data/scheduler/test_prompts.jsonl \
    --mean-interarrival-ms 7000 --seed 42 --out data/scheduler/real_manifest.json
```

## Restoring these into Tiger Data

`tiger-import` reads exactly these two files per run, so the archive is a complete backup of the
2,694 rows that were in `scheduler_events`. After setting credentials:

```bash
python -m scheduler tiger-import docs/measurements/real-fifo \
    docs/measurements/real-sejf docs/measurements/real-adaptive
python -m scheduler tiger-check     # expect rows_by_measurement = {"real": 2694}
```

Imports are idempotent — one row per `run_id` + `seq`, so re-running adds nothing twice.

## Regenerating the comparison table

```bash
python -m scheduler report docs/measurements/real-fifo docs/measurements/real-sejf \
    docs/measurements/real-adaptive --out docs/REAL_RESULTS.md
```

`report` refuses to compare runs that differ in anything but policy. It verified these three share
a manifest, backend, generation settings, threshold, request set, arrival timestamps, and
**predictor outputs matching to 0.0000 tokens**.
