# Orchestrator — InferenceScheduler26

Post-integration state. ML detail in `STATUS.md`; scheduler detail in `scheduler/README.md`,
`ui/README.md`, `docs/`. Measured results in `docs/REAL_RESULTS.md` and `STATUS.md`.

Repo root: the directory containing this file. Commands below assume you are in it.
Remote: `https://github.com/efazman/InferenceScheduler26.git`

```
STATE              INTEGRATED - real end-to-end experiment measured
MERGE              ab52ef9  (--no-ff, 0 conflicts, 65 files, 8747 insertions, 0 deletions)
HEAD               7243e9b  master == origin/master, tree clean
TESTS              185 passed, 1 skipped
FRONTEND           builds clean, serving real runs
TIGER DATA         2694 rows, all measurement=real
BLOCKERS           none
NEEDS_HUMAN        visual dashboard check (~5 min) + record the demo
DEMO CAVEAT        the scripted "Adaptive = best of both" beat is NOT what the data shows - see 4
```

---

## 1. What is done

| Phase | Result |
| --- | --- |
| Preserve GPU work | `backup/gpu-pre-merge` at `f4a6923`; committed `960a572`; pushed |
| Fetch Mac branch | tip **`f9f997a`**, confirmed current by fetch |
| Independent dry run | `merge-tree` vs the *new* HEAD `960a572`: **exit 0, 0 conflicting paths** |
| Merge | **`ab52ef9`**, `--no-ff` |
| Python tests | **185 passed, 1 skipped** (skip = Tiger live test behind `TIGER_INTEGRATION=1`) |
| Frontend build | clean — Next.js 16.3.8, TypeScript clean, node v24.19.0 |
| Real predictor | `distilbert:artifacts\maxlen_512` via `DistilBertPredictor`, **3.06 ms** |
| llama-server | healthy, `total_slots: 1` (**K=1**), build `b11381-836d57176` |
| Derived MAX_WAIT | **14,079 ms** = 3.0 × median service 4,693 ms (n=3001) |
| Real experiments | 4 runs × 112 requests, **0 failures**, manifest `dc37a5a86f129b1e` |
| Tiger Data | **2,694 rows, all `measurement=real`**, import idempotent |
| Dashboard | serving real runs; `/api/tiger` returns metrics matching local summaries |

Commit chain: `f4a6923` → `960a572` (ML) → `ab52ef9` (merge) → `0d5e450` (drivers + results)
→ `324a9bc` (`.sh` line endings) → `7243e9b` (real results + finding).

---

## 2. Live processes

| What | PID | Note |
| --- | --- | --- |
| `llama-server` | **1420** | keep for any further real runs and for the baseline's `/tokenize` |
| dashboard (`next start`) | **33060** | http://localhost:3000 |

Nothing is generating. Stop either with `Stop-Process -Id <pid>`.

---

## 3. Real measured results

112 held-out test-split prompts, one frozen manifest, real predictor, real Llama 3.1 8B Q4_K_M,
K=1. `python -m scheduler report` validated same manifest, backend, generation settings, threshold,
request set, arrival times and **predictor output to 0.0000 tokens** before comparing — only
execution order differed.

| Metric | FIFO | SEJF | Adaptive 3× | Adaptive 5× |
| --- | --- | --- | --- | --- |
| Mean latency (s) | 16.65 | **13.36** | 15.75 | 16.34 |
| p50 latency (s) | 12.27 | **8.78** | 10.62 | 10.58 |
| p95 latency (s) | 48.70 | 51.65 | **44.18** | 46.20 |
| p99 latency (s) | 57.76 | 80.56 | **51.80** | 53.71 |
| Mean queue wait (s) | 10.81 | **7.66** | 10.13 | 10.62 |
| Max queue wait (s) | 54.75 | 72.17 | **48.78** | 50.82 |
| Short-req mean latency (s) | 14.02 | **7.03** | 13.66 | 14.23 |
| Long-req mean latency (s) | 18.23 | 17.45 | **17.11** | 17.70 |
| Long-req max wait (s) | 54.32 | 72.17 | **48.37** | 50.34 |
| Throughput (req/min) | 7.94 | 7.94 | 7.95 | 7.92 |
| Starved, own threshold | 28 | 18 | 35 | 25 |

Per-run thresholds differ (14.08 s for the first three, 23.47 s for 5×), so **own-threshold
starvation counts are not comparable.** Recomputed from raw queue waits at a common threshold:

| Starved at | FIFO | SEJF | Adaptive 3× | Adaptive 5× |
| --- | --- | --- | --- | --- |
| > 14.08 s | 28 | **18** | 35 | 34 |
| > 23.47 s | 19 | **9** | 20 | 25 |

**Throughput is flat at ~7.94 req/min across all four.** Expected at K=1 with no preemption —
reordering cannot create decode capacity. **No throughput improvement is claimed**; it is a sanity
check that no run stalled.

---

## 4. The demo script needs one beat reframed

`docs/RUNBOOK_REAL_EXPERIMENT.md` §17 beat 5 says Adaptive gives "short-job preference, with
overdue requests served oldest-first, so waits are bounded". **The measured data supports the
second half only.**

Adaptive is best of all four on **every** tail metric — p95, p99, max queue wait, long-request max
wait — beating both FIFO *and* SEJF. But its short-request latency (13.66 s) is only 2.6% better
than FIFO, against SEJF's 7.03 s. It does not retain SEJF's short-job win.

**This is not a bug and not a threshold-tuning problem.** Both were checked:

- The policy is implemented correctly: overdue oldest-first, then SEJF among the rest.
- It is **not** running in FIFO mode. By its own selection reasons it chose by shortest-estimate
  **69%** of the time at 3× (77/112) and **78%** at 5× (87/112). Its p50 queue wait (3.43 s) is
  far better than FIFO's (5.94 s).
- Cause: at K=1 with no preemption, each overdue promotion of a long request blocks every short
  request behind it for that request's **full** service time, and service is heavy-tailed here
  (p50 output 407 tokens, p90 918). A few expensive promotions erase the short-job gain in the mean.
- **A 5× threshold was measured and made everything slightly worse** — fewer promotions (25 vs 35)
  but each later, with a larger backlog behind it. 3× is the better operating point.

**Honest framing, which is a stronger result than the script:** a genuine three-way tradeoff.
SEJF optimises the body of the latency distribution; Adaptive optimises the tail; FIFO does neither
well. Bounded wait at K=1 without preemption has a real, measured price.

An earlier version of this file blamed "mostly FIFO-like mode". That was wrong and is retracted —
the selection-reason counts above refute it.

---

## 5. Configuration of record

| | |
| --- | --- |
| Predictor | `distilbert:artifacts\maxlen_512` — test MAE 193.1 vs baseline 241.5, severe under 0.089 vs 0.143, 3.06 ms (0.07% of service time) |
| Fallback | `artifacts\maxlen_128` — worse MAE, half the severe underprediction. Unused; real runs showed no tail pathology attributable to the predictor |
| Model | Llama 3.1 8B Instruct Q4_K_M, `b11381-836d57176-win-cuda-12.4-x64` |
| Decoding | temp 0.7, top_p 0.9, `max_new_tokens` 1024, fixed system prompt (sha `75357d68…`) |
| Manifest | `dc37a5a86f129b1e`, 112 requests, 7,000 ms mean inter-arrival, seed 42 |
| MAX_WAIT | derived 3.0 × median, **14,079 ms**; recorded in `run_started` and `summary.json` |
| Runs on disk | `scheduler_runs\real-{fifo,sejf,adaptive,adaptive-5x}` |

---

## 6. Dataset caveats any write-up must carry

1. **Seeds are not reproducible on this backend.** llama-server reuses the KV/prompt cache by
   default; one request recorded as 1024/`length` in the base run deterministically yields
   539/`stop` with `cache_prompt: false`. The four runs per prompt are still four real samples, so
   the p90 target is valid — but do **not** call the dataset reproducible.
2. **That killed the 2048 extension pass.** A cache-affected censored generation cannot be
   uncensored by re-running it. Its 14 records are quarantined under `data/labels/_quarantine/`
   and never reached a label set.
3. **9.87% of base runs are right-censored** (296/3,000) across 100 of 750 prompts, all flagged
   `has_censored_target`.
4. **750 prompts, not 2000** (cut for time); predictor test set 112 examples.
5. Serving used `max_new_tokens` 1024, so measured output lengths are capped at 1024.

---

## 7. What remains

1. **Visual dashboard check (~5 min, needs human eyes).** APIs verified serving `measurement=real`
   and pages return HTTP 200, but the rendered queue reordering, execution timeline, REAL badge and
   comparison table have **not** been visually confirmed.
   → http://localhost:3000/?mode=runs&run=real-adaptive
2. **Record the demo.** `docs/RUNBOOK_REAL_EXPERIMENT.md` §17 and `docs/DEMO_FLOW.md`, with beat 5
   reframed per section 4 above.
3. Optional, not needed for the demo: resume label generation toward 1000/2000; a
   `cache_prompt: false` regeneration for a reproducible dataset (~8 h); more arrival-rate points.

---

## 8. Not implemented, by instruction

K>1 concurrency, multi-GPU, new scheduler policies, dashboard redesign, retraining, max_length
sweeps, label-set regeneration.

---

## 9. Integration issues found and fixed this session

| Issue | Resolution |
| --- | --- |
| `powershell.exe -File` bound only the first token of a `[string[]]` parameter, so the driver ran FIFO only then reported "all policies done" | policies passed as a comma-separated string, split in-script, with a guard that throws on an empty list |
| A first FIFO run overlapped a background Node install, biasing that run's latencies specifically | discarded at 225 events and re-run on an idle machine (CPU 7%, GPU 0%) |
| `winget install` hung on a UAC prompt it could not display non-interactively (exit 255) | Node installed with elevation; dashboard then built clean |
| `core.autocrlf` would rewrite `scripts/*.sh` to CRLF and break them on a fresh clone | `.gitattributes` with `*.sh text eol=lf` (`324a9bc`) |
| `npm install` under npm 11 stripped 4 `"peer": true` markers from the Mac's lockfile | reverted; their lockfile left pristine |
| Checkpoint summaries reported 0 censored runs for assembled labels (field renamed) | `_truncated_runs()` fallback in `datagen/make_checkpoint.py` |
