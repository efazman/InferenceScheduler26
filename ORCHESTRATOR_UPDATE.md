# Orchestrator — InferenceScheduler26

Current coordination state only. History, designs, rationale and the changelog live in `STATUS.md`.
Repo: `C:\Users\efazr\Desktop\Mhacks26\InferenceScheduler26` (**not** `~/MHacks26`).

```
STATE              label_generation_running
SCOPE              TARGET CUT 2000 -> 750 PROMPTS (time budget). Resumable to 2000 later.
BLOCKERS           none
NEEDS_HUMAN        no
NEXT_TRIGGER       auto-stop at 750 prompts -> extension -> assembly -> Phase 12
ETA_ALL_DONE       ~5.7 h from 2026-10-04T01:48Z  (base 3.3 + ext 1.9 + ML 0.5)
TESTS              95 passed, 0 skipped
```

---

## 1. Live processes — do not kill

| What | PID | Verify |
| --- | --- | --- |
| `llama-server` | **1420** | `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/health` -> 200 |
| generation runner | **28228** | `Get-Process -Id (Get-Content logs\generation.pid)` |
| auto-stop watcher | **2592** | `Get-Content logs\stop_at.log -Tail 3` |

The watcher stops the base run the moment 750 prompts are complete, then leaves `llama-server`
up for the extension pass. Cancel it with `Stop-Process -Id 2592` to let the run continue to 2000.

Both ALIVE as of this writing. PIDs also on disk: `logs/llama-server.pid`, `logs/generation.pid`.

---

## 2. Hard constraints while the base run is live

1. **One generator only.** Two writers would interleave appends into the same JSONL files and
   corrupt the label set. Before starting anything that generates, confirm PID 28228 is gone.
2. **Nothing heavy on this box — GPU *or* CPU.** CPU-only training plus test suites already cost a
   1.35 h generator stall (forensics in `STATUS.md`). The generator is a single-threaded Python
   loop that fsyncs every record; starving it halts the pipeline with no error raised.
3. **Do not touch the base run's settings, output dir, seeds, or `max_new_tokens`.** A restart
   re-validates `datagen/config.py` against the on-disk `generation_config.json`; any mismatch
   fails every restart with `ConfigMismatch` and silently stops the run.
4. **`data/labels/llama31_8b_q4km/` is read-only from here on.** Extension output and final labels
   go to separate directories.

---

## 3. Scope decision: 500 prompts, not 2000

The full 2000-prompt plan costs **16.7 h** end to end, which exceeded the available budget. Target
cut to **500 prompts = 3.4 h**. Measured comparison at 10.81 generations/min:

| Target | +generations | base | extension | ML | **total** |
| --- | --- | --- | --- | --- | --- |
| **500** | 1,192 | 1.8 h | 1.2 h | 0.4 h | **3.4 h** |
| 1000 | 3,192 | 4.9 h | 2.3 h | 0.6 h | 7.8 h |
| 2000 | 7,192 | 11.1 h | 4.6 h | 1.0 h | 16.7 h |

**Nothing is discarded and nothing was restarted.** 500 was already a planned checkpoint, the
subset file is still `subset_2000.jsonl`, and `NUM_PROMPTS` was deliberately left at 2000. Resuming
to 1000 or 2000 later is `.\scripts
un_generation.ps1 -Detach` — completed prompts are skipped,
and both the extension pass and the assembly are idempotent, so a later top-up re-runs cleanly.

**Cost of the cut, stated plainly:** 500 prompts gives a **75-example test set** (350 train / 75 val
/ 75 test). Severe-underprediction rate moves 1.3 points per single test example, so a 2–3 example
difference could flip the DistilBERT-vs-baseline verdict. That is enough to choose a predictor and
proceed to the scheduler; it is thin for a strong accuracy claim. The learning curve degrades to
two points (250, 500).

## 4. Progress — snapshot, refresh before acting

Taken 2026-10-04 01:45Z. Re-read rather than trusting these numbers:

```powershell
.\.venv\Scripts\python.exe -m datagen.status
.\.venv\Scripts\python.exe -m datagen.make_checkpoint --list
.\.venv\Scripts\python.exe -m datagen.extend_censored --backend llamacpp --dry-run
Get-Content logs\generation.log -Tail 30 -Wait
```

| Metric | Value |
| --- | --- |
| Prompts complete | **225 / 750** (target cut) |
| Generations | ~900 / 3,000 |
| Rate (recent) | ~10.8 / min |
| ETA to 750 | **~3.3 h** |
| Failures | **0 attempts, 0 permanent** |
| Base truncation at 1024 | ~9.5% |
| Idle so far | 1.35 h (one 80.8-min stall) |
| Checkpoints cut | none yet; 250, 500 and 750 will all cut |

Read `generations_per_min_recent`, not `generations_per_min` — the latter averages in idle time and
understates throughput. `eta_basis` reports which rate the ETA used.

Truncation is tracking **~10%**, not the 16% quoted in earlier briefs. At 750 prompts expect
roughly **310 censored runs** to extend (~1.9 h). They are the *longest* generations in the set and
rerun at 2048, so wall-clock is well above `count x 13 s`.

---

## 5. Queued pipeline — strict order, after the base run stops

```powershell
# 0. confirm done and the generator is gone
.\.venv\Scripts\python.exe -m datagen.status
Get-Process -Id (Get-Content logs\generation.pid) -ErrorAction SilentlyContinue   # expect nothing

# 1. size the extension work (read-only, safe any time)
.\.venv\Scripts\python.exe -m datagen.extend_censored --backend llamacpp --dry-run

# 2. extension pass at 2048 (resume = same command; idempotent)
.\.venv\Scripts\python.exe -m datagen.extend_censored --backend llamacpp

# 3. assemble final labels
.\.venv\Scripts\python.exe -m datagen.assemble_final_labels

# 4. checkpoints from the ASSEMBLED labels
.\.venv\Scripts\python.exe -m datagen.make_checkpoint `
    --labels data\labels\llama31_8b_q4km_final\labels_final.jsonl `
    --out-root data\labels\llama31_8b_q4km_final\checkpoints

# 5. input-length experiment (GPU now free)
.\scripts\run_maxlen_experiment.ps1 -Labels data\labels\llama31_8b_q4km_final\labels_final.jsonl

# 6. learning curve at the winning max_length
.\scripts\run_learning_curve.ps1 -MaxLength <winner>

# 7. success bar
.\.venv\Scripts\python.exe -m ml.compare_runs artifacts\curve_500 artifacts\curve_1000 artifacts\curve_2000
```

Recovery, safe from any state (completed work is skipped, never redone):

```powershell
.\scripts\start_llama_server.ps1 -Background   # only if the server died
.\scripts\run_generation.ps1 -Detach
```

**Report after steps 2–3:** base truncation count+rate, extension truncation count+rate, prompts
affected, prompts still carrying censored values.

Step 2 **refuses to start** while `logs/generation.pid` names a live process
(`--allow-concurrent-base` overrides, deliberately). Step 5 warns in the same situation.

---

## 6. Artifacts and when they are claimable

| Path | Ready |
| --- | --- |
| `data/lmsys/subset_2000.jsonl` | ✅ now |
| `data/labels/llama31_8b_q4km/labels.jsonl` | growing — **do not train on it** |
| `data/labels/llama31_8b_q4km_ext2048/extended_runs.jsonl` | after step 2 |
| `data/labels/llama31_8b_q4km_final/labels_final.jsonl` | after step 3 |
| `.../llama31_8b_q4km_final/checkpoints/checkpoint_00250/labels_250.jsonl` | after step 4 |
| `.../llama31_8b_q4km_final/checkpoints/checkpoint_00500/labels_500.jsonl` | after step 4 |
| `.../llama31_8b_q4km_final/checkpoints/checkpoint_00750/labels_750.jsonl` | after step 4 |
| `artifacts/distilbert_length_predictor/` | ✅ exists but trained on **synthetic** data — metrics meaningless, do not report |

Checkpoints are nested (cp500 ⊂ cp1000 ⊂ cp2000), so the learning curve measures dataset size
alone. Existing checkpoints are never overwritten.

---

## 7. Measurement-validity invariants — do not violate

1. 1024 and 2048 measurements are **never silently mixed**. Every final record carries its per-run
   `length_source` plus both base and extended lengths.
2. Censoring is **never hidden**. A run still capped at 2048 keeps `still_censored: true`.
   `has_censored_target` is set only when the p90 *actually reads* a censored value.
3. Training reads **only stable files**, never the growing `labels.jsonl`.
4. Controlled comparisons vary **one** thing: the max_length experiment varies max_length only, the
   learning curve varies size only. Early stopping stays off by default so epoch counts match.
5. **CPU predictor latency is not final.** Success-bar criterion 3 must be re-measured on GPU after
   generation completes; `eval_results.json` records the device used.

---

## 8. Locked decisions — do not re-litigate

| Decision | Value |
| --- | --- |
| Dataset size | **750 prompts** (cut from 2000 for time); resumable to 1000/2000 later |
| Base `MAX_NEW_TOKENS` | **1024** — irreversible for this dataset; a change needs a new output dir and a fresh run |
| `EXTENDED_MAX_NEW_TOKENS` | **2048**, not 1536 (context fits: 2000 + 2048 = 4048 < 4096) |
| `max_length` default | **still 128**, pending the measured 128/256/512 experiment |
| `epochs` default | **still 3** for the first controlled run |
| Early stopping | **opt-in, off by default** |
| Censored-target prompts | **kept and flagged**, not dropped |

Success bar (unchanged): DistilBERT is selected only if it beats baseline test MAE, beats or ties
baseline severe-underprediction rate, and keeps inference overhead under 5% of median Llama service
time. Criterion 3 already measured at ~0.05% on GPU (2.79 ms vs multi-second service time). If
DistilBERT loses on the others, the baseline ships through the same `predictor.predict()`
interface — an acceptable outcome, not a failure.

---

## 9. Do not implement yet

FIFO / SJF / adaptive scheduler, Tiger Data, dashboard, frontend, EGTP, RAG, K>1 concurrency,
multi-GPU. All come after the predictor choice is finalised.

---

## 10. Open items needing a human

1. **Whether to exclude `has_censored_target` prompts from training.** Flag is available on every
   final record; the decision is open. Count will be in `assembly.summary.json` after step 3.
2. **Nothing is committed to git.** 11 modified + 6 new files, no commit requested yet.

Not yet run, so no results exist: the extension pass, final assembly, and any Phase 12 training on
real labels.
