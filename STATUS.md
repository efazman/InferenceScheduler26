# Project status — real label generation RUNNING

Updated 2026-10-03 ~19:55 local. Machine: Windows 11 Pro 26200, RTX 3060 Ti 8 GB.

**Scope cut 2026-10-04: target reduced 2000 -> 500 prompts** for time budget (16.7 h -> 3.4 h end
to end). Nothing was restarted or discarded — 500 was already a planned checkpoint, `NUM_PROMPTS`
stays 2000, and `scripts/stop_at.ps1` stops the run at 500. Resuming to 1000/2000 later is
`.\scripts\run_generation.ps1 -Detach`; the extension pass and assembly are both idempotent.
Cost: a 75-example test set, so the DistilBERT-vs-baseline verdict is statistically thin.

Base run is in flight and untouched. The censored-output extension pass, final-label assembly and
the Phase 12 experiment harness are built and tested, and all wait on the base run finishing — see
*Censored-output handling* and *Phase 12 harness* below.

Project context: `General Project Context/adaptive_llm_scheduler_project_context.md`.
Pipeline docs: `datagen/README.md`, `ml/README.md`. The scheduler itself is not started yet, by design.

## Phase status

| Phase | State |
| --- | --- |
| 1 Machine / repo audit | **done** — 71 tests pass |
| 2 Python environment | **done** — `.venv` 3.11.16, CUDA torch verified on GPU |
| 3 LMSYS dataset | **done** — 1M rows → 316,816 clean → 2,000 selected |
| 4 llama.cpp + CUDA | **done** — prebuilt b11381 CUDA 12.4, 76 tok/s proves offload |
| 5 Llama 3.1 8B Q4_K_M | **done** — sha256 verified |
| 6 llama-server | **running** — PID 1420, kept up for the baseline's Llama tokenizer |
| 7 Smoke test | **done** — 10 real prompts × 4 seeds, 40/40, twice (512 and 1024) |
| 8 512-cap decision | **decided: raised to 1024** (see below) |
| 9 Real label generation | **done** — 750/750 prompts, 3,000 runs, 0 failures |
| 10 Checkpoints | **done** - 250 / 500 / 750 cut from assembled labels |
| 11 Unattended run | **done** — detached, auto-restart, auto-checkpoint |
| 8b Censored-output extension | **ABANDONED** - premise invalid (prompt-cache nondeterminism); records quarantined |
| 8c Final-label assembly | **done** - 750 prompts, base-only, censoring flagged |
| 12 Training on checkpoints | **DONE** — DistilBERT selected, all 3 criteria passed (see RESULTS) |

## Phase 8 decision: MAX_NEW_TOKENS 512 → 1024

Measured on the same 10 real LMSYS prompts × 4 seeds:

| | 512 | 1024 |
| --- | --- | --- |
| Runs hitting the cap | 13/40 = **32.5%** | 5/40 = **12.5%** |
| Prompts with ≥1 truncated run | 5/10 | 2/10 |
| Prompts with p90 target pinned at the cap | **4/10** | **1/10** |
| Mean output tokens | 362.7 | 455.6 |
| Mean latency | 4.83 s | 6.04 s |
| Projected 8000 generations | ~10.7 h | ~13.4 h |

Truncation was genuine long-form generation, not degenerate looping — the 512-token outputs were
coherent prose stopping mid-sentence. The offenders were legitimately long requests ("write an
article, 1500-2000 words", "explain addition in the style of a teen fantasy novel").

Raising the cap mattered more than the headline rate suggested: a prompt recorded as
`[512,512,512,512]` turned out to be `[1024,529,553,558]` — three of its four runs actually
finish near 550. At 512 those labels were all pinned to the same censored value, which would
have taught the predictor that a 550-token job and a 2000-token job cost the same. That is
exactly the distinction the scheduler needs.

**Decided by the project owner.** `datagen/config.py` now sets `MAX_NEW_TOKENS = 1024`. All labels
generated at 512 were discarded; nothing in the live dataset mixes the two settings (the config
check would refuse it anyway).

Context headroom is fine: 8000-char prompt cap ≈ 2000 tokens worst case + 1024 generated ≈ 3024,
inside the 4096 context. The selected subset maxes at 2,719 chars.

## Dataset

| | |
| --- | --- |
| Source | `lmsys/lmsys-chat-1m` (gated; access granted to `efazr`) |
| Raw | `data/raw/lmsys-chat-1m/`, 6 parquet shards, 1.4 GB |
| Rows read | 1,000,000 |
| Kept after filters | **316,816** |
| Dropped | 236,767 multi-turn · 222,547 non-English · 201,349 duplicate · 22,521 moderation-flagged |
| Clean file | `data/lmsys/clean_prompts.jsonl` |
| **Selected subset** | **`data/lmsys/subset_2000.jsonl`** — 2,000 unique prompts, seed 42 |
| Category counts | knowledge 400 · explanation 400 · coding 300 · summarization 300 · writing 300 · troubleshooting 300 |
| Prompt length | median 167 chars / 27 words; p90 1,519 chars; max 2,719 chars |

Every category quota was met exactly from the available pool — no shortfall redistribution.
Categories are approximate workload-balancing labels (`heuristic-v1`), not ground truth.

## Runtime and model (pinned)

| | |
| --- | --- |
| llama.cpp | `vendor/llama.cpp-b11381-cuda-12.4/`, build **b11381**, commit **836d57176**, Clang 20.1.8 |
| Source | `ggml-org/llama.cpp` release `b11381`, `llama-...-win-cuda-12.4-x64.zip` + matching cudart |
| Why CUDA 12.4 not 13.4 | unambiguously backward-compatible with driver 591.86 (reports CUDA 13.1) |
| Model | `models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` |
| Model source | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (public, ungated) |
| Size / sha256 | 4,920,739,232 B · `7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c` **verified** |
| Decode speed | 76 tok/s, full offload; 5,195 MiB / 8,192 MiB VRAM at ctx 4096 |
| Server flags | `-c 4096 -ngl 99 -np 1` — one slot, so queue order == execution order |

`runtime_version = b11381-836d57176-win-cuda-12.4-x64` is recorded in `generation_config.json`
and in every label record. A resume with a different build is **refused**, so one label set can
never mix two runtimes.

## Generation configuration (locked)

2,000 prompts × 4 generations = **8,000 generations**. temperature 0.7, top_p 0.9,
max_new_tokens **1024**, system prompt `"You are a helpful assistant."`
(sha256 `75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de`), seeds 42/43/44/45,
target = p90 of the 4 output-token counts, linear interpolation.

## Monitoring and recovery

```powershell
.\.venv\Scripts\python.exe -m datagen.status            # progress, ETA, truncation, failures
.\.venv\Scripts\python.exe -m datagen.make_checkpoint --list
Get-Content logs\generation.log -Tail 30 -Wait           # live
Get-Content logs\generation.pid                          # runner PID
nvidia-smi
```

Resume after any failure — **the same command**, from anywhere:

```powershell
.\scripts\start_llama_server.ps1 -Background   # only if the server died
.\scripts\run_generation.ps1 -Detach
```

Completed prompts are skipped and completed generations reused, so a restart never redoes
finished work and never starts from zero. Verified in practice: the runner's python child was
hard-killed mid-generation and `runs.jsonl` still parsed 39/39 valid lines with no loss.

Run **one** generator at a time — two writers would interleave appends to the same JSONL files.

## Output layout

```
data/labels/llama31_8b_q4km/
    generation_config.json   settings this label set was produced with
    runs.jsonl               one line per successful generation (incl. text, timings)
    labels.jsonl             one line per completed prompt + target_p90_output_tokens
    failures.jsonl           one line per failed attempt
    checkpoints/checkpoint_00250/{labels_250.jsonl, generation_config.json, checkpoint.summary.json}
```

Checkpoints are **nested** (ranked by subset position, not completion time), so
cp250 ⊂ cp500 ⊂ cp1000 ⊂ cp2000 regardless of retry ordering — the learning curve measures
dataset size alone. Existing checkpoints are never overwritten.

## Verified, not assumed

- `pytest`: **95 passed, 0 skipped** (see *Test state* below). Baseline before any edit this
  session was 63 passed + 1 skipped.
- `ml.train` on CUDA end-to-end; **predictor latency 2.79 ms median on GPU** vs multi-second Llama
  service time — success-bar criterion 3 already met with large margin.
- Real llama.cpp generation: 40/40 twice, coherent answers, chat template applied.
- Resume, crash-tail repair, generation reuse, nested checkpoints, mock-label refusal, status
  tolerance of a half-written line.
- Parquet reading path exercised on synthetic shards with the real LMSYS schema before the real
  data arrived.

## Changes made to the repo

- `datagen/config.py` — `NUM_PROMPTS` 500→**2000**; `MAX_NEW_TOKENS` 512→**1024**;
  added `CHECKPOINT_SIZES`, `RUNTIME_VERSION`.
- `datagen/generate_labels.py` — each label records `runtime_version`.
- `datagen/make_checkpoint.py` — **new**, nested incremental checkpoints.
- `datagen/status.py` — **new**, read-only progress/ETA.
- `scripts/start_llama_server.ps1`, `scripts/run_generation.ps1` — **new**.
- `tests/test_datagen.py` — 8 new tests; one assertion de-hardcoded from 512 to `config.MAX_NEW_TOKENS`.
- `datagen/README.md`, `.gitignore`, this file.

No existing behaviour removed. The ML model was not redesigned or tuned.

## Open item for later

`ml/config.py` has `max_length = 128` tokens for DistilBERT. The real subset's p90 prompt is
~1,519 chars (~380 tokens), so a meaningful share of prompts will be truncated at the predictor's
input. That is already flagged `REAL-DATA` in the file. Worth revisiting at Phase 12 — **not**
changed now, to avoid tuning during the generation run.

---

# Censored-output handling (added while the base run was in flight)

## The problem

A base generation that stopped at `max_new_tokens` carries `finish_reason: "length"`. Its recorded
length is a **lower bound**, not a measurement — the true length is `>=` the cap. Feeding those
values into a p90 target teaches the predictor that every long job costs exactly the cap, which
erases the short-vs-long signal the scheduler exists to exploit.

Live base truncation is settling around **7.5%** of runs. Earlier readings near 16% were
small-sample noise; `datagen.status` reports the current figure.

## Design: a separate extension pass, never an in-place edit

`datagen.extend_censored` re-runs **only** the capped `(prompt_id, seed)` pairs with
`EXTENDED_MAX_NEW_TOKENS = 2048` and *nothing else changed* — same prompt text, same seed, same
temperature, top_p, system prompt. Results go to a **separate directory**. The base files are
opened read-only and never mutated, so the original 1024 measurements survive exactly as recorded.

- **2048, not 1536** — the context has room. Worst-case prompt is `MAX_PROMPT_CHARS` 8000 chars
  ≈ 2000 tokens, and 2000 + 2048 = 4048 < 4096. The actual subset maxes at 2,719 chars (~680
  tokens), so real headroom is far larger. No server restart or context change needed.
- **Prompt text comes from the subset file** keyed by `prompt_id`, not from a base label record,
  so a capped run belonging to a not-yet-complete prompt can still be extended and prompt
  identity is guaranteed to match what the base run was given.
- **Resumable / idempotent / crash-safe** — every `(prompt_id, seed)` already in
  `extended_runs.jsonl` is skipped, each line is fsynced on write, and a half-written final line is
  repaired on start-up. Re-running a completed pass generates nothing.
- **Refuses to mix settings** — a resume with a different cap is rejected (`ConfigMismatch`), and
  it will not write into the base directory.
- **Refuses GPU contention by default** — if `logs/generation.pid` names a live process the pass
  exits with an explanation instead of competing with the base run for the GPU and distorting the
  latency it records. `--allow-concurrent-base` overrides, deliberately.
- **Still censored at 2048 stays censored** — the length is recorded, flagged `still_censored`, and
  counted. Never replaced by a guess.
- **Determinism audit** — same prompt + same seed should regenerate the same prefix, so an extended
  length should be `>=` the base cap. A shorter result means the backend was not deterministic; the
  summary counts those as `shorter_than_base` rather than hiding them.

## Design: final-label assembly

`datagen.assemble_final_labels` writes a new file and leaves both sources untouched. Per-run
precedence, applied independently to each generation:

| Base run | Extended run | Effective length | `length_source` | Censored |
| --- | --- | --- | --- | --- |
| finished naturally | — | base length | `base` | no |
| capped | finished naturally | **extended** length | `extended` | no |
| capped | also hit 2048 | extended length (2048) | `extended` | **yes** |
| capped | none exists | base length (1024) | `base_censored_unextended` | **yes** |

The p90 is recomputed from the effective lengths with the same `numpy` call the base pipeline
uses, so the arithmetic is identical and only the inputs improve.

`has_censored_target` is **not** "any run was censored". With 4 runs and linear interpolation the
p90 sits at sorted index 2.7, so it blends the two largest values — a censored run among the two
smallest cannot move it. Only a censored value the percentile actually reads sets the flag. Ties
are handled conservatively, and an unmodelled percentile method falls back to "if any run is
censored, flag it" rather than assuming this interpolation.

Audit fields on every final record: `n_base_truncated_runs`, `n_extended_runs`,
`n_still_censored_runs`, `n_unextended_censored_runs`, `has_censored_target`, and
`base_target_p90_output_tokens` (what the target would have been without the extension), plus
per-run `base_output_tokens` / `extended_output_tokens` / `effective_output_tokens` /
`length_source` / `is_censored`.

Prompts with `has_censored_target` are **kept, not dropped** — that is a modelling decision for
later, and the flag makes it available either way. The assembly summary reports how many there are.

## Paths

```
data/labels/llama31_8b_q4km/            base run, 1024 cap  — READ-ONLY from here on
data/labels/llama31_8b_q4km_ext2048/    extended_runs.jsonl, extension_failures.jsonl,
                                        extension_config.json
data/labels/llama31_8b_q4km_final/      labels_final.jsonl, assembly.summary.json
data/labels/llama31_8b_q4km_final/checkpoints/checkpoint_00500/labels_500.jsonl
```

## Commands, in order, after the base run finishes

```powershell
# 0. confirm the base run is done and its generator is gone
.\.venv\Scripts\python.exe -m datagen.status
Get-Process -Id (Get-Content logs\generation.pid) -ErrorAction SilentlyContinue   # expect nothing

# 1. size the work (read-only, safe at any time)
.\.venv\Scripts\python.exe -m datagen.extend_censored --backend llamacpp --dry-run

# 2. run the extension pass (needs llama-server up; resume = the same command)
.\.venv\Scripts\python.exe -m datagen.extend_censored --backend llamacpp

# 3. assemble final labels
.\.venv\Scripts\python.exe -m datagen.assemble_final_labels

# 4. cut learning-curve checkpoints from the ASSEMBLED labels
.\.venv\Scripts\python.exe -m datagen.make_checkpoint `
    --labels data\labels\llama31_8b_q4km_final\labels_final.jsonl `
    --out-root data\labels\llama31_8b_q4km_final\checkpoints
```

---

# Phase 12 harness

## Input-length experiment (max_length 128 / 256 / 512)

`max_length = 128` was chosen against the synthetic file. The real subset's p90 prompt is roughly
380 tokens, so 128 truncates a large share of real prompts before DistilBERT sees them. **The
default has NOT been changed** — this is a measured decision:

```powershell
.\scripts\run_maxlen_experiment.ps1 -Labels data\labels\llama31_8b_q4km_final\labels_final.jsonl
.\scripts\run_maxlen_experiment.ps1 -Labels <file> -Device cpu    # if the GPU is still busy
```

Only `max_length` varies. Data file, seed, split, optimizer, architecture, loss, epochs and batch
size are held fixed, so a difference in test MAE is attributable to prompt context length alone.
No hyperparameter search happens here.

## Learning curve (500 / 1000 / 2000)

```powershell
.\scripts\run_learning_curve.ps1 -MaxLength <winner from above>
```

Trains only from **stable** nested checkpoint files, never the growing `labels.jsonl`. Because the
checkpoints are nested, each larger run genuinely adds data rather than resampling it. Settings are
identical across sizes, so the curve measures the value of more data and not different training
choices.

## Epochs / early stopping

`epochs = 3` is kept as the default for the first controlled real-data run. Validation MAE is now
logged every epoch with a `*best*` / `no gain for N` marker. Early stopping is **opt-in**
(`--early-stopping --patience N`) and off by default, because a learning-curve or max_length
comparison needs every variant trained for the same number of epochs. When enabled it restores the
best-validation-MAE weights. If the real model is clearly underfitting, that gets reported, not
silently tuned away.

## Success bar

`ml.compare_runs` collates runs and applies all three criteria, reading the **measured** median
Llama service time from the base run's own `runs.jsonl` rather than a guess:

```powershell
.\.venv\Scripts\python.exe -m ml.compare_runs artifacts\maxlen_128 artifacts\maxlen_256 artifacts\maxlen_512
```

Criterion 3 is strict (`< 5%`, not `<=`). A missing input yields `indeterminate`, never a silent
pass. If DistilBERT loses, the baseline ships through the same `predictor.predict()` interface —
an acceptable outcome.

## GPU contention

Training must **not** use the GPU while generation does. Both `run_maxlen_experiment.ps1` and the
extension pass warn or refuse when `logs/generation.pid` is live. Use `-Device cpu` if training
has to happen first, and note that **CPU predictor latency is not the final number** — criterion 3
must be re-measured on GPU after generation completes.

---

# Bug found and fixed: label files were unreadable on Windows

`ml/data.py` opened label files without an explicit encoding, so Windows used cp1252 and raised
`UnicodeDecodeError` on real prompt text. **11 of 98** sampled real labels contain non-ASCII
characters, so `python -m ml.train --real` would have failed outright on the real dataset. The bug
was pre-existing and latent: the synthetic file and the earlier smoke checkpoints were pure ASCII.

Fixed by making the read and write explicitly utf-8, matching `datagen.jsonl` which already did.
`tests/test_pipeline.py::test_label_files_roundtrip_non_ascii_prompts` guards it and was confirmed
to fail when the fix is reverted. The same explicit encoding was applied to the JSON config
reads/writes in `make_checkpoint`, `status`, `extend_censored` and `assemble_final_labels`.

---

# Test state

**95 passed, 0 skipped.** 71 before this phase, plus 18 in `tests/test_censored_extension.py`
(censored detection, selection, identity preservation, resume, idempotence, crash-tail repair,
config-mismatch refusal, failure logging, p90 contributor logic, precedence, still-censored
behaviour, audit fields, loader compatibility) and 4 in `tests/test_pipeline.py` (max_length and
early-stopping knobs, success-bar logic, non-ASCII round-trip) and 2 in `tests/test_datagen.py`
(idle-aware ETA).

# Files added or changed this phase

| File | Change |
| --- | --- |
| `datagen/config.py` | **added** `EXTENDED_MAX_NEW_TOKENS=2048`, extension/final paths, `extension_config()`. `generation_config()` deliberately untouched — adding a key there would make the live run's own config check fail on its next auto-restart. |
| `datagen/extend_censored.py` | **new** — the extension pass |
| `datagen/assemble_final_labels.py` | **new** — final-label assembly |
| `ml/compare_runs.py` | **new** — collation + success bar |
| `ml/train.py` | `--max-length`, `--early-stopping`, `--patience`; val-MAE best tracking; `run_settings` echoed into `eval_results.json` |
| `ml/config.py` | `early_stopping*` fields; documented why `max_length` stays 128 for now |
| `ml/data.py` | **encoding fix** (see above) |
| `datagen/make_checkpoint.py`, `datagen/status.py` | explicit utf-8 on JSON config IO |
| `scripts/run_maxlen_experiment.ps1` | **new** |
| `scripts/run_learning_curve.ps1` | **new** |
| `tests/test_censored_extension.py` | **new** — 18 tests |
| `tests/test_pipeline.py` | +4 tests |
| `.gitignore` | `build/` |

The base generator, its output directory, its seeds, its decoding settings and its
`max_new_tokens` were **not** touched.

---

# Incident: 80-minute generator stall (operator-caused, data clean)

Recorded here because it is the only hard evidence we have about what starves this pipeline.

Between 2026-10-03 23:53 and 2026-10-04 01:14 UTC the generator sent **no requests** for 80.8
minutes. Diagnosis, in the order it was established:

- `llama-server` received **zero** requests in that window — its slot-launch log has a gap from
  minute 88.5 to 169.3 with 10–18 requests/min either side. The server sat idle; the *client* had
  stopped sending.
- Not a crash and not a restart: `logs/generation.log` shows a single `generator attempt 1`, and
  `failures.jsonl` is empty, so it was not an HTTP timeout either (those would have been logged
  after `REQUEST_TIMEOUT`).
- The window coincides exactly with **CPU-only** DistilBERT training runs and repeated full test
  suites executed on this machine.

## No measurement was corrupted

| Check | Before stall (n=532) | After stall (n=47) |
| --- | --- | --- |
| ms per output token, median | 13.31 | 13.35 |
| decode tok/s, median | 75.76 | 75.43 |

Decode speed is flat across all four quartiles of the whole run (75.8 / 75.6 / 75.9 / 75.6 tok/s).
Client-side request overhead is median 24 ms, p95 64 ms. The stall sat *between* generations, so
no recorded `latency_ms` spans it. The dataset is unaffected; only wall-clock was lost.

## Operational rule this establishes

**Do not run anything heavy on this box while the base run is live — GPU or not.** The documented
hazard was GPU contention, but what actually stalled the generator was CPU and disk pressure from
CPU-only training plus test suites. The generator is a single-threaded Python loop that fsyncs
after every record; starving it stops the pipeline without raising any error.

Task 8 of the brief permits CPU training if it must happen before generation ends. In practice
that still cost 1.35 h. Prefer waiting.

## Tooling change this prompted

`datagen.status` previously reported only a whole-run average rate, which a stall depresses
permanently — it was showing a 28.9 h ETA when the achievable rate implied under 10 h. It now
reports:

| Field | Meaning |
| --- | --- |
| `generations_per_min` | whole-run average, **includes** idle time |
| `generations_per_min_active` | excludes gaps longer than `IDLE_GAP_S` (120 s) |
| `generations_per_min_recent` | last `RECENT_WINDOW` (200) generations — **read this one** |
| `idle_hours` | total time spent in gaps > 120 s |
| `longest_idle_gap_min` | largest single gap, so a stall is visible |
| `eta_basis` | which rate the ETA was computed from |

The ETA now uses the recent rate, falling back to active, then overall. Covered by
`tests/test_datagen.py::test_status_eta_ignores_idle_gaps` and `test_status_without_gaps_reports_no_idle`.

---

# Finding: llama-server prompt caching breaks per-seed reproducibility

This invalidated the planned 2048 extension pass and is the most important caveat on the dataset.

## What was observed

The extension pass re-ran capped `(prompt_id, seed)` pairs with only `max_new_tokens` raised,
expecting to reveal each censored generation's true length. Within the first 12 re-runs, **42%
came back SHORTER than the 1024 base cap** — one at 104 tokens against a base run that had hit
1024. A re-run cannot be shorter than a cap the original exceeded, so the premise was wrong.

## Root cause, established by test

Picking one censored `(prompt, seed)` and repeating the request:

| Request | Result |
| --- | --- |
| cap 1024, seed 42, **default caching**, 3x | 539, 538, 538 tokens — all `stop`, **not reproducible** |
| cap 1024, seed 42, **`cache_prompt: false`**, 3x | 539, 539, 539 — `stop`, **reproducible** |
| cap 2048, seed 42, **`cache_prompt: false`**, 2x | 539, 539 — **identical to cap 1024** |
| what the base run recorded for that same request | **1024 tokens, `length`** |

So with the prompt cache disabled the backend is both deterministic *and* cap-independent — exactly
what the extension pass assumed. With it enabled (the default, and what the base run used), the
same request is not reproducible, and in this case differed enormously: 539 and `stop` versus 1024
and `length`.

The mechanism is visible in the server log: `selected slot by LCP similarity, f_sim_best = 1.000`
and `making room for prompt cache entry`. With `-np 1`, consecutive requests reuse the slot's KV
cache. Because the generator runs a prompt's 4 seeds back-to-back, runs 2–4 reuse the cached prompt
prefix while run 1 evaluates it fresh — so the four "repeats" per prompt are not numerically
equivalent draws, and the difference is systematic rather than random.

## Consequences

1. **A cache-affected censored generation cannot be uncensored by re-running it.** The extension
   approach was abandoned. Its 14 records are quarantined under
   `data/labels/_quarantine/ext2048_INVALID_cache_nondeterminism/` and never reached a label set.
2. **The "deterministic seed" premise in the project's locked decisions is false on this backend
   with caching on.** Seeds 42–45 still produced four distinct samples per prompt, so the p90 target
   remains a valid conservative statistic over four real generations — but those generations are
   not reproducible, and the dataset should not be described as reproducible.
3. **The labels remain real measurements from a real serving stack** that had caching enabled,
   which is also how the scheduler will run. For predicting service cost on this stack that is
   arguably the more representative choice. It is a defensible dataset, not a clean one.
4. **9.87% of runs stay right-censored** (296 of 3,000), affecting 100 of 750 prompts, every one of
   which has `has_censored_target: true` because a run capped at 1024 always lands among the top
   two values the p90 reads. `target_max` is therefore pinned at 1024.

## What a reproducible rerun would require

Send `"cache_prompt": false` on every request and use a fresh output directory. `datagen/backends.py`
does not currently set it — that is a deliberate one-line change left unmade, because adding it to
`generation_config()` would make the existing run's config check fail on its next resume. Cost of a
clean 750-prompt regeneration at cap 2048 with caching off was estimated at roughly 8 h, versus the
~2.4 h budget available, so it was not attempted.

---

# RESULTS — Phase 12 on real labels (750 prompts)

All numbers below were measured on this machine. Test set = 112 held-out prompts, stratified by
category, seed 42. Median Llama service time measured from the base run's own records: **4,693 ms**,
so the 5% overhead budget is **234.7 ms**.

## Input-length experiment (Task 5) — only `max_length` varied

Same data, seed, split, optimizer, architecture, loss, epochs (3) and batch size throughout.

| max_length | DB MAE | DB p90 abs err | DB under | **DB severe under** | mean signed err | latency |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 204.78 | 410.50 | 0.384 | **0.045** | **+43.2** | 2.79 ms |
| 256 | 193.61 | 418.47 | 0.446 | 0.080 | +5.0 | 2.81 ms |
| 512 | **193.14** | 415.20 | 0.420 | 0.089 | +12.8 | 2.84 ms |
| *baseline* | *241.49* | *483.92* | *0.473* | *0.143* | *−14.0* | — |

**Answer to the question the experiment was built to settle:** longer prompt context gives a real
but modest MAE gain — 204.8 → 193.1, about 5.7% — and it saturates by 256 (256 vs 512 differ by
0.5 tokens, which is noise on 112 examples). The prior expectation that 512 would clearly win
because it truncates fewer real prompts is **only weakly supported**.

The more interesting result is the opposite direction: **128 halves severe underprediction**
(0.045 vs 0.089) because it systematically over-predicts (+43.2 signed error vs +12.8). Truncating
the prompt makes the model less certain and therefore more conservative — which is the direction
this project explicitly prefers. On 112 test examples that is 5 cases vs 10, so it is a small-sample
difference, but it points the same way as the project's locked "underprediction is more harmful"
principle.

## Learning curve (Task 6) — only dataset size varied

Checkpoints are nested, so each larger size genuinely adds data.

**max_length 512**

| prompts | n_train | n_test | DB MAE | baseline MAE | DB severe | baseline severe | overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 250 | 175 | 38 | 246.7 | 280.1 | 0.132 | 0.184 | 0.06% |
| 500 | 350 | 75 | 217.2 | 269.0 | 0.027 | 0.160 | 0.08% |
| **750** | 525 | 112 | **193.1** | 241.5 | 0.089 | 0.143 | 0.06% |

**max_length 128**

| prompts | n_train | n_test | DB MAE | baseline MAE | DB severe | baseline severe | overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 250 | 175 | 38 | 247.2 | 280.1 | 0.105 | 0.184 | 0.07% |
| 500 | 350 | 75 | 206.7 | 269.0 | 0.027 | 0.160 | 0.06% |
| **750** | 525 | 112 | **204.8** | 241.5 | 0.045 | 0.143 | 0.06% |

**More data helps, and the curve has not flattened.** At 512, MAE falls monotonically
246.7 → 217.2 → 193.1 and was still dropping at 750. At 128 it plateaus after 500
(206.7 → 204.8), i.e. the longer context is what lets the model exploit extra data. That is a
concrete argument for resuming the base run toward 1000/2000 if time allows.

Severe-underprediction does **not** move monotonically (0.132 → 0.027 → 0.089 at 512). With test
sets of 38 / 75 / 112 examples, one example is worth 2.6 / 1.3 / 0.9 points, so that series is
dominated by noise and should not be read as a trend.

## Success bar (Task 9) — verdict

| Criterion | Requirement | Measured (max_length 512, 750 prompts) | Verdict |
| --- | --- | --- | --- |
| 1. Test MAE | beat baseline | **193.1 vs 241.5** (20% better) | ✅ |
| 2. Severe underprediction | beat or tie baseline | **0.089 vs 0.143** | ✅ |
| 3. Inference overhead | < 5% of median service time | **2.82 ms / 4,693 ms = 0.06%** | ✅ |

**DistilBERT is selected.** It passed all three criteria in **every one of the 9 training runs**
(3 max_lengths + 6 learning-curve points), so the verdict is not sensitive to the particular split
or dataset size.

Criterion 3 passes with roughly an 80x margin, measured on GPU after generation finished — not a
provisional CPU number.

## Scheduler handoff

```python
from ml.predictor import LengthPredictor
predictor = LengthPredictor.load("artifacts/maxlen_512")
predictor.predict(prompt)
# {"expected_output_tokens": float, "bin_probabilities": [20 floats], "uncertainty": float}
```

Verified working, with warm `predict()` latency ~3.3 ms on GPU. Sanity check on the ordering it
produces:

| prompt | predicted tokens | uncertainty |
| --- | --- | --- |
| "Write a 2000 word article about the production process of aspirin." | 766.7 | 2.46 |
| "My laptop will not boot after a BIOS update, help me fix it" | 580.6 | 2.70 |
| "What is the capital of France?" | 459.6 | 2.94 |
| "Summarize this in one line: the cat sat on the mat." | 250.2 | 2.78 |

The long-form request ranks highest and the one-line summary lowest, which is the discrimination
the scheduler needs. But note the short factual question is estimated at 459.6 tokens with
near-maximal uncertainty (2.94 against a ceiling of ln 20 ≈ 3.00): with 525 training examples the
model regresses toward the mean on prompts it is unsure about. The `uncertainty` field exposes
exactly that, and is available to the scheduler.

## Recommended max_length — a decision, not a measurement

The success bar as written ranks on MAE, which selects **512**, and the learning curve shows 512
scaling better with additional data. That is the default recommendation and what
`artifacts/maxlen_512` contains.

If scheduler experiments later show that underprediction is hurting tail latency more than the MAE
gap helps, **128 is the conservative alternative**: 5.7% worse MAE, half the severe
underprediction, same negligible overhead. Both artifacts are saved, so switching is a one-line
path change rather than a retrain.
