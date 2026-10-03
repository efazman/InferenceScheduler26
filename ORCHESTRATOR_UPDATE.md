# Orchestrator update — InferenceScheduler26

Generated 2026-10-03 19:26 local (02:26 UTC). Supersedes nothing; `STATUS.md` holds the full detail.
Repo: `C:\Users\efazr\Desktop\Mhacks26\InferenceScheduler26` (**not** `~/MHacks26`).

---

## 1. Headline

Machine bootstrap and data acquisition are **complete**. The 8,000-generation label run is
**in flight and healthy**. No blockers. No human action needed until a checkpoint lands.

```
STATE            label_generation_running
BLOCKERS         none
NEEDS_HUMAN      no
NEXT_TRIGGER     checkpoint_00500 exists -> start Phase 12 learning curve
ETA_COMPLETE     2026-10-04T12:35Z (±2h)
```

---

## 2. Live processes — do not kill

| What | PID | Verify | Notes |
| --- | --- | --- | --- |
| `llama-server` | **1420** | `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/health` → 200 | holds 5,195 MiB VRAM, 1 slot, ctx 4096 |
| generation runner | **28228** | `Get-Content logs\generation.pid` | detached PowerShell; auto-restarts generator, auto-cuts checkpoints |

PIDs are also on disk: `logs/llama-server.pid`, `logs/generation.pid`.

**Only one generator may run at a time.** Two writers would interleave appends into the same
JSONL files and corrupt the label set. Before starting anything that generates, confirm PID 28228
is gone.

GPU is at 98% utilization with ~3 GiB VRAM free. Do **not** schedule GPU work (including
DistilBERT training) against this box while generation runs — it will contend for VRAM and skew
the latency measurements that the labels record.

---

## 3. Progress

| Metric | Value |
| --- | --- |
| Prompts complete | **43 / 2,000** (2.2%) |
| Generations | **172 / 8,000** |
| Rate | 9.9 generations/min |
| Elapsed / ETA | 0.29 h / **13.2 h** → 2026-10-04T12:35Z |
| Failures | **0 attempts, 0 permanent** |
| Truncation at 1024 | 16.3% so far (smoke predicted 12.5%; small-sample noise) |
| Output tokens median / p90 | 429 / 1024 |
| Latency median | 5.72 s |

ETA drifts as the category mix cycles through long-form vs short prompts; treat ±2 h as normal.
Re-read it rather than extrapolating from this snapshot.

---

## 4. Phase ledger

| Phase | State | Owner next |
| --- | --- | --- |
| 1 Machine/repo audit | done | — |
| 2 Python env | done | — |
| 3 LMSYS dataset | done | — |
| 4 llama.cpp CUDA | done | — |
| 5 Model acquisition | done | — |
| 6 llama-server | running | — |
| 7 Smoke test | done (twice: 512 and 1024) | — |
| 8 Token-cap decision | **decided: 1024** | — |
| 9 Label generation | **RUNNING** | unattended |
| 10 Checkpoints | automatic | unattended |
| 11 Report + don't block | done | — |
| 12 Train learning curve | **waiting on cp500** | next agent |

---

## 5. Decisions already made — do not re-litigate

| Decision | Value | Who |
| --- | --- | --- |
| `MAX_NEW_TOKENS` | **1024** (raised from 512) | project owner, explicitly |
| llama.cpp acquisition | official prebuilt b11381 CUDA 12.4, not source build | project owner |
| torch | CUDA cu128 wheel installed | project owner |
| Subset size | 2,000 prompts × 4 generations | brief |
| Model | Llama 3.1 8B Instruct Q4_K_M, bartowski GGUF | brief |

The 1024 decision is **irreversible for this dataset**. `generation_config.json` pins it and
`generate_labels` refuses a resume whose settings differ, so a label set can never mix caps. If
anyone wants a different cap, it requires a **new `--output-dir`** and a fresh 13-hour run.

Rationale, for anyone tempted to revisit: at 512, 32.5% of runs hit the cap and 4/10 prompts had
their p90 target pinned at exactly 512. One prompt recorded `[512,512,512,512]` was actually
`[1024,529,553,558]` — three of its runs finish near 550. The cap was collapsing distinct costs
onto one censored value, destroying exactly the short-vs-long signal the scheduler exists to
exploit.

---

## 6. Artifacts an orchestrator can hand to downstream work

| Path | Contents | Ready |
| --- | --- | --- |
| `data/lmsys/subset_2000.jsonl` | 2,000 prompts, `prompt_id`/`prompt`/`category`, seed 42 | ✅ now |
| `data/labels/llama31_8b_q4km/labels.jsonl` | growing; one line per completed prompt | partial |
| `.../checkpoints/checkpoint_00250/labels_250.jsonl` | standalone training set | ~1.5 h |
| `.../checkpoints/checkpoint_00500/labels_500.jsonl` | standalone training set | ~3.5 h |
| `.../checkpoints/checkpoint_01000/labels_1000.jsonl` | standalone training set | ~7 h |
| `.../checkpoints/checkpoint_02000/labels_2000.jsonl` | full dataset | ~13 h |
| `artifacts/distilbert_length_predictor/` | predictor trained on **synthetic** data — pipeline proof only, metrics meaningless | ✅ now |

Checkpoints are **nested** (ranked by subset position, not completion time), so
cp250 ⊂ cp500 ⊂ cp1000 ⊂ cp2000 regardless of retry ordering. The learning curve therefore
measures dataset size alone. Existing checkpoints are never overwritten.

---

## 7. Dispatchable work, by dependency

**Claimable now, CPU-only, no contention:**
- Nothing on the critical path. Everything downstream needs labels.

**Claimable when `checkpoint_00500` exists (Phase 12):**
```powershell
.\.venv\Scripts\python.exe -m ml.train --real `
  --data data\labels\llama31_8b_q4km\checkpoints\checkpoint_00500\labels_500.jsonl `
  --output-dir artifacts\distilbert_llama31_cp500
.\.venv\Scripts\python.exe -m ml.evaluate --real --artifacts artifacts\distilbert_llama31_cp500
```
Record: test MAE, p90 absolute error, underprediction rate, severe-underprediction rate,
predictor latency. Compare against the `baseline_prompt_length_linreg` block that `ml.train`
already emits in the same output. Repeat at 1000 and 2000.

Constraint: **do not tune hyperparameters during the generation run** and do not redesign the
model. One training run per checkpoint, same settings, so the curve is interpretable.

Caveat for whoever takes it: training wants the GPU, which generation is saturating. Either wait
for generation to finish, or pass `--device cpu` (DistilBERT on ~350 train examples at 128 tokens
is minutes on 12 cores).

**Blocked until Phase 12 concludes:** scheduler (FIFO / SJF / adaptive), Tiger Data, dashboard,
EGTP, RAG, multi-GPU, K>1 concurrency, frontend. All explicitly out of scope right now.

---

## 8. Success bar for Phase 12 (unchanged)

DistilBERT is chosen over the baseline only if it (1) beats baseline test MAE, (2) beats or ties
baseline severe-underprediction rate, (3) keeps inference overhead under 5% of median Llama
service time.

Criterion 3 is **already met with large margin**: measured predictor latency is **2.79 ms** median
on GPU against a **5.72 s** median Llama service time — about 0.05%.

If DistilBERT loses on (1) or (2), that is not a project failure. The scheduler consumes either
model through the same `predictor.predict()` interface.

---

## 9. Recovery runbook

Everything is resumable. Completed prompts are skipped and completed generations reused, so a
restart never redoes finished work and never starts from zero.

```powershell
# server died?
.\scripts\start_llama_server.ps1 -Background
# generator died? (same command resumes from any state)
.\scripts\run_generation.ps1 -Detach
# check
.\.venv\Scripts\python.exe -m datagen.status
```

Verified under real failure, not just by design: the generator's Python child was hard-killed
mid-generation and `runs.jsonl` still parsed 39/39 valid lines with zero loss. Every line is
fsynced on write and a half-written final line is repaired on startup.

Failure triage: `failures.jsonl` logs every failed attempt. HTTP 4xx is not retried (a bad
request, e.g. over-context prompt); 5xx/429/transport errors retry with backoff. A prompt still
failing after retries is left incomplete and retried on the next run.

---

## 10. Known open items (not blocking, do not fix mid-run)

1. **`ml/config.py max_length = 128` tokens.** The real subset's p90 prompt is ~380 tokens, so a
   meaningful share of prompts truncate at the *predictor's* input. Already flagged `REAL-DATA` in
   the file. Belongs in Phase 12 as a deliberate experiment, not a silent config change.
2. **`ml/config.py epochs = 3`** is tuned for the synthetic file. Real-data training may want more
   plus early stopping on val MAE — also flagged `REAL-DATA`.
3. **Categories are `heuristic-v1` keyword rules**, approximate workload-balancing labels only.
   Not ground-truth semantic classes; don't report them as classification accuracy.
4. **Truncation remains ~16%** even at 1024. Labels with `finish_reason: "length"` are
   right-censored lower bounds. `n_truncated_runs` is recorded per label so this is auditable.
5. **Uncommitted git state** (below). Nothing has been committed; the owner has not asked for it.

```
 M .gitignore  datagen/README.md  datagen/config.py
 M datagen/generate_labels.py  tests/test_datagen.py
?? STATUS.md  ORCHESTRATOR_UPDATE.md  datagen/make_checkpoint.py
?? datagen/status.py  scripts/
```

---

## 11. Test state

`pytest`: **71 passed, 0 skipped.** Baseline before any edit this session was 63 passed / 1
skipped; the 8 added tests cover the new checkpoint and status tooling, and the previously skipped
test now runs because a trained artifact exists. No existing behaviour was removed or changed.
