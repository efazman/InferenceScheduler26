# Orchestrator — InferenceScheduler26

Cross-machine coordination state. Deep detail on the ML side lives in `STATUS.md`; on the
scheduler side in `scheduler/README.md`, `ui/README.md` and `docs/` (arriving with the merge).

Repo: `C:\Users\efazr\Desktop\Mhacks26\InferenceScheduler26` (**not** `~/MHacks26`)
Remote: `https://github.com/efazman/InferenceScheduler26.git`

```
STATE              two workstreams complete, awaiting merge
ML (GPU machine)   DONE  - DistilBERT selected, all 3 success criteria passed in 9/9 runs
SCHEDULER (other)  DONE  - 17 commits on origin/feature/scheduler-ui-parallel
MERGE              VERIFIED CLEAN  (git merge-tree: exit 0, 0 conflicting paths)
BLOCKERS           none
NEEDS_HUMAN        YES - authorise the merge + push (and confirm the winner, see 7)
WINNER ARTIFACT    artifacts\maxlen_512   (alternative: artifacts\maxlen_128, see 7)
TESTS              95 passed locally; +~45 scheduler tests arrive with the merge
```

---

## 1. Repo topology — verified from the remote, not assumed

`git fetch --all` at the time of writing:

| Ref | Head | Age | Contents |
| --- | --- | --- | --- |
| `master` (local) | `f4a6923` | — | ML + datagen, **in sync with origin** (0 ahead, 0 behind) |
| `origin/master` | `f4a6923` | 5 h | same |
| `origin/feature/scheduler-ui-parallel` | `f9f997a` | minutes | scheduler + UI + Tiger Data |

**Merge base is `f4a6923`** — which *is* the current master head. The other machine branched from
the finished ML work, so there is no divergence to reconcile: `git rev-list --count $BRANCH..master`
is **0**. This is a pure additive merge, taken with `--no-ff` to keep it identifiable.

### Uncommitted on this machine (commit before merging)

```
 M ORCHESTRATOR_UPDATE.md      this file
 M STATUS.md                   results + determinism finding
 M datagen/make_checkpoint.py  censored-run counting fix for assembled labels
?? scripts/watch_for_750.sh    the 750-prompt watch
```

None of these four is touched by the scheduler branch, so committing them cannot introduce a
conflict. Everything else from this session is already in `62d88be` and `f4a6923`.

---

## 2. What the other machine built

17 commits, 65 files, **8,747 insertions, 0 deletions**.

| Area | Files | What it is |
| --- | --- | --- |
| `scheduler/` | 20 | request model, K=1 engine, FIFO / SEJF / adaptive policies, metrics, clocks, event sinks, manifests, derived MAX_WAIT, workloads, simulator, CLI (`__main__.py`, 524 lines) |
| `scheduler/tigerdata.py` + `.sql` | 2 | non-blocking Tiger Data event sink, schema, init/check/import, credential guards |
| `ui/` | 27 | Next.js dashboard — queue reordering, execution timeline, live metrics, replay |
| `docs/` | 11 | runbook, merge plan, sim results (n=60, n=1000), threshold sweep, demo flow, PM status |
| `tests/` | 6 | ~45 scheduler tests, `conftest.py` (5 fixtures incl. a fake llama-server), repo-hygiene test |
| `requirements-tiger.txt` | 1 | `psycopg` for the Tiger sink, deliberately outside `requirements.txt` |

Simulated FIFO / SEJF / adaptive results are in `docs/SIM_RESULTS.md`. The **real** measured run is
designed to happen on *this* machine after the merge.

The brief's "do not implement yet" list (scheduler, Tiger Data, dashboard, frontend) has been
**superseded by parallel work** — all of it exists. Earlier versions of this file said otherwise;
that guidance is withdrawn.

---

## 3. Merge risk assessment — evidence, not optimism

| Check | Method | Result |
| --- | --- | --- |
| Conflicting paths | `git merge-tree --write-tree --name-only HEAD $BRANCH` | **exit 0, 0 paths** |
| Do they touch my files? | `git diff --name-only HEAD..$BRANCH -- ml/ datagen/ scripts/ requirements.txt pytest.ini STATUS.md ORCHESTRATOR_UPDATE.md tests/test_{datagen,pipeline,censored_extension}.py` | **empty — none** |
| `.gitignore` (only shared file) | diffed both sides | theirs is a **strict superset** of mine; `comm -23` shows **zero** of my lines missing |
| `tests/conftest.py` collision | they add it; do I have one? | **I do not** — no collision |
| Deletions anywhere | `--stat` | **0 deletions** across all 65 files |

Their `docs/MERGE_PLAN.md` states 54 added files. That count is **stale** — written at `2e289d9`,
before the four Tiger Data commits. The real figure is 65 files changed. The plan's *reasoning*
still holds; only the number is out of date.

Their plan correctly warns that their dry run covered pushed master only. I re-ran it here against
local `HEAD`: clean. But my four uncommitted files were not in that tree, so commit them and re-run
before merging (step 2 below).

---

## 4. The merge — exact commands

```powershell
# 0. preserve a rollback point
git branch backup/gpu-pre-merge

# 1. commit the outstanding GPU-side work (none of it is touched by the branch)
git add STATUS.md ORCHESTRATOR_UPDATE.md datagen\make_checkpoint.py scripts\watch_for_750.sh
git commit -m "ML results, determinism finding, checkpoint censoring fix, 750 watch"
git push origin master

# 2. re-run the dry run against the real tree (exit 0 = clean)
git fetch origin
git merge-tree --write-tree --name-only HEAD origin/feature/scheduler-ui-parallel

# 3. merge, keeping the scheduler work as one identifiable merge commit
git merge --no-ff origin/feature/scheduler-ui-parallel

# 4. verify BOTH workstreams
.\.venv\Scripts\python.exe -m pytest            # expect 95 ML/datagen + ~45 scheduler
cd ui; npm install; npm run build; cd ..        # needs Node >= 18.18

# 5. publish
git push origin master
```

Back out with `git merge --abort` mid-merge, or `git reset --hard backup/gpu-pre-merge` if the
result is wrong and **not yet pushed**. After pushing, fix forward with a new commit.

`.gitignore`, if git ever does ask: **keep every line from both sides.** Mine covers `data/`,
`artifacts/`, `models/`, `vendor/`, `logs/`, `build/`; theirs adds `scheduler_runs/`,
`data/scheduler/` and credential patterns `*.env`, `.env*`, `tiger-cloud-*credentials*`. Dropping
either side either leaks credentials or commits a 4.9 GB model.

---

## 5. What git does NOT carry — the most important operational fact

These are gitignored, so a `git pull` on any machine gets **none** of them:

| Path | Size | Needed by |
| --- | --- | --- |
| `artifacts/maxlen_512/` (also `maxlen_128`, `curve*`) | 255 MB each | `--predictor distilbert:...` |
| `data/labels/llama31_8b_q4km_final/labels_final.jsonl` | 750 labels | retraining, `prompts-from-split` |
| `data/labels/llama31_8b_q4km/` (runs, labels, checkpoints) | ~100 MB | derived MAX_WAIT, retraining |
| `data/lmsys/subset_2000.jsonl` | 2,000 prompts | real workloads |
| `models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4.9 GB | llama-server |
| `vendor/llama.cpp-b11381-cuda-12.4/` | ~1.2 GB | llama-server |

**No transfer is required.** Their runbook puts the real experiment on *this* machine, and the
predictor is chosen by a CLI flag (`--predictor distilbert:artifacts\<winner>`) rather than a
hardcoded path. All six items already exist here. The other machine only needs the simulated
workloads it already committed under `ui/public/sim/`.

If an artifact ever does need to reach another host: copy `artifacts/maxlen_512/` wholesale
(255 MB, self-contained — see §6) plus `ml/__init__.py`, `ml/predictor.py`, `ml/model.py`,
`ml/config.py`. CPU `torch` suffices; `sklearn` is only needed for the baseline.

---

## 6. API contract — verified compatible, not assumed

Their `scheduler/predictors.py` wraps my code. Both sides' real signatures:

| They call | My signature | |
| --- | --- | --- |
| `LengthPredictor.load(artifacts_dir, device=device)` | `load(cls, out_dir, device: str \| None = None)` | ✅ |
| `.predict(prompt)` → reads `expected_output_tokens`, `uncertainty` | returns those plus `bin_probabilities`; extras ignored | ✅ |
| `PromptLengthBaseline.load(dir / "baseline.json", tokenizer)` | `load(cls, path, fallback_tokenizer=None)` | ✅ |
| `datagen.config` constants, `datagen.backends.OpenAICompatBackend(...)` | unchanged this session | ✅ |
| `splits.json` with `train/val/test` id lists | written by `ml.train` into every artifact dir | ✅ |

**Artifact portability, tested on a simulated fresh host** (copied elsewhere, all network calls
blocked, `HF_HUB_OFFLINE=1`): loads and predicts fine — it bundles `encoder/` and `tokenizer/`, so
nothing is fetched at runtime. 3.3 ms on GPU, 11.7–14.5 ms on CPU (0.07% and 0.25–0.31% of the
4,693 ms median service time).

Two notes for their side:
- `config.json` carries `prompt_token_counter = "llamacpp:http://127.0.0.1:8080"`. The DistilBERT
  predictor **ignores** it (verified with the network blocked). Only `LinearBaselinePredictor`
  would try to reach it, so the baseline path needs llama-server up or that field changed.
- `scheduler/config.py` sets `SIM_PREDICTOR_LATENCY_MS = 3.0`. Measured is **2.82 ms** on GPU —
  close enough that the simulations stand, but worth updating in the real-run report.

---

## 7. The one input the scheduler side needs from me

Their runbook says: *"write down `artifacts\<winner>`; every command below uses it."*

**Winner: `artifacts\maxlen_512`** — best test MAE (193.1 vs baseline 241.5) and it scaled best
with data. Use `--predictor distilbert:artifacts\maxlen_512`.

**Conservative alternative: `artifacts\maxlen_128`** — 5.7% worse MAE (204.8) but **half the severe
underprediction** (0.045 vs 0.089), because it systematically over-predicts (+43.2 mean signed
error). Since underprediction is what causes head-of-line blocking, if the real run shows 512
hurting tail latency, swap the flag. Both artifacts exist; no retrain needed.

Success bar, 112 held-out prompts, median service time 4,693 ms:

| Criterion | Requirement | Measured (maxlen_512) | |
| --- | --- | --- | --- |
| Test MAE | beat baseline | 193.1 vs 241.5 (20% better) | ✅ |
| Severe underprediction | beat or tie baseline | 0.089 vs 0.143 | ✅ |
| Overhead | < 5% of median service time | 2.82 ms = **0.06%** | ✅ |

Passed in **all 9** training runs (3 max_lengths + 6 learning-curve points), so the verdict is not
an artifact of one split or dataset size.

---

## 8. Dataset caveats the real-run write-up must carry

1. **Seeds are not reproducible on this backend.** llama-server reuses the KV/prompt cache by
   default, so the same request is not repeatable: one case recorded as 1024/`length` in the base
   run deterministically yields 539/`stop` with `cache_prompt: false`. The four runs per prompt are
   still four real samples, so the p90 target is valid — but do **not** describe the dataset as
   reproducible. Full evidence in `STATUS.md`.
2. **That is why the 2048 extension pass was abandoned.** A cache-affected censored generation
   cannot be uncensored by re-running it. Its 14 records are quarantined under
   `data/labels/_quarantine/` and never reached a label set.
3. **9.87% of runs remain right-censored** (296 of 3,000) across 100 of 750 prompts, all flagged
   `has_censored_target: true`. `target_max` is pinned at 1024.
4. **Scope is 750 prompts, not 2000** (cut for time). Test set 112 examples — enough to choose a
   predictor, thin for a strong accuracy claim.
5. **The learning curve had not flattened** at 750 (MAE 246.7 → 217.2 → 193.1), so more data would
   likely still help.

---

## 9. Ownership, post-merge

| Area | Owner | Notes |
| --- | --- | --- |
| `ml/`, `datagen/`, `scripts/`, `requirements.txt`, `pytest.ini` | GPU machine | the scheduler adapts to these, not the reverse |
| `scheduler/`, `ui/`, `docs/`, `tests/test_scheduler_*`, `requirements-tiger.txt` | other machine | — |
| `.gitignore`, `tests/conftest.py` | shared | union of both sides, never one side |
| `STATUS.md` | GPU machine | add a "Scheduler" pointer after merging; don't overwrite ML content |
| `ORCHESTRATOR_UPDATE.md` | shared coordination | this file |

---

## 10. Optional follow-ups, none blocking

1. **Resume the base run toward 1000/2000** for a stronger curve — unattended,
   `.\scripts\run_generation.ps1 -Detach`, then re-assemble, re-cut checkpoints with `--force`,
   retrain. Completed prompts are skipped; prompt 751's banked generation is reused.
2. **A reproducible dataset** needs `"cache_prompt": false` on every request plus a fresh output
   directory. `datagen/backends.py` does not set it — deliberately left unmade, since adding it to
   `generation_config()` would break the existing run's resume check. ~8 h for a clean 750.
3. `llama-server` PID **1420** is still up (5.2 GB VRAM), kept for the baseline's `/tokenize`. The
   real scheduler run needs it anyway, so leave it unless you need the VRAM.

---

## 11. Corrections to earlier versions of this file

- The "do not implement yet" list is **withdrawn** — scheduler, Tiger Data, dashboard and UI all
  exist on the branch.
- An earlier handoff note offered to add a batched predict call. Unnecessary:
  **`LengthPredictor.predict_batch(prompts)` already exists** in `ml/predictor.py`.
- Earlier versions implied there was no git remote and only a 2-commit history. Both wrong: the
  remote is `efazman/InferenceScheduler26` and master is at `f4a6923` with the ML work committed
  and pushed.
