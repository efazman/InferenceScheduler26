# Merge plan: `feature/scheduler-ui-parallel` → `master` (on the RTX 3060 Ti machine)

**Verified 2026-10-03 against the fetched `origin/master` = `f4a6923` ("Finishing up ML"):**
`git merge-tree` reports a clean merge. The branch adds 54 files and modifies one (`.gitignore`,
appended lines only). That result covers *pushed* master only. Any GPU-side work that's
committed locally but not pushed, or uncommitted, hasn't been checked, so redo the dry run on the
GPU machine first (step 3 below).

## File ownership

`git diff origin/master...origin/feature/scheduler-ui-parallel --name-status`

| Class | Files | Merge risk |
| --- | --- | --- |
| **A. Mac-owned (new on this branch)** | `scheduler/**`, `ui/**`, `docs/**`, `tests/test_scheduler_*.py`, `tests/conftest.py` | None, unless the GPU side created a file at the same path |
| **B. GPU-owned (untouched by this branch)** | `ml/**`, `datagen/**`, `scripts/**`, `STATUS.md`, `ORCHESTRATOR_UPDATE.md`, `requirements.txt`, `pytest.ini`, `tests/test_datagen.py`, `tests/test_pipeline.py`, `tests/test_censored_extension.py`, `data/`, `artifacts/`, `logs/` | None. This branch doesn't modify any of them. |
| **C. Shared** | `.gitignore` (modified); `tests/conftest.py` (new path the GPU side might also create); the `ml` / `datagen` APIs the scheduler imports (not edited, but depended on) | Low. Details below. |

## File-by-file notes for class C

**`.gitignore`.** This branch appends `scheduler_runs/` (local run logs, which contain prompts) and
`data/scheduler/` (test prompts and manifests with LMSYS text). If master appended its own lines,
**keep every line from both sides**: master's data/artifact/log ignores and these two. Don't
pick one side.

**`tests/conftest.py`.** New on this branch: a `fake_llama_url` fixture (a minimal llama-server
stand-in) used by `test_scheduler_engine.py`, `test_scheduler_sim.py` and
`test_scheduler_experiment.py`. If the GPU side also created a `tests/conftest.py`, the result
must contain **both sides' fixtures and imports**, with no duplicate fixture names.

**`requirements.txt`.** This branch doesn't change it. The scheduler needs only the standard
library plus what `ml`/`datagen` already need. Dashboard dependencies live in `ui/package.json`
and need Node 18.18 or later. If the GPU side changed `requirements.txt`, keep its version.

**`pytest.ini`.** Not changed here. `testpaths = tests` picks up the scheduler tests on its own.
Keep the GPU side's version.

**`STATUS.md`, `ORCHESTRATOR_UPDATE.md`, ML/datagen READMEs.** Not touched. Scheduler docs live in
`scheduler/README.md`, `ui/README.md` and `docs/`. After merging, *add* a short "Scheduler"
pointer to `STATUS.md` if you want one. Don't move or overwrite its ML content.

**APIs this branch depends on** (a rename on the GPU side breaks imports without causing a text
conflict; the listed tests catch it):

| Used by the scheduler | Contract | Caught by |
| --- | --- | --- |
| `datagen.config` | `BACKEND_URL`, `REQUEST_TIMEOUT`, `MODEL_NAME`, `SYSTEM_PROMPT`, `TEMPERATURE`, `TOP_P`, `MAX_NEW_TOKENS` (also read if present: `QUANTIZATION`, `RUNTIME`, `RUNTIME_VERSION`, `SYSTEM_PROMPT_SHA256`) | `test_llamacpp_*`, `test_real_run_records_*` |
| `datagen.backends` | `OpenAICompatBackend(base_url, model, system_prompt, temperature, top_p, max_new_tokens, timeout)`, `.check()`, `.generate(prompt, seed)` → `text, output_tokens, latency_ms, finish_reason, prompt_tokens, server_timings`; `BackendError(retryable=)` | same |
| `ml.predictor` | `LengthPredictor.load(dir, device=None)`, `.predict(prompt)` → `expected_output_tokens`, `uncertainty` | `test_real_predictor_adapters_*` (needs local artifacts) |
| `ml.baseline` | `PromptLengthBaseline.load(path, tokenizer)`, `.predict([prompt])` | same |
| `ml.config`, `ml.data` | `Config.load(path)` with `data_path`, `target_field`; `load_jsonl(path, target_field)`; `splits.json` with `train/val/test` id lists | `test_prompts_from_split_*` |
| datagen `runs.jsonl` | one line per real generation with `latency_ms` and `is_mock` | `test_max_wait_from_label_runs_*` |

If one of these was renamed, fix the call in `scheduler/backends.py`, `scheduler/predictors.py`,
`scheduler/max_wait.py` or `scheduler/__main__.py`. Don't change the ML side to suit the scheduler.

## Commands (GPU machine, after generation and training are done)

```powershell
# 1. Make sure no GPU-side work is left uncommitted
git status
git add <the GPU-side files you intend to keep>     # review: data\ and artifacts\ are gitignored
git commit -m "<GPU-side work>"
git push origin master                              # optional, but puts the GPU state on the remote first

# 2. Safety pointer and fetch
git branch backup/gpu-pre-merge
git fetch origin

# 3. Dry run: lists conflicting paths without touching the working tree (exit code 0 = clean)
git merge-tree --write-tree --name-only HEAD origin/feature/scheduler-ui-parallel

# 4. Merge (never fast-forward, so the scheduler work stays one identifiable merge)
git merge --no-ff origin/feature/scheduler-ui-parallel
#    on conflicts: resolve each file per the notes above, then
git add <resolved files>; git commit
#    to back out mid-merge:  git merge --abort

# 5. Verify both workstreams
.\.venv\Scripts\python.exe -m pytest                # everything: ML + datagen + scheduler
cd ui; npm install; npm run build; cd ..

# 6. Publish
git push origin master
```
If the merged result is wrong **and not yet pushed**, `git reset --hard backup/gpu-pre-merge`
returns to the state from step 2. After pushing, fix forward with a new commit instead.

The tip to merge is the current head of `origin/feature/scheduler-ui-parallel`. Its hash is
reported when the branch is pushed.
