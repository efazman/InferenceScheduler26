# Runbook: from finished labels to the recorded demo

These commands run on the **RTX 3060 Ti machine** (Windows, PowerShell, repo root, `.venv`).
Steps 1–5 are the ML side's own commands (`STATUS.md`, `ORCHESTRATOR_UPDATE.md`). Steps 6–17
use this branch.

> Don't start any step while the base label run is still generating. CPU load alone stalled it for
> 80 minutes once (STATUS.md, *Incident*).

## 1–5. ML side: finish labels, train, pick the predictor

```powershell
.\.venv\Scripts\python.exe -m datagen.status                                   # 1. base run finished?
.\.venv\Scripts\python.exe -m datagen.extend_censored --backend llamacpp       # 2. 2048-token extension
.\.venv\Scripts\python.exe -m datagen.assemble_final_labels                    # 3. final labels
.\scripts\run_maxlen_experiment.ps1 -Labels data\labels\llama31_8b_q4km_final\labels_final.jsonl   # 4. train/evaluate
.\.venv\Scripts\python.exe -m ml.compare_runs artifacts\maxlen_128 artifacts\maxlen_256 artifacts\maxlen_512
```
5. The success bar picks **DistilBERT** (the winning `artifacts\<dir>`) or the **prompt-length
   baseline** (stored in the same artifacts directory as `baseline.json`). Write down
   `artifacts\<winner>`. Every command below uses it.

## 6–8. Merge this branch and test

```powershell
git status                                   # GPU-side work committed? Commit it first (see docs/MERGE_PLAN.md)
git fetch origin
git switch master
git merge --no-ff origin/feature/scheduler-ui-parallel
.\.venv\Scripts\python.exe -m pytest         # 8. ML + datagen + scheduler suites, all must pass
cd ui; npm install; npm run build; cd ..     # dashboard builds
```
Conflict handling is in [`docs/MERGE_PLAN.md`](MERGE_PLAN.md).

## 9. Configure the real predictor (no code changes)

The scheduler only calls `predict(prompt) -> {"expected_output_tokens": float, "uncertainty": float|None}`.
Pick the predictor with one flag:

| Winner | Flag | What loads | Initialization call |
| --- | --- | --- | --- |
| DistilBERT | `--predictor distilbert:artifacts\<winner>` | `encoder\`, `tokenizer\`, `head.pt`, `bins.json`, `config.json` | `scheduler.predictors.DistilBertPredictor(dir)`, which wraps `ml.predictor.LengthPredictor.load(dir)` |
| Baseline | `--predictor baseline:artifacts\<winner>` | `baseline.json` (+ `tokenizer\` if the counter is `distilbert`) | `scheduler.predictors.LinearBaselinePredictor(dir)`, which wraps `ml.baseline.PromptLengthBaseline.load(...)` |

The swap point is the `--predictor` flag, parsed by `scheduler.predictors.load_predictor(spec)`.
If the baseline's `baseline.json` says `"prompt_token_counter": "llamacpp:<url>"`, llama-server
has to be running. It will be, from step 10.

Quick check:
```powershell
.\.venv\Scripts\python.exe -c "from scheduler.predictors import load_predictor; print(load_predictor('distilbert:artifacts/<winner>').predict('How do I reset my VPN?'))"
```

## 10. Configure the llama.cpp backend

```powershell
.\scripts\start_llama_server.ps1 -Background   # same server as label generation: -c 4096 -ngl 99 -np 1
```
`-np 1` (one slot) is what makes our queue order the execution order. The client is
`scheduler.backends.LlamaCppBackend`, which wraps the label generator's own `OpenAICompatBackend`.
Settings:

| Setting | Default | Override |
| --- | --- | --- |
| URL | `http://127.0.0.1:8080` | `LLM_BACKEND_URL` env var or `--url` |
| Timeout per generation | `datagen.config.REQUEST_TIMEOUT` (300 s) | `LLM_REQUEST_TIMEOUT` or `--timeout-s` |
| Model, system prompt, temperature, top_p, max_new_tokens | `datagen/config.py` (same as the labels) | change only there |

For each request it captures client-side latency, `completion_tokens`, `finish_reason`,
`prompt_tokens` and llama.cpp `timings`. A failed generation becomes a `request_failed` event,
and the run continues. Serving uses `MAX_NEW_TOKENS` = 1024, the base-run cap, so actual lengths
are capped at 1024 even though the training labels were extended to 2048.

## 11–12. Measure the median service time and derive MAX_WAIT

```powershell
.\.venv\Scripts\python.exe -m scheduler measure-service --from data\labels\llama31_8b_q4km\runs.jsonl
```
This prints `median_service_ms` and `derived_max_wait_ms` = **3.0 × median** (change it with
`--multiplier`). The base `runs.jsonl` holds thousands of real generations with the same model,
server and settings the scheduler will use. Mock or simulated measurements are refused.
`run` records the value and its source in `run_started` and `summary.json`, and the dashboard
shows it.

## 13. Same arrival trace under FIFO, SEJF and Adaptive

```powershell
# held-out prompts the predictor never trained on
.\.venv\Scripts\python.exe -m scheduler prompts-from-split --artifacts artifacts\<winner> --split test `
    --out data\scheduler\test_prompts.jsonl

# freeze ONE trace: request set, prompts, arrival times, seeds (no scheduling order inside)
.\.venv\Scripts\python.exe -m scheduler make-manifest --prompts-file data\scheduler\test_prompts.jsonl `
    --mean-interarrival-ms 7000 --seed 42 --out data\scheduler\real_manifest.json

# replay it once per policy, sequentially (never two runs at once: one GPU, K = 1)
foreach ($p in "fifo","sejf","adaptive") {
  .\.venv\Scripts\python.exe -m scheduler run --backend llamacpp --manifest data\scheduler\real_manifest.json `
      --predictor distilbert:artifacts\<winner> --policy $p `
      --median-service-from data\labels\llama31_8b_q4km\runs.jsonl --run-name real-$p
}
```
- Every policy gets the same requests, prompts, arrival times and per-request seeds. Predictor
  outputs are deterministic per prompt. Only the order changes.
- `--mean-interarrival-ms` sets the load: load ≈ mean service ÷ mean inter-arrival. At about
  6 s mean service, 7000 ms gives roughly 0.85. Use the same manifest for all three runs. A new
  load means a new manifest, run under all three policies again.
- Runtime is about the number of requests × mean service time per policy. The test split is 15%
  of the labelled prompts: about 112 at 750 labels, × ~6 s ≈ 11 min per policy and ~35 min for all
  three. This is an estimate; measure it.
- A run never appends to an existing run directory. Use a new `--run-name` to redo one.

## 14. Save and check events and metrics

```powershell
.\.venv\Scripts\python.exe -m scheduler report scheduler_runs\real-fifo scheduler_runs\real-sejf scheduler_runs\real-adaptive `
    --out docs\REAL_RESULTS.md
```
`report` refuses a comparison that isn't like-for-like. It checks that all runs share the same
manifest, request set, arrival times, predictions, backend settings and threshold, and exits
non-zero if not. Each run directory holds `events.jsonl` (fsynced per event), `summary.json` and
`manifest.json`.

**Metrics to lead with (latency and fairness):** mean, p50 and p95 end-to-end latency, mean and
max queue wait, short- and long-request mean latency, and starvation count. Report p99 only with
100 or more completed requests (the table flags it). Throughput is a sanity check only. With
K = 1 and no preemption, reordering shouldn't change it, and if it differs noticeably, look for
failures or stalls.

## 15. Load the results in the dashboard

```powershell
cd ui; npm run start     # http://localhost:3000/?mode=runs&run=real-adaptive
```
The **Recorded / live runs** tab shows each run, labelled **REAL measurement · llama.cpp**. When
the three runs share a manifest, it adds a same-manifest comparison (queue strips, timeline,
summary). Runs started while the dashboard is open are shown live.

## 16. Tiger Data

The `scheduler_events` hypertable already exists in the `db-inference` Tiger Cloud service (it was
created from the Mac with `tiger-init`, and it's empty). On the GPU machine:

1. Copy `tiger-cloud-db-inference-credentials.env` to the repo root **by hand** (USB or a password
   manager, never git or chat). It's gitignored there. Alternatively, set `TIGER_DATA_DSN` in the
   shell.
2. Install the driver and check the connection, read-only:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements-tiger.txt
   .\.venv\Scripts\python.exe -m scheduler tiger-check
   ```
3. Either add `--tiger` to the three `run` commands in step 13, which streams events live, or upload
   afterwards:
   ```powershell
   .\.venv\Scripts\python.exe -m scheduler tiger-import scheduler_runs\real-fifo scheduler_runs\real-sejf scheduler_runs\real-adaptive
   ```

Each row is labelled `measurement = real | mock | simulated`. `tiger-import` uploads only real runs
unless you pass `--allow-non-real`. Uploads are idempotent (one row per run_id + seq), so re-running
an import, or importing a run that was also streamed live, adds nothing twice.

The sink can't hurt a run. Writes happen on a background thread with retries, and if the database
is unreachable the run carries on, `summary.json` records the rows that didn't make it, and
`tiger-import` fills them in later. Example query (per-policy latency for one manifest) at the
bottom of `scheduler/tigerdata_schema.sql`.

## 17. Record the demo

Tell the story in this order, and keep simulated and real results separate:

1. **Simulation tab, `head_of_line`** (banner: *SIMULATED*): the same requests arrive under all
   three policies.
2. **FIFO:** the long request blocks the shorts (`LONG | SHORT | SHORT`).
3. **SEJF:** the shorts move forward (`SHORT | SHORT | LONG`).
4. **The downside:** switch to `mostly_long` or `bursty` and scrub forward. Under SEJF, long
   requests pile up and turn red (overdue).
5. **Adaptive:** short-job preference, with overdue requests served oldest-first, so waits are
   bounded.
6. **Recorded tab, `real-*` runs** (badge: *REAL measurement*): the real predictor estimated each
   prompt's length…
7. …and real Llama 3.1 8B executed them on the RTX 3060 Ti.
8. **The same-manifest comparison table:** measured latency and fairness for the three policies.
   Only these numbers are real measurements.
