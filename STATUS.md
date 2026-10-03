# Project status — real label generation RUNNING

Updated 2026-10-03 ~19:12 local. Machine: Windows 11 Pro 26200, RTX 3060 Ti 8 GB.

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
| 6 llama-server | **running** — PID in `logs/llama-server.pid` |
| 7 Smoke test | **done** — 10 real prompts × 4 seeds, 40/40, twice (512 and 1024) |
| 8 512-cap decision | **decided: raised to 1024** (see below) |
| 9 Real label generation | **RUNNING** — PID in `logs/generation.pid` |
| 10 Checkpoints | tooling ready; first cut at 250 prompts |
| 11 Unattended run | **done** — detached, auto-restart, auto-checkpoint |
| 12 Training on checkpoints | tooling ready and verified on this GPU |

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

- `pytest`: **71 passed, 0 skipped** (63 pre-existing + 8 new). Baseline before any edit was 63+1 skip.
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
