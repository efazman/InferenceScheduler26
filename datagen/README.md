# Label generation: LMSYS-Chat-1M to Llama 3.1 8B output-length labels

This pipeline produces the real training labels for the DistilBERT length predictor in `ml/`.
Every setting lives in `datagen/config.py`.

| | Selected |
| --- | --- |
| Model | Meta Llama 3.1 8B Instruct, GGUF **Q4_K_M** |
| Runtime | llama.cpp `llama-server`, one slot (`-np 1`), so one generation at a time |
| Hardware | RTX 3060 Ti 8 GB |
| Decoding | temperature 0.7, top_p 0.9, max_new_tokens 1024, fixed system prompt, seeds 42–45 |
| Dataset | LMSYS-Chat-1M: English, single user turn, exact duplicates removed |
| Subset | **2,000 prompts**: 20% knowledge, 20% explanation, 15% each for coding, summarization, writing and troubleshooting |
| Target | p90 of 4 output-token counts (linear interpolation, so x3 + 0.7·(x4 − x3)) |
| Scale | 2,000 prompts × 4 generations = **8,000 generations** |
| Checkpoints | Standalone training sets cut at 250 / 500 / 1000 / 2000 completed prompts |

```
LMSYS parquet ──preprocess_lmsys──▶ clean_prompts.jsonl ──select_subset──▶ subset_2000.jsonl
      ──generate_labels (llama-server ×4 seeds)──▶ labels.jsonl ──python -m ml.train --real──▶ predictor
```

## 1. Get LMSYS-Chat-1M (any machine, no GPU)
The dataset is gated. Accept its terms at https://huggingface.co/datasets/lmsys/lmsys-chat-1m
(approval is automatic), then:
```bash
source .venv/bin/activate
hf auth login                      # paste a read token from huggingface.co/settings/tokens
hf download lmsys/lmsys-chat-1m --repo-type dataset --local-dir data/raw/lmsys-chat-1m   # ~1.5 GB
```
`data/raw/` is gitignored. If the shards are somewhere else, pass `--input <dir>` in step 2.

## 2. Preprocess and pick the subset (any machine)
```bash
python -m datagen.preprocess_lmsys     # -> data/lmsys/clean_prompts.jsonl (+ .summary.json with drop counts)
python -m datagen.select_subset        # -> data/lmsys/subset_2000.jsonl (+ .summary.json with category counts)
```
Both steps stream their input and are deterministic, so the same input and seed give byte-identical
output. Check `subset_2000.summary.json`. If a category had too few prompts, its shortfall was
filled from the other categories, and the summary says so.

Categories come from the ordered keyword rules in `datagen/categories.py` ("heuristic-v1").
They're approximate on purpose. To swap them for something better, keep the signature
`assign_category(prompt) -> str`.

## 3. Dry run now with the mock backend (no GPU)
```bash
python -m datagen.generate_labels --backend mock                       # -> data/labels/mock/
python -m datagen.generate_labels --backend mock --mock-fail-rate 0.05 --backoff 0   # exercises retries
```
**Mock labels are synthetic and test-only.** Every record carries `"is_mock": true` and a
warning. `ml.data.load_jsonl` refuses them, and the mock backend can't write into the real
label directory.

## 4. On the RTX 3060 Ti
```bash
# a) environment
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# b) llama.cpp with CUDA (or use a prebuilt CUDA release from github.com/ggml-org/llama.cpp/releases)
git clone https://github.com/ggml-org/llama.cpp && cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON \
  && cmake --build llama.cpp/build --config Release -j

# c) model (any Q4_K_M GGUF of Llama-3.1-8B-Instruct; this is a common build)
hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models

# d) server: full GPU offload, 4096 context, ONE slot so our order is the execution order
llama.cpp/build/bin/llama-server -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -c 4096 -ngl 99 -np 1 --host 127.0.0.1 --port 8080

# e) smoke test: 3 prompts into a throwaway dir. Open runs.jsonl and check that the text reads as a
#    normal assistant answer (proves the chat template is applied) and that finish_reason is "stop".
python -m datagen.generate_labels --backend llamacpp --limit 3 --output-dir data/labels/smoke

# f) the real run -> data/labels/llama31_8b_q4km/ (8,000 generations)
python -m datagen.generate_labels --backend llamacpp

# g) train and evaluate on the real labels
python -m ml.train --real
python -m ml.evaluate --real
```
If the server runs on another host or port, set `LLM_BACKEND_URL=http://host:port` or pass `--url`.
`LLM_REQUEST_TIMEOUT`, `LLM_MAX_RETRIES` and `LLM_RETRY_BACKOFF_S` override the other connection settings.

## Resuming
**Rerun the same command.** Prompts already in `labels.jsonl` are skipped, and generations already
in `runs.jsonl` are reused. A crash, `Ctrl-C` or `kill -9` loses at most the one generation that
was in flight, because every line is fsynced as it's written and a half-written final line is
repaired on start-up. A prompt that still fails after `MAX_RETRIES` is logged to `failures.jsonl`
and retried on the next run. HTTP 4xx errors, such as a prompt that exceeds the context, aren't
retried. A resume with different settings (model, decoding, seeds, system prompt) is refused,
because it would mix two label sets. Use a new `--output-dir` for a new configuration.

## Output files (`data/labels/<run>/`)
| File | One line per | Purpose |
| --- | --- | --- |
| `generation_config.json` | — | The settings this label set was produced with |
| `runs.jsonl` | Successful generation | Raw measurements, including text, finish_reason, server timings |
| `labels.jsonl` | Completed prompt | Training input: 4 runs plus `target_p90_output_tokens` |
| `failures.jsonl` | Failed attempt | Error, attempt number, whether it was final |

`n_truncated_runs` counts runs that stopped at `max_new_tokens` (`finish_reason: "length"`).
Those lengths are lower bounds, not true lengths.

## Monitoring a long run
`datagen.status` only reads, tolerates a half-written final line, and is safe to run at any time
while generation is in flight:
```bash
python -m datagen.status          # progress, generations/min, ETA, truncation rate, failures
```
Throughput and ETA are derived from the timestamps in `runs.jsonl`, so they describe the whole
run rather than the current session and survive restarts.

## Incremental checkpoints
`datagen.make_checkpoint` cuts standalone training datasets from an in-progress run, so the
predictor can be trained and a learning curve measured before all 2,000 prompts finish:
```bash
python -m datagen.make_checkpoint --list     # what is ready / already cut, writes nothing
python -m datagen.make_checkpoint            # cut every ready size in config.CHECKPOINT_SIZES
```
Checkpoints are **nested**: labels are ranked by position in the subset file, not by completion
time, so `labels_250` ⊂ `labels_500` ⊂ `labels_1000` ⊂ `labels_2000` no matter how retries and
resumes interleaved. That makes the 250 → 500 → 1000 → 2000 curve a measurement of dataset size
alone. An existing checkpoint is **never overwritten** (`--force` overrides), and mock labels are
refused.

Each checkpoint is self-sufficient:
```
data/labels/llama31_8b_q4km/checkpoints/checkpoint_00500/
    labels_500.jsonl           same schema as labels.jsonl
    generation_config.json     copied from the source run
    checkpoint.summary.json    category counts, target stats, truncation counts, sha256
```
Train on one with:
```bash
python -m ml.train --real --data data/labels/llama31_8b_q4km/checkpoints/checkpoint_00500/labels_500.jsonl \
                   --output-dir artifacts/distilbert_llama31_cp500
```

## Provenance
`generation_config.json` and every label record carry `runtime_version` (the exact llama.cpp
build, e.g. `b11381-836d57176-win-cuda-12.4-x64`) alongside model, quantization, decoding settings and
`system_prompt_sha256`. A resume whose `runtime_version` differs is refused, so one label set can
never mix two runtimes. Override it with the `LLAMA_CPP_BUILD` environment variable when the
binary changes.
