"""Central configuration for the real label-generation experiment.

Every value used by preprocessing, subset selection and label generation lives here. CLI flags
may override a few of them per run, and backend connection values can also be set with
environment variables, so nothing needs editing when moving to the RTX 3060 Ti machine.

Items marked DECISION are placeholders that still need an explicit project decision.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# --- experiment ---------------------------------------------------------------
NUM_PROMPTS = 500
NUM_GENERATIONS = 4
SEED = 42
# One distinct, deterministic sampling seed per repeated generation (same set for every prompt).
GENERATION_SEEDS = [SEED + i for i in range(NUM_GENERATIONS)]
TARGET_PERCENTILE = 90  # target = this percentile of the NUM_GENERATIONS output-token counts
PERCENTILE_METHOD = "linear"  # numpy percentile method; with 4 runs p90 = x3 + 0.7 * (x4 - x3)
TARGET_FIELD = f"target_p{TARGET_PERCENTILE}_output_tokens"  # field the DistilBERT pipeline trains on

# --- model / decoding ---------------------------------------------------------
MODEL_NAME = "Llama-3.1-8B-Instruct"
QUANTIZATION = "Q4_K_M"
RUNTIME = "llama.cpp"
HARDWARE = "RTX 3060 Ti 8GB"
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 512
# DECISION: exact fixed system prompt. Its SHA-256 is recorded with every label so a change is detectable.
SYSTEM_PROMPT = "You are a helpful assistant."
SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

# --- backend ------------------------------------------------------------------
# Any OpenAI-compatible chat endpoint works (llama.cpp `llama-server` is the target).
BACKEND_URL = os.environ.get("LLM_BACKEND_URL", "http://127.0.0.1:8080")
REQUEST_TIMEOUT = float(os.environ.get("LLM_REQUEST_TIMEOUT", "300"))  # seconds per generation
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))  # extra attempts after the first
RETRY_BACKOFF_S = float(os.environ.get("LLM_RETRY_BACKOFF_S", "2.0"))  # doubles each retry

# --- dataset ------------------------------------------------------------------
DATASET_NAME = "lmsys-chat-1m"
DATASET_HF_REPO = "lmsys/lmsys-chat-1m"
LMSYS_RAW_DIR = DATA_DIR / "raw" / "lmsys-chat-1m"  # where `hf download` puts the parquet shards
CLEAN_PROMPTS_PATH = DATA_DIR / "lmsys" / "clean_prompts.jsonl"
SUBSET_PATH = DATA_DIR / "lmsys" / f"subset_{NUM_PROMPTS}.jsonl"

LANGUAGE = "English"
MIN_PROMPT_CHARS = 1  # after stripping whitespace; anything shorter is treated as empty
# Keeps prompt + system prompt + MAX_NEW_TOKENS inside a 4096-token llama.cpp context (~4 chars/token).
MAX_PROMPT_CHARS = 8000
# DECISION: dropping prompts the OpenAI moderation pass flagged keeps the workload closer to an
# enterprise assistant. Set False to keep them.
EXCLUDE_MODERATION_FLAGGED = True

CATEGORY_PROPORTIONS = {
    "knowledge_factual": 0.20,
    "explanation_reasoning": 0.20,
    "coding_debugging": 0.15,
    "summarization_transformation": 0.15,
    "writing_structured": 0.15,
    "troubleshooting_advice": 0.15,
}

# --- label output -------------------------------------------------------------
LABELS_DIR_REAL = DATA_DIR / "labels" / "llama31_8b_q4km"
LABELS_DIR_MOCK = DATA_DIR / "labels" / "mock"
LABELS_FILE = "labels.jsonl"  # one record per fully completed prompt (training input)
RUNS_FILE = "runs.jsonl"  # one record per successful generation (raw measurements, includes text)
FAILURES_FILE = "failures.jsonl"  # one record per failed attempt
GENERATION_CONFIG_FILE = "generation_config.json"
REAL_LABELS_PATH = LABELS_DIR_REAL / LABELS_FILE


def generation_config(backend_name: str) -> dict:
    """The settings that define one label set. A resumed run must match these exactly."""
    return {
        "model": MODEL_NAME,
        "quantization": QUANTIZATION,
        "runtime": RUNTIME,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_generations": NUM_GENERATIONS,
        "seeds": list(GENERATION_SEEDS),
        "target_percentile": TARGET_PERCENTILE,
        "percentile_method": PERCENTILE_METHOD,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "backend": backend_name,
    }
