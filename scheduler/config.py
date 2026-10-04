"""Scheduler, simulator and metrics settings.

Decoding settings for real llama.cpp runs are NOT duplicated here: LlamaCppBackend reads them from
datagen.config, so real scheduler runs generate exactly the way the training labels were produced.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- adaptive / fairness -------------------------------------------------------
# A queued request that has waited at least this long (since arrival) is "overdue": the adaptive
# policy serves overdue requests oldest-first before falling back to shortest-estimated-first.
DEFAULT_MAX_WAIT_MS = 15_000.0
# Metrics count a request as "starved" when its queue wait exceeded this. Defaults to the same
# value so FIFO / SEJF / adaptive are judged against one threshold.
DEFAULT_STARVATION_THRESHOLD_MS = DEFAULT_MAX_WAIT_MS

# --- request size classes (metrics only; never used by a policy) ---------------
# Requests whose actual output is at least this many tokens count as "long" in per-class metrics
# when the workload does not label them itself.
LONG_TOKEN_THRESHOLD = 300

# --- simulator service-time model ----------------------------------------------
# service_time_ms = SIM_BASE_OVERHEAD_MS + output_tokens * SIM_MS_PER_TOKEN
# 13.3 ms/token is the median decode cost measured on the RTX 3060 Ti during label generation
# (~75 tok/s, see STATUS.md); the overhead covers request handling + prompt processing.
# These only shape SIMULATED runs; real runs measure service time directly.
SIM_BASE_OVERHEAD_MS = 60.0
SIM_MS_PER_TOKEN = 13.3
# Virtual time charged for one prediction in simulation (GPU DistilBERT measured ~2.8 ms median).
SIM_PREDICTOR_LATENCY_MS = 3.0

# Real backend URL / timeout / decoding settings: see datagen.config (BACKEND_URL honours the
# LLM_BACKEND_URL environment variable), shared so scheduler runs match label generation.

# --- outputs ----------------------------------------------------------------------
RUNS_DIR = PROJECT_ROOT / "scheduler_runs"  # event logs from `python -m scheduler run/simulate`
UI_SIM_DIR = PROJECT_ROOT / "ui" / "public" / "sim"  # bundled simulated runs for the dashboard
