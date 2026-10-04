# Policy comparison: test_prompts

**REAL MEASUREMENTS (llama.cpp)**

Manifest `dc37a5a86f129b1e`, 112 requests. Adaptive max wait: 14.1 s = 3 x median service 4.69 s (data\labels\llama31_8b_q4km\runs.jsonl, n=3001). Short/long split: workload labels, else actual output >= 300 tokens.

| Check | Result | Detail |
| --- | --- | --- |
| same manifest_id | PASS | dc37a5a86f129b1e |
| same backend | PASS | llamacpp |
| same generation settings | PASS | datagen.config |
| same max wait / starvation threshold | PASS | 14079.436500003794 ms |
| one run per policy | PASS | fifo, sejf, adaptive |
| same request set | PASS | 112 requests |
| same arrival timestamps | PASS | from the manifest |
| same predictor output per prompt | PASS | max difference 0.0000 tokens |

| Metric | FIFO | SEJF | ADAPTIVE |
| --- | --- | --- | --- |
| Mean latency (s) | 16.65 | 13.36 | 15.75 |
| p50 latency (s) | 12.27 | 8.78 | 10.62 |
| p95 latency (s) | 48.70 | 51.65 | 44.18 |
| p99 latency (s) | 57.76 | 80.56 | 51.80 |
| Mean queue wait (s) | 10.81 | 7.66 | 10.13 |
| Max queue wait (s) | 54.75 | 72.17 | 48.78 |
| Short-request mean latency (s) | 14.02 | 7.03 | 13.66 |
| Long-request mean latency (s) | 18.23 | 17.45 | 17.11 |
| Long-request max wait (s) | 54.32 | 72.17 | 48.37 |
| Starved (wait > threshold) | 28 | 18 | 35 |
| Throughput, sanity check (req/min) | 7.94 | 7.94 | 7.95 |
| Failed requests | 0 | 0 | 0 |
