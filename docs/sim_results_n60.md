# Simulated policy comparison

> SIMULATED: synthetic workloads, mock predictor, mock backend (service = base overhead + tokens x ms/token). Validates policy behaviour only; not a measurement of Llama or of the predictor.

Settings: seed 42, offered load 0.95, prediction noise sigma 0.25 (lognormal), adaptive max wait 15.0 s (also the starvation threshold), service = 60 ms + 13.3 ms/token. Short = 20-150 tokens, long = 400-1000 tokens.

### mostly_short (60 requests: 44 short / 16 long)

80% short / 20% long requests, Poisson arrivals

| Metric | FIFO | SEJF | Adaptive |
| --- | --- | --- | --- |
| Mean latency (s) | 15.68 | 9.08 | 12.27 |
| p50 latency (s) | 15.16 | 6.29 | 8.32 |
| p95 latency (s) | 26.35 | 28.25 | 29.29 |
| p99 latency (s) (unstable, n<100) | 32.22 | 58.19 | 33.08 |
| Mean queue wait (s) | 12.68 | 6.07 | 9.26 |
| Max queue wait (s) | 24.83 | 54.96 | 25.40 |
| Short-request mean latency (s) | 14.50 | 4.69 | 9.31 |
| Long-request mean latency (s) | 18.93 | 21.13 | 20.40 |
| Long-request max wait (s) | 24.75 | 54.96 | 25.40 |
| Starved (wait > threshold) | 24 | 4 | 18 |
| Throughput, sanity check (req/min) | 18.11 | 18.11 | 18.11 |

### balanced (60 requests: 26 short / 34 long)

50% short / 50% long requests, Poisson arrivals

| Metric | FIFO | SEJF | Adaptive |
| --- | --- | --- | --- |
| Mean latency (s) | 43.67 | 27.17 | 43.29 |
| p50 latency (s) | 44.23 | 10.41 | 44.23 |
| p95 latency (s) | 65.63 | 126.78 | 65.63 |
| p99 latency (s) (unstable, n<100) | 70.08 | 205.22 | 70.08 |
| Mean queue wait (s) | 37.74 | 21.24 | 37.36 |
| Max queue wait (s) | 63.18 | 236.25 | 63.18 |
| Short-request mean latency (s) | 40.05 | 5.66 | 39.13 |
| Long-request mean latency (s) | 46.44 | 43.63 | 46.46 |
| Long-request max wait (s) | 59.12 | 236.25 | 59.12 |
| Starved (wait > threshold) | 55 | 15 | 55 |
| Throughput, sanity check (req/min) | 9.97 | 9.97 | 9.97 |

### mostly_long (60 requests: 15 short / 45 long)

20% short / 80% long requests, Poisson arrivals

| Metric | FIFO | SEJF | Adaptive |
| --- | --- | --- | --- |
| Mean latency (s) | 25.46 | 21.07 | 24.72 |
| p50 latency (s) | 19.31 | 13.27 | 19.01 |
| p95 latency (s) | 64.05 | 81.97 | 64.05 |
| p99 latency (s) (unstable, n<100) | 68.06 | 116.55 | 68.06 |
| Mean queue wait (s) | 18.08 | 13.69 | 17.34 |
| Max queue wait (s) | 62.94 | 110.38 | 62.94 |
| Short-request mean latency (s) | 18.08 | 7.44 | 15.02 |
| Long-request mean latency (s) | 27.92 | 25.61 | 27.95 |
| Long-request max wait (s) | 62.94 | 110.38 | 62.94 |
| Starved (wait > threshold) | 30 | 10 | 29 |
| Throughput, sanity check (req/min) | 6.62 | 6.62 | 6.62 |

### head_of_line (12 requests: 10 short / 2 long)

a long job is running; another long request queues first, short ones arrive behind it

| Metric | FIFO | SEJF | Adaptive |
| --- | --- | --- | --- |
| Mean latency (s) | 18.52 | 13.60 | 14.33 |
| p50 latency (s) | 19.26 | 12.70 | 12.70 |
| p95 latency (s) | 24.06 | 21.61 | 24.00 |
| p99 latency (s) (unstable, n<100) | 24.25 | 24.52 | 24.32 |
| Mean queue wait (s) | 16.41 | 11.49 | 12.22 |
| Max queue wait (s) | 23.76 | 19.47 | 23.17 |
| Short-request mean latency (s) | 20.03 | 12.98 | 14.14 |
| Long-request mean latency (s) | 10.99 | 16.69 | 15.27 |
| Long-request max wait (s) | 8.07 | 19.47 | 16.64 |
| Starved (wait > threshold) | 9 | 3 | 3 |
| Throughput, sanity check (req/min) | 28.46 | 28.46 | 28.46 |

### bursty (80 requests: 56 short / 24 long)

mixed background traffic, then a burst of short requests within one second

| Metric | FIFO | SEJF | Adaptive |
| --- | --- | --- | --- |
| Mean latency (s) | 47.54 | 22.03 | 46.64 |
| p50 latency (s) | 49.31 | 10.01 | 49.31 |
| p95 latency (s) | 76.53 | 62.27 | 76.53 |
| p99 latency (s) (unstable, n<100) | 86.27 | 201.82 | 86.27 |
| Mean queue wait (s) | 43.85 | 18.33 | 42.95 |
| Max queue wait (s) | 82.08 | 255.95 | 82.08 |
| Short-request mean latency (s) | 46.07 | 9.20 | 44.58 |
| Long-request mean latency (s) | 50.98 | 51.94 | 51.45 |
| Long-request max wait (s) | 82.08 | 255.95 | 82.08 |
| Starved (wait > threshold) | 73 | 20 | 69 |
| Throughput, sanity check (req/min) | 15.55 | 15.55 | 15.55 |

