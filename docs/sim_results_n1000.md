# Simulated policy comparison

> SIMULATED: synthetic workloads, mock predictor, mock backend (service = base overhead + tokens x ms/token). Validates policy behaviour only; not a measurement of Llama or of the predictor.

Settings: seed 42, offered load 0.95, prediction noise sigma 0.25 (lognormal), adaptive max wait 15.0 s (also the starvation threshold), service = 60 ms + 13.3 ms/token. Short = 20-150 tokens, long = 400-1000 tokens.

### mostly_short (1000 requests: 794 short / 206 long)

80% short / 20% long requests, Poisson arrivals

| Metric | FIFO | SEJF | Adaptive |
| --- | --- | --- | --- |
| Mean latency (s) | 41.45 | 15.91 | 40.51 |
| p50 latency (s) | 26.42 | 5.73 | 26.73 |
| p95 latency (s) | 112.68 | 37.24 | 112.68 |
| p99 latency (s) | 121.76 | 292.67 | 121.76 |
| Mean queue wait (s) | 38.59 | 13.06 | 37.66 |
| Max queue wait (s) | 125.90 | 756.57 | 125.90 |
| Short-request mean latency (s) | 38.92 | 5.10 | 37.61 |
| Long-request mean latency (s) | 51.18 | 57.59 | 51.68 |
| Long-request max wait (s) | 118.23 | 756.57 | 118.23 |
| Starved (wait > threshold) | 611 | 92 | 583 |
| Throughput, sanity check (req/min) | 18.87 | 18.87 | 18.87 |

### balanced (1000 requests: 488 short / 512 long)

50% short / 50% long requests, Poisson arrivals

| Metric | FIFO | SEJF | Adaptive |
| --- | --- | --- | --- |
| Mean latency (s) | 39.35 | 23.39 | 38.42 |
| p50 latency (s) | 35.21 | 9.93 | 34.99 |
| p95 latency (s) | 91.52 | 86.00 | 91.52 |
| p99 latency (s) | 104.51 | 246.59 | 104.51 |
| Mean queue wait (s) | 34.03 | 18.06 | 33.09 |
| Max queue wait (s) | 108.03 | 1159.00 | 108.03 |
| Short-request mean latency (s) | 36.31 | 5.69 | 34.34 |
| Long-request mean latency (s) | 42.25 | 40.25 | 42.30 |
| Long-request max wait (s) | 102.55 | 1159.00 | 102.55 |
| Starved (wait > threshold) | 720 | 166 | 697 |
| Throughput, sanity check (req/min) | 10.53 | 10.53 | 10.53 |

### mostly_long (1000 requests: 190 short / 810 long)

20% short / 80% long requests, Poisson arrivals

| Metric | FIFO | SEJF | Adaptive |
| --- | --- | --- | --- |
| Mean latency (s) | 62.54 | 47.50 | 62.08 |
| p50 latency (s) | 40.33 | 14.43 | 40.44 |
| p95 latency (s) | 171.13 | 141.72 | 171.13 |
| p99 latency (s) | 192.90 | 689.12 | 192.90 |
| Mean queue wait (s) | 54.69 | 39.66 | 54.24 |
| Max queue wait (s) | 195.92 | 1802.89 | 195.92 |
| Short-request mean latency (s) | 49.90 | 6.22 | 48.08 |
| Long-request mean latency (s) | 65.50 | 57.19 | 65.37 |
| Long-request max wait (s) | 195.92 | 1802.89 | 195.92 |
| Starved (wait > threshold) | 707 | 258 | 706 |
| Throughput, sanity check (req/min) | 7.03 | 7.03 | 7.03 |

### head_of_line (168 requests: 166 short / 2 long)

a long job is running; another long request queues first, short ones arrive behind it

| Metric | FIFO | SEJF | Adaptive |
| --- | --- | --- | --- |
| Mean latency (s) | 105.46 | 83.31 | 100.90 |
| p50 latency (s) | 104.75 | 72.86 | 101.07 |
| p95 latency (s) | 191.70 | 192.60 | 191.70 |
| p99 latency (s) | 200.18 | 203.42 | 200.18 |
| Mean queue wait (s) | 104.16 | 82.01 | 99.60 |
| Max queue wait (s) | 200.74 | 212.88 | 200.74 |
| Short-request mean latency (s) | 106.60 | 82.95 | 101.94 |
| Long-request mean latency (s) | 10.99 | 113.39 | 14.64 |
| Long-request max wait (s) | 8.07 | 212.88 | 15.37 |
| Starved (wait > threshold) | 165 | 139 | 152 |
| Throughput, sanity check (req/min) | 46.09 | 46.09 | 46.09 |

### bursty (1333 requests: 947 short / 386 long)

mixed background traffic, then a burst of short requests within one second

| Metric | FIFO | SEJF | Adaptive |
| --- | --- | --- | --- |
| Mean latency (s) | 238.12 | 118.27 | 237.82 |
| p50 latency (s) | 249.36 | 12.74 | 249.36 |
| p95 latency (s) | 503.64 | 448.58 | 503.64 |
| p99 latency (s) | 546.47 | 1708.80 | 546.47 |
| Mean queue wait (s) | 234.58 | 114.73 | 234.28 |
| Max queue wait (s) | 557.19 | 3477.00 | 557.19 |
| Short-request mean latency (s) | 239.98 | 69.21 | 239.55 |
| Long-request mean latency (s) | 233.54 | 238.62 | 233.56 |
| Long-request max wait (s) | 552.16 | 3477.00 | 552.16 |
| Starved (wait > threshold) | 1277 | 528 | 1267 |
| Throughput, sanity check (req/min) | 16.92 | 16.92 | 16.92 |

