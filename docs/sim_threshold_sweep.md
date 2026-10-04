> SIMULATED: synthetic workloads, mock predictor, mock backend (service = base overhead + tokens x ms/token). Validates policy behaviour only; not a measurement of Llama or of the predictor.

1000 requests per run, seed 42, prediction noise sigma 0.25.

| Workload | Load | Policy | Max wait param (s) | Mean (s) | p50 (s) | p95 (s) | p99 (s) | Max wait (s) | Short mean (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mostly_short | 0.7 | FIFO | - | 9.6 | 7.5 | 27.3 | 37.7 | 43.5 | 7.9 |
| mostly_short | 0.7 | SEJF | - | 6.9 | 4.1 | 19.9 | 37.4 | 222.1 | 4.2 |
| mostly_short | 0.7 | Adaptive | 15 | 8.4 | 5.4 | 29.0 | 37.4 | 43.5 | 6.2 |
| mostly_short | 0.7 | Adaptive | 30 | 7.4 | 4.2 | 24.6 | 43.3 | 38.3 | 4.9 |
| mostly_short | 0.7 | Adaptive | 60 | 7.0 | 4.1 | 21.4 | 44.5 | 68.0 | 4.3 |
| mostly_short | 0.7 | Adaptive | 120 | 6.9 | 4.1 | 20.0 | 42.4 | 126.2 | 4.3 |
| mostly_short | 0.85 | FIFO | - | 22.2 | 16.1 | 57.5 | 67.9 | 75.7 | 20.0 |
| mostly_short | 0.85 | SEJF | - | 10.9 | 5.4 | 29.2 | 99.1 | 510.5 | 4.9 |
| mostly_short | 0.85 | Adaptive | 15 | 21.1 | 12.0 | 57.5 | 67.9 | 75.7 | 18.4 |
| mostly_short | 0.85 | Adaptive | 30 | 19.5 | 9.3 | 57.6 | 67.9 | 75.7 | 16.2 |
| mostly_short | 0.85 | Adaptive | 60 | 12.9 | 6.1 | 68.9 | 79.9 | 75.5 | 7.2 |
| mostly_short | 0.85 | Adaptive | 120 | 11.5 | 5.7 | 38.0 | 134.1 | 129.5 | 5.6 |
| mostly_short | 0.95 | FIFO | - | 41.4 | 26.4 | 112.7 | 121.8 | 125.9 | 38.9 |
| mostly_short | 0.95 | SEJF | - | 15.9 | 5.7 | 37.2 | 292.7 | 756.6 | 5.1 |
| mostly_short | 0.95 | Adaptive | 15 | 40.5 | 26.7 | 112.7 | 121.8 | 125.9 | 37.6 |
| mostly_short | 0.95 | Adaptive | 30 | 38.8 | 19.6 | 112.7 | 121.8 | 125.9 | 35.2 |
| mostly_short | 0.95 | Adaptive | 60 | 34.7 | 10.5 | 112.7 | 121.8 | 125.9 | 29.7 |
| mostly_short | 0.95 | Adaptive | 120 | 21.8 | 7.5 | 131.3 | 139.8 | 138.7 | 11.7 |
| balanced | 0.7 | FIFO | - | 13.3 | 11.6 | 33.8 | 42.3 | 44.4 | 9.7 |
| balanced | 0.7 | SEJF | - | 10.7 | 8.5 | 28.6 | 60.4 | 93.4 | 4.5 |
| balanced | 0.7 | Adaptive | 15 | 12.1 | 9.7 | 34.0 | 42.4 | 44.4 | 7.1 |
| balanced | 0.7 | Adaptive | 30 | 11.1 | 8.8 | 35.0 | 48.1 | 48.4 | 5.2 |
| balanced | 0.7 | Adaptive | 60 | 10.8 | 8.6 | 28.7 | 60.4 | 70.0 | 4.7 |
| balanced | 0.7 | Adaptive | 120 | 10.7 | 8.5 | 28.6 | 60.4 | 93.4 | 4.5 |
| balanced | 0.85 | FIFO | - | 21.3 | 19.4 | 47.7 | 63.1 | 70.7 | 17.9 |
| balanced | 0.85 | SEJF | - | 14.9 | 9.2 | 53.9 | 104.1 | 221.6 | 5.1 |
| balanced | 0.85 | Adaptive | 15 | 19.9 | 16.1 | 47.7 | 63.1 | 70.7 | 15.0 |
| balanced | 0.85 | Adaptive | 30 | 17.3 | 11.7 | 49.4 | 63.2 | 70.7 | 9.6 |
| balanced | 0.85 | Adaptive | 60 | 15.3 | 9.9 | 61.2 | 79.4 | 77.8 | 5.8 |
| balanced | 0.85 | Adaptive | 120 | 14.9 | 9.2 | 53.9 | 106.1 | 130.0 | 5.1 |
| balanced | 0.95 | FIFO | - | 39.3 | 35.2 | 91.5 | 104.5 | 108.0 | 36.3 |
| balanced | 0.95 | SEJF | - | 23.4 | 9.9 | 86.0 | 246.6 | 1159.0 | 5.7 |
| balanced | 0.95 | Adaptive | 15 | 38.4 | 35.0 | 91.5 | 104.5 | 108.0 | 34.3 |
| balanced | 0.95 | Adaptive | 30 | 35.9 | 32.0 | 91.5 | 104.5 | 108.0 | 28.9 |
| balanced | 0.95 | Adaptive | 60 | 30.4 | 15.4 | 92.4 | 104.5 | 108.0 | 17.6 |
| balanced | 0.95 | Adaptive | 120 | 25.0 | 11.5 | 132.8 | 142.1 | 141.1 | 7.6 |
