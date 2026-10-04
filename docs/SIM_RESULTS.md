# Simulated scheduler results: FIFO vs SEJF vs Adaptive

> **SIMULATED ONLY.** These runs use synthetic workloads, a mock predictor (true length ×
> lognormal noise, σ = 0.25) and a mock backend (service = 60 ms + 13.3 ms/token, the decode
> rate measured on the RTX 3060 Ti). They show how the *policies* behave. They are not
> measurements of Llama 3.1 8B, of the DistilBERT predictor, or of the real system.

Reproduce everything here from the repo root:

```bash
python -m scheduler compare --n 60   --out docs/sim_results_n60.md     # small runs (demo-sized)
python -m scheduler compare --n 1000 --out docs/sim_results_n1000.md   # tail percentiles
python -m scheduler sweep            --out docs/sim_threshold_sweep.md  # threshold x load
```

Full tables: [n = 60](sim_results_n60.md) · [n = 1000](sim_results_n1000.md) · [threshold sweep](sim_threshold_sweep.md)
(JSON versions sit next to the first two).

## Findings

1. **FIFO suffers head-of-line blocking, as expected.** In `head_of_line`, the 10 short requests
   average 20.0 s under FIFO and 13.0 s under SEJF, because FIFO makes them wait behind a long job
   that arrived first.
2. **SEJF gives the lowest average latency and starves long jobs, as expected.** At n = 1000 and
   load 0.95, SEJF cuts mean latency by 24–62% against FIFO and gets short requests down to about
   5–6 s. Its worst-case wait explodes: 757 s (mostly_short), 1,159 s (balanced) and 1,803 s
   (mostly_long), against FIFO's 108–196 s. p99 gets 2.4–3.6× worse.
3. **Adaptive bounds starvation. Whether it keeps SEJF's gains depends on the threshold relative
   to load.**
   - **Small runs (n = 60, load 0.95)**, where queues stay shallow: Adaptive sits between the
     other two. In mostly_short its mean latency is 12.3 s (FIFO 15.7, SEJF 9.1) and its max wait
     25.4 s (FIFO 24.8, SEJF 55.0).
   - **Long runs at load 0.95 with the 15 s default:** the typical queue wait (FIFO mean 34–55 s) is far
     above 15 s, so nearly every request is overdue. Overdue requests run oldest-first, so
     Adaptive becomes FIFO. Its starvation bound matches FIFO's, but it keeps almost none of SEJF's
     gain (mostly_short mean 40.5 s, FIFO 41.4 s, SEJF 15.9 s).
   - **With the threshold above the typical wait**, Adaptive does what it was designed to do:

     | Workload, load | Policy | Mean latency | Short-request mean | Max wait |
     | --- | --- | --- | --- | --- |
     | mostly_short, 0.7 | FIFO | 9.6 s | 7.9 s | 43.5 s |
     | | SEJF | 6.9 s | 4.2 s | 222.1 s |
     | | Adaptive, 30 s threshold | 7.4 s | 4.9 s | **38.3 s** |
     | mostly_short, 0.85 | FIFO | 22.2 s | 20.0 s | 75.7 s |
     | | SEJF | 10.9 s | 4.9 s | 510.5 s |
     | | Adaptive, 120 s threshold | 11.5 s | 5.6 s | 129.5 s |
     | balanced, 0.85 | FIFO | 21.3 s | 17.9 s | 70.7 s |
     | | SEJF | 14.9 s | 5.1 s | 221.6 s |
     | | Adaptive, 60 s threshold | 15.3 s | 5.8 s | 77.8 s |

4. **Throughput is the same for every policy in every run.** With K = 1, no preemption and the
   same work, reordering changes *who* waits, not how much work gets done.
5. **"Starved" counts can mislead on their own.** SEJF often has fewer requests over the threshold
   than FIFO while starving a few far worse. Read the count together with max wait and p99.

## What this means for the real run (decision needed)

The 15 s default fits light load only. The right threshold depends on the real service-time
distribution and arrival rate, and neither has been measured for the scheduler yet. Options, none
implemented:

- keep 15 s and run the real demo at moderate load;
- choose the threshold from measured data (e.g. a multiple of the median Llama service time, which
  is already in the label run's `runs.jsonl`); or
- sweep it on the real backend with `python -m scheduler run`.

## Caveats

- One seed (42) per configuration. Run-to-run variation hasn't been quantified.
- At n = 1000, `head_of_line` and `bursty` become sustained overloads (their arrivals don't scale
  with n), so use their n = 60 numbers.
- p99 is flagged unstable below 100 completed requests.
- Mock prediction error is symmetric lognormal. The real predictor's errors will have their own
  shape, so SEJF and Adaptive results will move once it's plugged in.
