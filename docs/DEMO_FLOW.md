# Demo flow (60–90 s)

```bash
cd ui && npm install && npm run build && npm start   # http://localhost:3000
```
The page opens on the **Head-of-line blocking** preset, already paused at the head-of-line moment.

| # | Time | Do | Say |
| --- | --- | --- | --- |
| 1 | 0:00 | Open `localhost:3000`. Point at the **SIMULATED** badge and the 4-line explainer. | "Same requests go to all three policies. The predictor estimates each request's length first, and one GPU serves one request at a time." |
| 2 | 0:10 | Point at the **FIFO** lane: the long `r1` is marked **NEXT** with short requests behind it. | "FIFO is about to run a long job while eight short ones wait. That's head-of-line blocking." |
| 3 | 0:20 | Point at **SEJF** and **Adaptive**: shorts first, `r1` last. Press **Play**. | "Shortest-estimated-first moves short jobs forward." |
| 4 | 0:30 | Click **2 · Overdue promotion**. SEJF's `r1` waits behind shorter jobs, while Adaptive shows `r1 ⚠ OVERDUE ↑` at the front. | "SEJF can starve long jobs. Adaptive keeps the short-job preference, but once a request waits past MAX_WAIT it goes first." |
| 5 | 0:45 | Click **Mostly long** and scrub forward. SEJF's queue fills with red, overdue long requests. | "Under heavy long traffic, SEJF's worst wait explodes." |
| 6 | 0:55 | Click **3 · Results** and scroll to **End-of-run comparison** (✓ marks the best value in each row). | "SEJF wins on average latency; Adaptive caps the worst wait. Throughput is identical, because reordering moves waiting around without changing capacity." |
| 7 | 1:05 | Switch to **Recorded / live runs**. | "These are recorded runs. A REAL badge means real Llama 3.1 8B on the RTX 3060 Ti; MOCK means a stand-in." |
| 8 | 1:15 | *(Once real runs exist)* Pick a `real-*` run. The same-manifest comparison shows the three policies on the identical arrival trace, with `MAX_WAIT = 3 × median service time`. | "Here are the measured results: the real predictor, real Llama, real latency." |
| 9 | 1:25 | Scroll to **Run history in Tiger Data**. | "Every run is also stored in Tiger Data. If the database is down, the dashboard falls back to local files." |

Until the GPU runs exist, steps 8–9 show honest empty states: **NO REAL RUNS YET**, and
"Connected. No runs stored yet". Don't fill them with mock data.

## Screenshot links

| Shot | URL (simulated unless noted) |
| --- | --- |
| FIFO head-of-line blocking | `/` (opens there) or `/?workload=head_of_line` |
| SEJF reordered queue | same frame: the SEJF lane |
| Adaptive fairness promotion | `/` then **2 · Overdue promotion** |
| SEJF starvation | `/?workload=mostly_long&t=130000` |
| Final metrics comparison | **3 · Results**, then the End-of-run comparison card |
| Real-run dashboard | `/?mode=runs&run=real-adaptive` *(after the GPU run; badge must read REAL)* |
| Tiger Data history | bottom of `/?mode=runs` |

Every simulated frame carries the **SIMULATED** badge, and every mock run the **MOCK** badge. Only
runs recorded on the GPU machine show **REAL**.
