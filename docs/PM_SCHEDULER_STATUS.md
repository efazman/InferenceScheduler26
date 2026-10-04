# Scheduler & Dashboard: Project Status

*As of 2026-10-04 · scheduler workstream*

**The scheduler, the demo dashboard and the Tiger Data connection are built and tested, and the
dashboard is polished for judging.** They can switch to the real predictor and the real Llama
model as soon as the label run on the GPU machine finishes, with no new engineering. The one open
setting, the adaptive policy's fairness threshold, is decided (see below).

## Where things stand

| Piece | Status | What it gives us |
| --- | --- | --- |
| Scheduler with three policies | Done | Decides which request runs next: first-come-first-served (FIFO), shortest-job-first (SEJF), or our adaptive policy |
| Simulator | Done | Replays realistic traffic patterns in seconds, with no GPU needed |
| Event log and metrics | Done | Records every step of every request: wait time, latency, throughput, starvation |
| Demo dashboard | Done, demo-ready | Shows the scheduling story at a glance; real results slot in without changes (details below) |
| Connection to the real Llama model | Built, not yet run | Tested against a stand-in server. The real run needs the GPU machine. |
| Connection to the real predictor | Built, not yet run | Works with whichever predictor wins (DistilBERT or the simple baseline) |
| Tiger Data (analytics database) | Done, waiting for real data | Connected and tested. The results table exists and is empty until the real runs. |

All of this is on a separate branch, so it can't interfere with the label run on the GPU machine.
It merges cleanly with the latest main code. 191 automated checks pass (185 backend, 6 dashboard).

## Tiger Data

| Item | Status |
| --- | --- |
| Connection to our Tiger Cloud database (`db-inference`) | **Working.** Verified from the Mac. |
| Results table (`scheduler_events`) | **Created, and empty.** Real results go in once the GPU runs happen. |
| Sending results | Two ways: live while each run executes, or uploaded afterwards. Uploading twice never double-counts. |
| Telling real from test data | Every row is labelled *real*, *mock* or *simulated*, so test data can't pass for results. Only real runs are uploaded by default. |
| Effect on the demo if the database is down | **None.** Runs never wait on the database, every result is also saved locally, and anything missed can be uploaded later. |
| Credential safety | The database password is excluded from git and has been checked absent from everything pushed to GitHub. An automated check fails if a password ever gets committed. |
| Remaining step | Someone copies the credentials file to the GPU machine **by hand** (USB or a password manager, not git or chat). This takes about 5 minutes. |

No test or made-up data was put in the results table. The connection test used a temporary table
and deleted it afterwards.

## Dashboard (frontend)

**Status: demo-ready.** A judge sees the point within seconds of opening it, without clicking
anything.

**What it shows when it opens:** the same requests under all three policies, at the same moment,
stacked one above the other:

- **FIFO:** a long request is next, with every short one stuck behind it (`LONG | SHORT | SHORT | SHORT`).
- **SEJF:** the short requests go first (`SHORT | SHORT | SHORT | LONG`).
- **Adaptive:** short requests go first, but any request that has waited too long is labelled
  **⚠ OVERDUE** and moved to the front.

| What | Status |
| --- | --- |
| Opening screen | Starts on the head-of-line example, paused at the moment FIFO is about to run the long request. Everything important fits on one laptop screen. |
| Queue view | Per policy: which request is on the GPU, the queue in the order it will be served (next one marked), each request's predicted size and wait, and running totals |
| One-click story | Three buttons jump to the key moments: **1 · Head-of-line**, **2 · Overdue promotion**, **3 · Results**. The moments are found in the data, not staged. |
| Traffic presets | Head-of-line blocking (default), Mostly short, Balanced, Mostly long, Bursty |
| Results table | FIFO vs SEJF vs Adaptive on average, typical and slow-case latency, wait times, short- vs long-request latency and starvation, with the best value in each row ticked. Throughput is shown as a sanity check only. |
| Fairness setting | Always shown, with where it came from (for real runs: "3 × median model answer time") |
| Real vs test labelling | Every screen carries a **SIMULATED**, **MOCK** or **REAL** badge. Anything unlabelled shows **UNKNOWN**, never real. Live, recorded and stopped runs are labelled too. |
| Real-run tab | Ready. Says **NO REAL RUNS YET** until the GPU runs exist, then shows them with the same views, plus which model and predictor were used |
| Tiger Data history | Shows stored runs. If the database is down, the dashboard says so and keeps working from local files. |
| Checked | Builds cleanly, no browser errors, fits laptop and desktop screens, 6 automated checks |

**Demo script:** a 60–90 second walkthrough with direct links for each screenshot is in
`docs/DEMO_FLOW.md`. No real-run screenshots exist yet. Those wait for the GPU runs, and nothing
is faked in the meantime.

## Results so far (simulated, not real measurements)

These come from the simulator, using made-up traffic and a stand-in model. They show how the
policies behave, not how fast the real system is.

| | FIFO | SEJF | Adaptive (default setting) |
| --- | --- | --- | --- |
| Short requests | Stuck behind long ones | **Much faster** (about 5 s versus 39 s for FIFO) | Depends on how busy the system is |
| Longest wait anyone suffers | Moderate (2 min) | **Very bad** (up to 13 min) | Same as FIFO (2 min) |
| Requests completed per minute | Same | Same | Same |

*Busy-system run: 1,000 requests, mostly short, 95% load.*

- **SEJF** is fastest on average but can leave a long request waiting for many minutes. That's the
  starvation problem the adaptive policy exists to solve.
- **Adaptive** always prevents that starvation. Whether it also keeps SEJF's speed depends on its
  one setting, the maximum wait before a request jumps the queue.
- **Reordering never changes throughput.** It only changes *who* waits.

## The fairness setting (now decided)

The adaptive policy's maximum-wait setting defaults to **15 seconds**. That only works when the
system is lightly loaded:

| How busy | Adaptive at 15 s | Adaptive with the setting raised |
| --- | --- | --- |
| Light (70%) | Between FIFO and SEJF, with no starvation (8.4 s average, against 9.6 s for FIFO and 6.9 s for SEJF) | **At 30 s:** nearly SEJF's speed (7.4 s), and a shorter longest wait than even FIFO |
| Busy (85%) | Barely better than FIFO, because almost everyone waits over 15 s and counts as overdue | **At 60–120 s:** close to SEJF's speed, with the longest wait still capped |
| Very busy (95%) | Behaves just like FIFO | **At 120 s:** recovers much of SEJF's speed (21.8 s average, against 41.4 s for FIFO and 15.9 s for SEJF) |

**Decided:** for real runs, the maximum wait is **3 × the measured median answer time** of the
real model, taken from the label run's existing measurements. The scheduler records the value it
used, and the dashboard shows it. Other multipliers (2×, 5×) can be tried later.

## Blockers

| Blocker | What it holds up | What unblocks it |
| --- | --- | --- |
| The GPU machine is busy with the label run | Real-model runs, and the merge itself | The label run and predictor training finishing |
| The branch needs merging on the GPU machine | Real-model runs | The label run finishing; steps are in docs/MERGE_PLAN.md |
| The GPU machine doesn't have the Tiger Data credentials yet | Sending real results to Tiger Data (not the runs themselves) | Copy the credentials file over by hand |

## Next steps once the GPU machine is free

1. Merge the scheduler branch into main.
2. Plug in the winning predictor and point the scheduler at the running Llama server.
3. Take the held-out test prompts the predictor never saw, and run each policy on the same
   prompts in the same order.
4. Set the maximum wait (see the decision above) and record the runs.
5. Send the results to Tiger Data: live during the runs, or uploaded right after. The connection and table are ready.
6. Demo: follow `docs/DEMO_FLOW.md`. Use the simulation tab for the story and the recorded tab for the real results, then take the real-run screenshots.

## Risks

| Risk | Impact | Fallback |
| --- | --- | --- |
| The real predictor is less accurate than the stand-in | SEJF and Adaptive gain less | The baseline predictor plugs in the same way, and the results will show the difference honestly |
| A run on the GPU machine during label generation | It slows or stalls the label run (this happened once already) | Only run after generation finishes |
| Real answer times differ from the simulator's assumptions | Simulated results don't carry over exactly | The real runs replace the simulated numbers; the simulator only explains the behaviour |
| Tiger Data unreachable during the demo | The live database copy lags | Runs never depend on it; results are saved locally and can be uploaded afterwards |
