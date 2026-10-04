# Scheduler & Dashboard: Project Status

*As of 2026-10-04 · scheduler workstream*

**The scheduler, the demo dashboard and the Tiger Data connection are built and tested.** They
can switch to the real predictor and the real Llama model as soon as the label run on the GPU
machine finishes, with no new engineering. The one open setting, the adaptive policy's fairness
threshold, is now decided (see below).

## Where things stand

| Piece | Status | What it gives us |
| --- | --- | --- |
| Scheduler with three policies | Done | Decides which request runs next: first-come-first-served (FIFO), shortest-job-first (SEJF), or our adaptive policy |
| Simulator | Done | Replays realistic traffic patterns in seconds, with no GPU needed |
| Event log and metrics | Done | Records every step of every request: wait time, latency, throughput, starvation |
| Demo dashboard | Done | Shows the queue reordering live, side by side for all three policies |
| Connection to the real Llama model | Built, not yet run | Tested against a stand-in server. The real run needs the GPU machine. |
| Connection to the real predictor | Built, not yet run | Works with whichever predictor wins (DistilBERT or the simple baseline) |
| Tiger Data (analytics database) | Done, waiting for real data | Connected and tested. The results table exists and is empty until the real runs. |

All of this is on a separate branch, so it can't interfere with the label run on the GPU machine.
It merges cleanly with the latest main code. 185 automated checks pass.

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

## What the demo shows

The dashboard puts the same requests under all three policies, at the same moment, one above the
other:

- **FIFO:** a long request at the front blocks every short one behind it (`LONG | SHORT | SHORT | SHORT`).
- **SEJF:** the short requests go first (`SHORT | SHORT | SHORT | LONG`).
- **Adaptive:** short requests go first until someone has waited too long, then that request is
  served (it turns red on screen).

Below that sit a timeline of which request held the GPU when, live metrics, and an end-of-run
comparison table. A second tab plays back recorded runs, and shows real runs live while they
execute.

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
6. Demo: the simulation tab for the story, and the recorded tab for the real runs.

## Risks

| Risk | Impact | Fallback |
| --- | --- | --- |
| The real predictor is less accurate than the stand-in | SEJF and Adaptive gain less | The baseline predictor plugs in the same way, and the results will show the difference honestly |
| A run on the GPU machine during label generation | It slows or stalls the label run (this happened once already) | Only run after generation finishes |
| Real answer times differ from the simulator's assumptions | Simulated results don't carry over exactly | The real runs replace the simulated numbers; the simulator only explains the behaviour |
| Tiger Data unreachable during the demo | The live database copy lags | Runs never depend on it; results are saved locally and can be uploaded afterwards |
