# Adaptive LLM Inference Scheduler — Project Context

## 1. Project Goal

Build an adaptive scheduler for a locally hosted large language model (LLM) serving workload.

The project is intended to demonstrate that request scheduling can materially improve inference responsiveness and efficiency when many heterogeneous prompts compete for limited compute.

The main idea is:

1. inspect an incoming prompt before generation,
2. estimate how expensive the request is likely to be,
3. use that estimate to decide when the request should run,
4. execute the request on a constrained local inference backend,
5. measure the effect of different scheduling policies on latency, throughput, and fairness.

This is a systems project first and an ML project second.

The ML component exists to provide the scheduler with a useful estimate of likely request cost. The scheduler and measurement system are the core of the project.

---

## 2. Intended Real-World Use Case

The target scenario is a private or self-hosted enterprise assistant running on constrained compute.

The system should represent a realistic mixed workload rather than one narrow academic domain.

Representative request types include:

- internal knowledge questions,
- document or RAG-style questions,
- troubleshooting and support prompts,
- summarization,
- coding assistance,
- general reasoning,
- structured generation.

The project should not require building a full enterprise assistant, RAG stack, or document-ingestion system.

Instead, it models the generation side of such a system: heterogeneous prompts arrive, compete for compute, and must be scheduled intelligently.

---

## 3. High-Level System Architecture

The project has four primary components:

```text
Incoming Prompt
      |
      v
Prompt Cost Estimator
      |
      v
Adaptive Scheduler
      |
      v
Inference Backend
      |
      v
Measurement / Analytics
```

### Component 1: Prompt Cost Estimator

Purpose:

Estimate how costly a request is likely to be before generation.

Current direction:

- local ML model,
- lightweight enough to run before inference,
- predicts expected output length as a proxy for service cost,
- produces both a continuous estimate and uncertainty information.

The estimator is intentionally separated from the scheduler so either component can be changed independently.

### Component 2: Adaptive Scheduler

Purpose:

Choose request execution order and admission based on predicted request cost and current workload composition.

The scheduler is the main technical focus of the project.

It should be possible to compare multiple policies, including at minimum:

- FIFO,
- shortest-estimated-job-first,
- adaptive policy.

The adaptive policy should consider the current mix of requests rather than assume one static policy is always optimal.

Fairness and starvation prevention must be considered.

### Component 3: Inference Backend

Purpose:

Execute real LLM inference locally.

Critical design constraint:

**Our scheduler must retain control over admission and effective execution order.**

A backend with opaque internal scheduling can invalidate experiments because the backend may reorder, batch, or preempt requests after our scheduler makes its decision.

For the safest MVP:

- use one in-flight request at a time,
- ensure queue order directly determines execution order,
- then optionally add a configurable concurrency limit `K`.

The progression should be:

```text
K = 1
-> deterministic scheduler control

K > 1
-> scheduler controls admission and concurrency

optional later work
-> deeper batching / preemption integration
```

### Component 4: Measurement / Analytics

Purpose:

Measure whether the scheduler actually improves the system.

The project should rely on measured execution behavior, not simulated claims.

Primary system-level metrics:

- p50 latency,
- p99 latency,
- throughput,
- queue wait time,
- job completion time,
- fairness / starvation behavior,
- scheduler overhead.

Tiger Data is planned as the telemetry and analytics backend.

The dashboard should make workload composition, queue state, scheduler decisions, and performance changes visible.

---

## 4. Core Scheduling Hypothesis

The project is built around the hypothesis that:

> Different prompt workloads benefit from different scheduling behavior, and a scheduler that understands likely request cost and current queue composition can outperform naive FIFO scheduling.

A motivating example:

```text
Workload A:
15 short requests
5 long requests

Workload B:
2 short requests
18 long requests
```

A single static policy may behave differently across these workload mixes.

The adaptive scheduler should use predicted request cost and queue composition to decide which requests to admit or prioritize.

---

## 5. Demo Strategy

The demo should visibly show the scheduling problem and its effect.

A good demonstration pattern:

1. submit a mixed batch of short and long prompts,
2. show FIFO causing short requests to wait behind long ones,
3. switch to a cost-aware scheduling policy,
4. show queue order changing,
5. compare latency and throughput,
6. display the measured difference live.

The demo should make head-of-line blocking easy to understand.

The project should avoid relying only on aggregate charts; the queue itself should be visible.

---

## 6. Primary Inference Setup

Current primary plan:

- local Llama-family model,
- NVIDIA RTX 3060 Ti,
- prerecorded primary demo if necessary,
- constrained execution so our scheduler controls request order.

Exact model is still intentionally undecided.

Important:

The classifier and scheduler should be written behind generic interfaces so the backend can be replaced if required.

Suggested conceptual interfaces:

```python
predictor.predict(prompt)

scheduler.enqueue(request)

backend.generate(prompt)

metrics.record(event)
```

Potential emergency fallback:

- TinyLlama on Mac CPU,
- same scheduler architecture,
- separately calibrated predictor.

The fallback should not affect the primary design unless necessary.

---

## 7. ML Cost Estimator — Current Direction

The selected primary ML architecture is:

**DistilBERT-style auxiliary predictor**

The model should predict expected output length before generation.

Output length is being used as a proxy for request cost because autoregressive decoding cost is strongly related to the number of generated tokens.

The predictor should not be treated as the main project deliverable.

It exists to provide the scheduler with useful information.

### Predictor Output

The predictor should expose something conceptually like:

```python
{
    "expected_output_tokens": ...,
    "bin_probabilities": [...],
    "uncertainty": ...
}
```

The full probability distribution should be preserved rather than returning only one scalar.

This leaves room for the scheduler to use confidence later.

---

## 8. ML Predictor — Locked Decisions

The following decisions are currently considered fixed unless experiments justify changing them.

### Target

Use the approximate **p90 output-token length across repeated generations**.

Reason:

- underprediction is more harmful than mild overprediction,
- p90 is conservative,
- p90 is less sensitive to one extreme outlier than a strict maximum,
- percentile choice is easy to tune experimentally.

### Number of Generations per Prompt

Use:

```text
4 generations per prompt
```

This should be configurable.

### Decoding Settings During Data Collection

Keep decoding fixed during training-data generation.

Current defaults:

```text
temperature = 0.7
top_p = 0.9
max_new_tokens = fixed
system_prompt = fixed
```

The exact `max_new_tokens` and system prompt are still undecided.

### Binning

Use:

```text
20 quantile bins
```

Reason:

- balanced class populations,
- adapts to whatever response-length distribution the chosen dataset/model produces,
- easy to modify,
- avoids many nearly empty bins in heavy-tailed data.

### Predictor Output

Preserve:

- probability distribution over bins,
- expected continuous output length.

### Evaluation Metrics

Primary:

- MAE.

Secondary:

- p90 absolute error,
- underprediction rate,
- severe-underprediction rate,
- predictor inference latency,
- predictor memory overhead.

A severe underprediction can initially be defined as:

```text
predicted_length < 0.5 * actual_length
```

The threshold should remain configurable.

### Latency Budget

Predictor overhead should remain below:

```text
5% of median inference service time
```

This is preferred over a fixed millisecond threshold because it remains meaningful across hardware.

### Baselines

Always include a naive baseline:

```text
input token count -> linear regression -> output length
```

A TF-IDF-based model may also be implemented as a stronger lightweight baseline.

The purpose is to demonstrate that the learned predictor adds value beyond simple prompt length.

---

## 9. ML Research Basis

The predictor design is informed by two relevant papers.

### Response Length Perception and Sequence Scheduling — Zheng et al. (2023)

Relevant ideas:

- response length can be estimated before full generation,
- repeated generations can be used to construct conservative targets,
- underprediction is more damaging than overprediction,
- length prediction is useful only if it improves end-to-end scheduling,
- binning can simplify prediction,
- prediction errors must be accounted for by the scheduler.

Ideas from this work should inform:

- training-label construction,
- conservative targets,
- failure handling,
- scheduling evaluation.

### EGTP — Predicting LLM Output Length via Entropy-Guided Representations

Relevant ideas:

- output-length prediction is a regression problem with heavy-tailed targets,
- a distribution over length bins can be more useful than a single hard class,
- soft labels can preserve distance between neighboring bins,
- expected output length can be reconstructed from bin probabilities,
- MAE is a useful primary prediction metric,
- prediction uncertainty can be preserved for downstream scheduling.

The project is **not currently implementing EGTP hidden-state pooling**.

That remains a possible stretch direction.

Current primary architecture is DistilBERT because it is easier to train, isolate, debug, and integrate.

---

## 10. Data Requirements

The dataset has not yet been selected.

That is intentional.

The selected dataset should represent heterogeneous enterprise-assistant traffic rather than only one specialized task.

Desired prompt categories include:

- short factual queries,
- knowledge / document questions,
- summarization,
- troubleshooting,
- coding,
- reasoning,
- structured generation.

The dataset should contain enough variation in expected response length for scheduling to matter.

The project should avoid choosing a dataset solely because it is convenient.

The dataset should instead support the project story:

> a private/local enterprise assistant serving mixed workloads on constrained compute.

---

## 11. Training Data Generation Pipeline

Once the exact model and dataset are chosen, the intended flow is:

```text
Dataset prompts
      |
      v
Run each prompt 4 times
      |
      v
Record output token counts
      |
      v
Compute target percentile
      |
      v
Assign quantile bins
      |
      v
Train DistilBERT predictor
      |
      v
Evaluate on held-out prompts
```

Suggested initial split:

```text
70% train
15% validation
15% test
```

Suggested random seed:

```text
42
```

The pipeline should be reproducible and configurable.

---

## 12. Scheduler Baselines

At minimum, the scheduler should support:

### FIFO

Requests execute in arrival order.

This is the primary naive baseline.

### Shortest Estimated Job First

Requests are ordered by predicted output length.

This is the primary cost-aware baseline.

### Adaptive Scheduler

The custom policy.

The exact policy is still undecided.

It should use:

- predicted request cost,
- queue composition,
- fairness / starvation considerations.

Potential future inputs:

- predictor uncertainty,
- request age,
- queue depth,
- current concurrency,
- backend service characteristics.

---

## 13. Scheduler Fairness

The scheduler must not indefinitely delay expensive requests.

Potential mechanisms include:

- aging,
- maximum wait thresholds,
- priority boosts over time,
- bounded reordering.

The exact fairness mechanism is still undecided.

Fairness should be measured, not only discussed.

Possible metrics:

- maximum queue wait,
- p95/p99 queue wait,
- starvation count,
- latency by request-size class.

---

## 14. Concurrency Strategy

The safest initial backend configuration is:

```text
K = 1
```

Only one request runs at a time.

This ensures:

```text
our queue order == execution order
```

Once the system works:

```text
K = configurable
```

The scheduler can admit up to `K` requests.

This allows experimentation with:

- serial scheduling,
- bounded concurrency,
- workload-sensitive admission.

True backend-native continuous batching should not be introduced until scheduler control remains measurable and attributable.

---

## 15. Tiger Data Integration

Tiger Data should be used as the measurement and analytics backbone.

It should store request and scheduler telemetry.

Possible event types:

- request arrival,
- cost prediction,
- enqueue,
- dequeue,
- scheduling decision,
- inference start,
- inference completion,
- timeout/failure,
- policy change.

Potential fields:

```text
request_id
timestamp
prompt_category
prompt_token_count
predicted_output_tokens
prediction_distribution
prediction_uncertainty
actual_output_tokens
queue_wait_ms
service_time_ms
end_to_end_latency_ms
scheduler_policy
queue_depth
workload_mix
backend_config
```

Tiger Data should support:

- live metrics,
- historical comparisons,
- policy comparisons,
- latency percentiles,
- throughput,
- request-size analysis.

It should be meaningfully used, not added as decorative infrastructure.

---

## 16. Dashboard Requirements

The dashboard should make scheduling behavior visually obvious.

Useful views:

- current queue,
- predicted request sizes,
- request age,
- active request(s),
- current policy,
- p50 latency,
- p99 latency,
- throughput,
- recent completed requests,
- FIFO vs adaptive comparison.

The exact frontend technology is still undecided.

The dashboard should prioritize clarity over visual complexity.

---

## 17. Important Experimental Principle

Do not judge the ML predictor only by prediction accuracy.

The real question is:

> Does the predictor improve scheduling outcomes?

A model with slightly worse MAE may still produce better scheduler behavior if its errors are less harmful.

Therefore evaluate both:

### Predictor-level

- MAE,
- tail error,
- underprediction,
- predictor latency.

### System-level

- p50 latency,
- p99 latency,
- throughput,
- queue wait,
- fairness.

---

## 18. Important Failure Modes

Future agents should actively watch for the following.

### Backend Scheduler Interference

If the inference runtime performs its own opaque scheduling, batching, or preemption, our scheduler may have little effect.

This is a critical architectural risk.

### Predictor Overhead

If the classifier consumes too much time or memory, any scheduling improvement may disappear.

### Dataset Mismatch

A predictor trained on one workload may perform poorly on a very different prompt distribution.

### Model-Specific Labels

Output length depends on:

- model,
- decoding settings,
- system prompt,
- stop conditions.

A predictor trained for one model should not automatically be assumed to generalize to another.

### Overfitting to Synthetic Workloads

The demo should contain enough workload diversity to make the scheduling problem realistic.

### Scheduler Starvation

Shortest-job-first-style policies can indefinitely delay long jobs without fairness controls.

### Overcomplicating the MVP

Do not begin with:

- distributed multi-GPU scheduling,
- true continuous batching internals,
- custom CUDA,
- KV-cache orchestration,
- complex reinforcement learning.

The first goal is a clean, measurable scheduler.

---

## 19. Project Development Order

Recommended order:

### Phase 1 — Backend Control

- select exact local LLM,
- verify inference runs reliably,
- force deterministic scheduler ownership of execution order,
- benchmark baseline service times.

### Phase 2 — Measurement Infrastructure

- implement request IDs and timestamps,
- record queue wait and service time,
- connect Tiger Data,
- create minimal metric queries.

### Phase 3 — Scheduler Baselines

- FIFO,
- shortest-estimated-job-first using temporary synthetic costs,
- queue visualization.

### Phase 4 — ML Predictor

- select dataset,
- generate labeled data,
- implement naive baseline,
- train DistilBERT,
- evaluate prediction quality and overhead.

### Phase 5 — Predictor Integration

- replace synthetic costs with learned predictions,
- validate that scheduler behavior changes,
- measure end-to-end improvements.

### Phase 6 — Adaptive Policy

- use workload composition,
- add fairness,
- tune policy parameters.

### Phase 7 — Demo

- construct reproducible mixed workloads,
- display queue behavior,
- compare policies,
- record primary GPU demo.

### Stretch Goals

Only after the full pipeline works:

- K > 1 concurrency,
- prediction uncertainty-aware scheduling,
- EGTP-style estimator,
- dynamic batching,
- multiple inference workers,
- distributed coordination.

---

## 20. Current Open Decisions

The following are intentionally unresolved.

### Inference

- exact Llama-family model,
- exact runtime,
- quantization,
- exact `max_new_tokens`,
- exact fixed system prompt.

### Dataset

- exact training prompt source,
- final prompt categories and proportions,
- initial training-set size.

### Scheduler

- exact adaptive scheduling formula,
- starvation prevention mechanism,
- whether uncertainty is used immediately,
- whether K > 1 is included in the hackathon MVP.

### Dashboard

- frontend framework,
- charting library,
- exact visual layout.

### Tiger Data

- final schema,
- aggregation queries,
- retention strategy.

---

## 21. Non-Goals

The MVP is not intended to:

- train an LLM,
- build a general chatbot product,
- build a full RAG stack,
- solve semantic routing between different LLMs,
- reproduce a production inference engine,
- outperform vLLM's full production scheduler,
- invent a new ML architecture,
- become a distributed multi-GPU serving platform.

The goal is to build and measure a clear scheduling system around a real local LLM workload.

---

## 22. Core Project Story

The concise project story is:

> Local LLM deployments often serve a mixture of lightweight and expensive prompts on limited compute. Naive FIFO execution creates head-of-line blocking and poor tail latency. This project predicts likely response length before generation and uses that information to adaptively schedule requests. A real local LLM backend executes the workload, while Tiger Data records scheduling and inference telemetry. The system compares adaptive scheduling against fixed baselines using measured latency, throughput, and fairness.

---

## 23. Guiding Principle for Future Agents

Future work should preserve these priorities, in order:

1. **measurement validity,**
2. **scheduler control,**
3. **end-to-end working system,**
4. **clear demo,**
5. **classifier sophistication,**
6. **additional optimization.**

If a technically impressive feature makes it harder to attribute results or harder to produce a stable demo, it should probably be postponed.

The strongest version of this project is not the most complicated version.

It is the version where:

- the scheduler clearly controls execution,
- the predictor provides useful information,
- the measurements are credible,
- the performance difference is visible,
- and every major component has a clear reason to exist.
