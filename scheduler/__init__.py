"""Adaptive LLM request scheduler: K=1, non-preemptive, policy-pluggable.

The engine composes four injected parts, none of which knows about the others:
    Predictor (scheduler.predictors) -> SchedulerPolicy (scheduler.policies)
    -> InferenceBackend (scheduler.backends) -> EventSink (scheduler.events)
"""
