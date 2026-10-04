// Event schema shared with scheduler/events.py. Simulated bundles carry a compact subset of these
// fields; recorded runs (events.jsonl) carry all of them. The UI only relies on the subset.

export interface SchedEvent {
  event_type: string;
  timestamp_ms: number;
  seq: number;
  request_id?: string | null;
  queue_depth?: number | null;
  predicted_output_tokens?: number | null;
  actual_output_tokens?: number | null;
  queue_wait_ms?: number | null;
  service_time_ms?: number | null;
  end_to_end_latency_ms?: number | null;
  success?: boolean | null;
  error?: string | null;
  scheduler_policy?: string;
  data?: Record<string, any>;
}

export interface RunSummary {
  n_completed: number;
  n_failed: number;
  mean_latency_ms: number | null;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
  p99_latency_ms: number | null;
  p99_reliable: boolean;
  mean_queue_wait_ms: number | null;
  max_queue_wait_ms: number | null;
  throughput_rps: number | null;
  starvation_count: number;
  starvation_threshold_ms: number;
  short_mean_latency_ms: number | null;
  long_mean_latency_ms: number | null;
  long_max_queue_wait_ms: number | null;
  [key: string]: number | boolean | null;
}

export interface PolicyRun {
  policy: string;
  events: SchedEvent[]; // sorted by (timestamp_ms, seq)
  summary: RunSummary | null;
}

export interface SimIndex {
  workloads: { name: string; description: string; n: number }[];
  max_wait_ms: number;
  warning: string;
}

export interface SimBundle {
  warning: string;
  max_wait_ms: number;
  workload: { name: string; description: string; seed: number; params: Record<string, number> };
  requests: { request_id: string; prompt: string; size_class: string }[];
  runs: Record<string, { summary: RunSummary; events: SchedEvent[] }>;
}

export interface RecordedRunInfo {
  name: string;
  policy: string | null;
  backend: string | null;
  predictor: string | null;
  workload: string | null;
  complete: boolean;
  warning: string | null;
  manifest_id: string | null; // runs with the same manifest replayed the same arrival trace
  measurement: "real" | "mock" | null;
  max_wait: string | null; // e.g. "18.1 s = 3 x median service 6.04 s (runs.jsonl, n=2000)"
  max_wait_info: Record<string, any> | null; // structured: mode, multiplier, median_service_ms, ...
  generation: Record<string, any> | null; // real runs: model_name, quantization, max_new_tokens, ...
  updated_ms: number; // events.jsonl mtime: an incomplete run that stopped updating is not "live"
  summary: RunSummary | null;
}

export type ReqState = "arriving" | "queued" | "running" | "completed" | "failed";

export interface ReqView {
  id: string;
  sizeClass: string | null; // known in simulation; null for real prompts
  arrival: number;
  enqueue?: number;
  start?: number;
  end?: number;
  predicted?: number | null;
  actual?: number | null;
  state: ReqState;
  selectedBy?: string;
  error?: string | null;
}

export interface ReplayState {
  now: number;
  running: ReqView | null;
  queue: ReqView[]; // in the policy's service order
  completed: ReqView[]; // in completion order
  failed: ReqView[];
  arrived: number;
}

export const POLICIES = ["fifo", "sejf", "adaptive"] as const;
export const POLICY_LABEL: Record<string, string> = { fifo: "FIFO", sejf: "SEJF", adaptive: "Adaptive" };
