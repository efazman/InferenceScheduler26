-- Tiger Data (PostgreSQL + TimescaleDB) schema for scheduler events.
-- NOT APPLIED ANYWHERE YET: no credentials exist. Local JSONL (scheduler_runs/<run>/events.jsonl)
-- stays the source of truth; this table is a copy for live queries and dashboards.
-- Rows are produced by scheduler.events.tiger_row(event); column names follow that mapping.

CREATE TABLE IF NOT EXISTS scheduler_events (
    "timestamp"              timestamptz      NOT NULL,  -- event wall_time (UTC)
    engine_time_ms           double precision NOT NULL,  -- timestamp_ms: engine clock, ms since run start
    seq                      integer          NOT NULL,  -- order within a run (tie-break)
    run_id                   text             NOT NULL,
    manifest_id              text,                       -- same value for runs that replayed one trace
    event_type               text             NOT NULL,
    request_id               text,
    policy                   text             NOT NULL,  -- fifo | sejf | adaptive
    queue_depth              integer,
    predicted_output_tokens  double precision,
    uncertainty              double precision,
    actual_output_tokens     integer,
    queue_wait_ms            double precision,
    service_time_ms          double precision,
    end_to_end_latency_ms    double precision,
    workload_name            text,
    backend                  text,                       -- llamacpp | mock
    predictor                text,
    simulated                boolean          NOT NULL,
    success                  boolean,
    failure_reason           text,
    data                     jsonb                       -- event-specific extras (queue order, max wait, timings)
);

SELECT create_hypertable('scheduler_events', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS scheduler_events_run ON scheduler_events (run_id, seq);
CREATE INDEX IF NOT EXISTS scheduler_events_manifest ON scheduler_events (manifest_id, policy);

-- Example: per-policy latency for one manifest (completed requests only).
-- SELECT policy, count(*), avg(end_to_end_latency_ms),
--        percentile_cont(0.5) WITHIN GROUP (ORDER BY end_to_end_latency_ms) AS p50,
--        percentile_cont(0.95) WITHIN GROUP (ORDER BY end_to_end_latency_ms) AS p95,
--        max(queue_wait_ms) AS max_wait
-- FROM scheduler_events
-- WHERE event_type = 'inference_completed' AND manifest_id = '<id>' AND NOT simulated
-- GROUP BY policy;
