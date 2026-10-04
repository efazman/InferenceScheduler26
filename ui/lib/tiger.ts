// Server-only: read-only run history from Tiger Data (scheduler_events, see scheduler/tigerdata_schema.sql).
// The dashboard never depends on it: every view works from local events.jsonl files, and any failure
// here becomes a plain status ("not_configured" / "unavailable"), never a raw error or a credential.
import { promises as fs } from "fs";
import path from "path";
import { Client } from "pg";

export interface TigerRun {
  run_id: string;
  policy: string | null;
  measurement: string | null;
  manifest_id: string | null;
  started: string | null;
  completed: number;
  mean_latency_ms: number | null;
  p95_latency_ms: number | null;
  max_wait_ms: number | null;
}

export type TigerHistory =
  | { status: "connected"; table_exists: boolean; runs: TigerRun[] }
  | { status: "not_configured" | "unavailable"; runs: [] };

const ENV_FILE = path.resolve(/*turbopackIgnore: true*/ process.cwd(), "..", "tiger-cloud-db-inference-credentials.env");

async function resolveDsn(): Promise<string | null> {
  const fromEnv = process.env.TIGER_DATA_DSN || process.env.TIMESCALE_SERVICE_URL;
  if (fromEnv) return fromEnv;
  try {
    const text = await fs.readFile(process.env.TIGER_ENV_FILE ?? ENV_FILE, "utf-8");
    for (const key of ["TIGER_DATA_DSN", "TIMESCALE_SERVICE_URL"]) {
      const m = text.match(new RegExp(`^${key}=(.+)$`, "m"));
      if (m) return m[1].trim().replace(/^["']|["']$/g, "");
    }
  } catch {
    // no credentials file: not configured
  }
  return null;
}

const HISTORY_SQL = `
  select run_id, max(policy) as policy, max(measurement) as measurement, max(manifest_id) as manifest_id,
         min("timestamp")::text as started,
         count(*) filter (where event_type = 'inference_completed')::int as completed,
         avg(end_to_end_latency_ms) filter (where event_type = 'inference_completed') as mean_latency_ms,
         percentile_cont(0.95) within group (order by end_to_end_latency_ms)
           filter (where event_type = 'inference_completed') as p95_latency_ms,
         max(queue_wait_ms) filter (where event_type = 'inference_completed') as max_wait_ms
  from scheduler_events group by run_id order by min("timestamp") desc limit 50`;

export async function tigerHistory(): Promise<TigerHistory> {
  const dsn = await resolveDsn();
  if (!dsn) return { status: "not_configured", runs: [] };
  // Tiger Cloud requires TLS; pg's sslmode=require does not verify the certificate chain either.
  const client = new Client({ connectionString: dsn.replace(/[?&]sslmode=[^&]*/, ""), ssl: { rejectUnauthorized: false },
                              connectionTimeoutMillis: 5000, query_timeout: 8000 });
  try {
    await client.connect();
    const exists = (await client.query("select to_regclass('public.scheduler_events') is not null as e")).rows[0].e;
    if (!exists) return { status: "connected", table_exists: false, runs: [] };
    const rows = (await client.query(HISTORY_SQL)).rows as TigerRun[];
    return { status: "connected", table_exists: true, runs: rows };
  } catch {
    return { status: "unavailable", runs: [] }; // deliberately no error text: it can contain connection details
  } finally {
    client.end().catch(() => undefined);
  }
}
