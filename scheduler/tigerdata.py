"""Tiger Data (Tiger Cloud: PostgreSQL + TimescaleDB) event storage.

Local JSONL stays the source of truth. Tiger Data is an optional copy for live queries:

    python -m scheduler tiger-check                     # read-only: can we connect? does the table exist?
    python -m scheduler tiger-init                      # create the scheduler_events hypertable (idempotent)
    python -m scheduler run ... --tiger                 # also stream a run's events to Tiger Data
    python -m scheduler tiger-import scheduler_runs/<run>   # upload a recorded run (idempotent)

Credentials are never stored in the repo. They're resolved, in order, from:
  1. the TIGER_DATA_DSN or TIMESCALE_SERVICE_URL environment variable,
  2. an explicit --tiger-env-file, or
  3. the Tiger Cloud download ``tiger-cloud-db-inference-credentials.env`` in the repo root.
The repo's .gitignore excludes *.env, .env* and tiger-cloud-*credentials*, and
tests/test_repo_hygiene.py fails if a credential file or a password-bearing URL is ever tracked.
Error messages pass through ``redact`` so a password is never printed.

Requires ``pip install -r requirements-tiger.txt`` (psycopg). Nothing else in the scheduler imports
psycopg, so simulation and local runs work without it.
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from pathlib import Path

from scheduler import config

TABLE = "scheduler_events"
SCHEMA_PATH = Path(__file__).with_name("tigerdata_schema.sql")
ENV_VARS = ("TIGER_DATA_DSN", "TIMESCALE_SERVICE_URL")
DEFAULT_ENV_FILE = config.PROJECT_ROOT / "tiger-cloud-db-inference-credentials.env"

TIGER_COLUMNS = ("timestamp", "engine_time_ms", "seq", "run_id", "manifest_id", "event_type", "request_id",
                 "policy", "queue_depth", "predicted_output_tokens", "uncertainty", "actual_output_tokens",
                 "queue_wait_ms", "service_time_ms", "end_to_end_latency_ms", "workload_name", "backend",
                 "predictor", "simulated", "measurement", "success", "failure_reason", "data")

_URL_SECRET = re.compile(r"(postgres(?:ql)?://[^:/@\s]+:)[^@\s]+@")
_KV_SECRET = re.compile(r"(password\s*=\s*)\S+", re.IGNORECASE)


class TigerConfigError(RuntimeError):
    pass


def redact(text: str) -> str:
    """Remove passwords from connection strings / libpq messages."""
    return _KV_SECRET.sub(r"\1****", _URL_SECRET.sub(r"\1****@", str(text)))


def _parse_env_file(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def resolve_dsn(env_file: str | Path | None = None) -> tuple[str, str]:
    """-> (dsn, where it came from). The returned description never contains the secret."""
    for var in ENV_VARS:
        if os.environ.get(var):
            return os.environ[var], f"${var}"
    candidates = [Path(env_file)] if env_file else [DEFAULT_ENV_FILE]
    for path in candidates:
        if path.exists():
            values = _parse_env_file(path)
            for var in ENV_VARS:
                if values.get(var):
                    return values[var], f"{path.name} ({var})"
            raise TigerConfigError(f"{path} has neither {' nor '.join(ENV_VARS)}")
    raise TigerConfigError(f"no Tiger Data credentials: set {ENV_VARS[0]} or place the Tiger Cloud .env file at "
                           f"{DEFAULT_ENV_FILE} (gitignored), or pass --tiger-env-file")


def connect(dsn: str, timeout_s: float = 10.0):
    try:
        import psycopg
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise TigerConfigError("psycopg is not installed: pip install -r requirements-tiger.txt") from e
    try:
        return psycopg.connect(dsn, connect_timeout=int(timeout_s), autocommit=True)
    except Exception as e:  # noqa: BLE001 - re-raised without the password
        raise TigerConfigError(f"could not connect to Tiger Data: {redact(e)}") from None


def schema_sql(table: str = TABLE) -> str:
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
        raise ValueError(f"invalid table name {table!r}")
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    return sql if table == TABLE else sql.replace(TABLE, table)


def apply_schema(conn, table: str = TABLE) -> None:
    conn.execute(schema_sql(table))


def table_status(conn, table: str = TABLE) -> dict:
    cur = conn.execute("select to_regclass(%s) is not null", (f"public.{table}",))
    exists = cur.fetchone()[0]
    info = {"table": table, "exists": exists}
    if exists:
        info["rows"] = conn.execute(f'select count(*) from "{table}"').fetchone()[0]
        info["rows_by_measurement"] = dict(conn.execute(
            f'select measurement, count(*) from "{table}" group by measurement').fetchall())
    return info


# ----------------------------------------------------------------------------- rows

def measurement_of(event: dict, run_info: dict | None = None) -> str:
    if run_info and run_info.get("measurement"):
        return run_info["measurement"]
    if event.get("simulated"):
        return "simulated"
    return "mock" if event.get("backend") == "mock" else "real"


def tiger_row(event: dict, manifest_id: str | None = None, measurement: str | None = None) -> dict:
    """Map one event (Event.to_dict() or a JSONL line) to a scheduler_events row
    (scheduler/tigerdata_schema.sql). Run-level fields come from the run's run_started event."""
    return {
        "timestamp": event.get("wall_time"), "engine_time_ms": event["timestamp_ms"], "seq": event["seq"],
        "run_id": event.get("run_id"), "manifest_id": manifest_id, "event_type": event["event_type"],
        "request_id": event.get("request_id"), "policy": event.get("scheduler_policy"),
        "queue_depth": event.get("queue_depth"), "predicted_output_tokens": event.get("predicted_output_tokens"),
        "uncertainty": event.get("uncertainty"), "actual_output_tokens": event.get("actual_output_tokens"),
        "queue_wait_ms": event.get("queue_wait_ms"), "service_time_ms": event.get("service_time_ms"),
        "end_to_end_latency_ms": event.get("end_to_end_latency_ms"), "workload_name": event.get("workload_name"),
        "backend": event.get("backend"), "predictor": event.get("predictor"), "simulated": bool(event.get("simulated")),
        "measurement": measurement or measurement_of(event), "success": event.get("success"),
        "failure_reason": event.get("error"), "data": event.get("data") or {},
    }


class _RunInfo:
    """Remembers run-level fields (manifest_id, measurement) from each run's run_started event."""

    def __init__(self):
        self._runs: dict[str, dict] = {}

    def row(self, event: dict) -> dict:
        run_id = event.get("run_id")
        if event["event_type"] == "run_started":
            self._runs[run_id] = {"manifest_id": (event.get("data") or {}).get("manifest_id"),
                                  "measurement": (event.get("data") or {}).get("measurement")}
        info = self._runs.get(run_id, {})
        return tiger_row(event, info.get("manifest_id"), measurement_of(event, info))


def _insert_sql(table: str) -> str:
    cols = ", ".join(f'"{c}"' for c in TIGER_COLUMNS)
    marks = ", ".join(["%s"] * len(TIGER_COLUMNS))
    return f'INSERT INTO "{table}" ({cols}) VALUES ({marks}) ON CONFLICT DO NOTHING'


def _params(row: dict) -> tuple:
    from psycopg.types.json import Jsonb

    return tuple(Jsonb(row[c]) if c == "data" else row[c] for c in TIGER_COLUMNS)


def write_rows(conn, rows: list[dict], table: str = TABLE) -> int:
    """Insert rows (duplicates ignored). Returns the number of rows actually inserted."""
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(_insert_sql(table), [_params(r) for r in rows])
        return max(cur.rowcount, 0)  # psycopg 3: total rows affected; conflicts count as 0


def import_events(conn, events: list[dict], table: str = TABLE, batch_size: int = 500) -> int:
    info, inserted = _RunInfo(), 0
    rows = [info.row(e) for e in sorted(events, key=lambda e: (e["timestamp_ms"], e["seq"]))]
    for i in range(0, len(rows), batch_size):
        inserted += write_rows(conn, rows[i:i + batch_size], table)
    return inserted


# ----------------------------------------------------------------------------- live sink

class TigerDataEventSink:
    """Streams events to Tiger Data without ever slowing the scheduler.

    emit() only enqueues a row and returns immediately. A background thread batches the inserts
    (every ``batch_size`` rows or ``flush_interval_s``), reconnects with backoff, and counts
    failures instead of raising. close() flushes for up to ``close_timeout_s`` and then gives up,
    counting the rest as dropped. Every run's events are also in its local events.jsonl, and
    ``tiger-import`` can upload anything that was dropped, idempotently.

    Use it as a secondary sink: FanoutSink(LocalJsonlEventSink(...), TigerDataEventSink()).
    """

    def __init__(self, dsn: str | None = None, table: str = TABLE, env_file: str | Path | None = None,
                 batch_size: int = 200, flush_interval_s: float = 1.0, close_timeout_s: float = 15.0,
                 max_queue: int = 100_000, connect_fn=None, write_fn=None):
        schema_sql(table)  # validates the table name
        self.table = table
        self.source = "custom connection" if connect_fn else None
        if connect_fn is None:
            dsn_value, self.source = (dsn, "argument") if dsn else resolve_dsn(env_file)
            connect_fn = lambda: connect(dsn_value)  # noqa: E731
        self._connect = connect_fn
        self._write = write_fn or write_rows
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.close_timeout_s = close_timeout_s
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._info = _RunInfo()
        self._stop = threading.Event()
        self._conn = None
        self.rows_queued = self.rows_written = self.rows_dropped = self.errors = 0
        self.last_error: str | None = None
        self._thread = threading.Thread(target=self._loop, name="tigerdata-sink", daemon=True)
        self._thread.start()

    def emit(self, event) -> None:
        row = self._info.row(event.to_dict() if hasattr(event, "to_dict") else event)
        try:
            self._queue.put_nowait(row)
            self.rows_queued += 1
        except queue.Full:
            self.rows_dropped += 1

    def _flush(self, batch: list[dict]) -> bool:
        try:
            if self._conn is None:
                self._conn = self._connect()
            self._write(self._conn, batch, self.table)
            self.rows_written += len(batch)
            return True
        except Exception as e:  # noqa: BLE001 - never propagate into the scheduler
            self.errors += 1
            self.last_error = redact(e)
            try:
                if self._conn is not None:
                    self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
            return False

    def _loop(self) -> None:
        batch: list[dict] = []
        backoff = 0.5
        deadline = None
        while True:
            stopping = self._stop.is_set()
            if stopping and deadline is None:
                deadline = time.monotonic() + self.close_timeout_s
            try:
                batch.append(self._queue.get(timeout=0.05 if stopping else self.flush_interval_s))
                while len(batch) < self.batch_size:
                    batch.append(self._queue.get_nowait())
            except queue.Empty:
                pass
            if batch:
                if self._flush(batch):
                    batch, backoff = [], 0.5
                elif not stopping:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 10.0)
            if stopping and (not batch and self._queue.empty()):
                break
            if deadline is not None and time.monotonic() > deadline:
                self.rows_dropped += len(batch) + self._queue.qsize()
                break
            if stopping and batch:
                time.sleep(min(backoff, max(0.0, deadline - time.monotonic())))
                backoff = min(backoff * 2, 10.0)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(self.close_timeout_s + 5)
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    def stats(self) -> dict:
        return {"table": self.table, "credentials_from": self.source, "rows_queued": self.rows_queued,
                "rows_written": self.rows_written, "rows_dropped": self.rows_dropped, "errors": self.errors,
                "last_error": self.last_error}
