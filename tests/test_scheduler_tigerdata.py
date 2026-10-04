"""Tiger Data sink. Unit tests use an injected fake connection (no network). The integration test
runs only with TIGER_INTEGRATION=1 and uses a throwaway table that it drops afterwards."""

import json
import os
import time
import uuid

import pytest

from scheduler import tigerdata as td
from scheduler.clock import VirtualClock
from scheduler.engine import SchedulerEngine
from scheduler.events import FanoutSink, InMemoryEventSink
from scheduler.policies import make_policy
from scheduler.workloads import make_workload

SECRET = "s3cr3t-Pa55"


def events_of_one_run(policy="adaptive", meta=None):
    wl = make_workload("head_of_line", n=60)
    mem = InMemoryEventSink()
    SchedulerEngine(wl.mock_predictor(), wl.mock_backend(), make_policy(policy), mem, VirtualClock(),
                    run_id=f"run-{policy}", run_metadata=meta or {}).run(wl.to_requests())
    return mem.events


class FakeConn:
    def __init__(self, fail_times=0, delay_s=0.0):
        self.rows, self.fail_times, self.delay_s, self.closed = [], fail_times, delay_s, False

    def close(self):
        self.closed = True


def fake_writer(conn, rows, table):
    time.sleep(conn.delay_s)
    if conn.fail_times > 0:
        conn.fail_times -= 1
        raise ConnectionError(f"server closed the connection (postgres://user:{SECRET}@host/db)")
    conn.rows.extend(rows)
    return len(rows)


# --------------------------------------------------------------------------- credentials

def test_dsn_resolution_order_and_redaction(tmp_path, monkeypatch):
    for var in td.ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / "creds.env"
    env.write_text(f"# comment\nTIMESCALE_SERVICE_URL=postgres://tsdbadmin:{SECRET}@h.example:5432/tsdb?sslmode=require\n"
                   f"PGPASSWORD={SECRET}\n")
    dsn, source = td.resolve_dsn(env)
    assert dsn.startswith("postgres://tsdbadmin:") and SECRET not in source
    monkeypatch.setenv("TIGER_DATA_DSN", "postgres://a:b@c/d")
    assert td.resolve_dsn(env) == ("postgres://a:b@c/d", "$TIGER_DATA_DSN")  # env var wins
    monkeypatch.delenv("TIGER_DATA_DSN")
    with pytest.raises(td.TigerConfigError, match="no Tiger Data credentials"):
        td.resolve_dsn(tmp_path / "missing.env")
    msg = td.redact(f"failed: postgres://tsdbadmin:{SECRET}@h:1/db and password={SECRET} host=h")
    assert SECRET not in msg and "postgres://tsdbadmin:****@h:1/db" in msg and "password=****" in msg


def test_table_name_is_validated():
    with pytest.raises(ValueError):
        td.schema_sql('scheduler_events"; drop table x; --')
    assert "CREATE TABLE IF NOT EXISTS scheduler_events_test" in td.schema_sql("scheduler_events_test")


def test_schema_columns_match_rows():
    sql = td.schema_sql()
    for col in td.TIGER_COLUMNS:
        assert (f'"{col}"' if col == "timestamp" else f"    {col} ") in sql, col
    assert "CREATE UNIQUE INDEX IF NOT EXISTS scheduler_events_run_seq" in sql


def test_rows_carry_run_level_fields():
    info = td._RunInfo()
    rows = [info.row(e.to_dict()) for e in events_of_one_run(meta={"manifest_id": "m1", "measurement": "real"})]
    assert all(tuple(r) == td.TIGER_COLUMNS for r in rows)
    assert {r["manifest_id"] for r in rows} == {"m1"} and {r["measurement"] for r in rows} == {"real"}
    done = [r for r in rows if r["event_type"] == "inference_completed"]
    assert done and all(r["policy"] == "adaptive" and r["end_to_end_latency_ms"] is not None for r in done)
    sim = td._RunInfo()
    assert {sim.row(e.to_dict())["measurement"] for e in events_of_one_run()} == {"simulated"}


# --------------------------------------------------------------------------- live sink

def test_sink_writes_every_event_once_in_batches():
    conn = FakeConn()
    sink = td.TigerDataEventSink(connect_fn=lambda: conn, write_fn=fake_writer, batch_size=7, flush_interval_s=0.05)
    events = events_of_one_run(meta={"manifest_id": "m1", "measurement": "mock"})
    for e in events:
        sink.emit(e)
    sink.close()
    assert len(conn.rows) == len(events) and sink.rows_written == len(events) and sink.rows_dropped == 0
    assert [r["seq"] for r in conn.rows] == [e.seq for e in events]
    assert conn.closed


def test_emit_never_blocks_on_a_slow_database():
    conn = FakeConn(delay_s=0.3)
    sink = td.TigerDataEventSink(connect_fn=lambda: conn, write_fn=fake_writer, batch_size=1000, close_timeout_s=5)
    events = events_of_one_run()
    t0 = time.perf_counter()
    for e in events:
        sink.emit(e)
    assert time.perf_counter() - t0 < 0.1  # the engine thread never waits for the network
    sink.close()
    assert sink.rows_written == len(events)


def test_transient_failures_are_retried_and_redacted():
    conn = FakeConn(fail_times=2)
    sink = td.TigerDataEventSink(connect_fn=lambda: conn, write_fn=fake_writer, flush_interval_s=0.05)
    events = events_of_one_run()
    for e in events:
        sink.emit(e)
    sink.close()
    assert sink.errors == 2 and sink.rows_written == len(events) and sink.rows_dropped == 0
    assert SECRET not in json.dumps(sink.stats()) and "****" in sink.last_error


def test_unreachable_database_drops_rows_but_the_run_completes():
    def refuse():
        raise ConnectionError(f"could not connect: password={SECRET}")

    mem = InMemoryEventSink()
    tiger = td.TigerDataEventSink(connect_fn=refuse, close_timeout_s=0.5, flush_interval_s=0.05)
    fan = FanoutSink(mem, tiger)
    wl = make_workload("head_of_line", n=60)
    done = SchedulerEngine(wl.mock_predictor(), wl.mock_backend(), make_policy("sejf"), fan, VirtualClock()).run(
        wl.to_requests())
    fan.close()
    assert len(done) == 12 and len(mem.events) == tiger.rows_queued  # local log complete
    assert tiger.rows_written == 0 and tiger.rows_dropped == tiger.rows_queued and tiger.errors >= 1
    assert SECRET not in json.dumps(tiger.stats())


def test_run_with_tiger_but_no_credentials_fails_before_starting(tmp_path, monkeypatch):
    from scheduler.__main__ import main as cli

    for var in td.ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(SystemExit, match="--tiger: no Tiger Data credentials"):
        cli(["run", "--backend", "mock", "--workload", "head_of_line", "--n", "60", "--policy", "fifo",
             "--speedup", "500", "--runs-dir", str(tmp_path), "--run-name", "x", "--tiger",
             "--tiger-env-file", str(tmp_path / "missing.env")])
    assert not (tmp_path / "x" / "summary.json").exists()


# --------------------------------------------------------------------------- real database (opt-in)

@pytest.mark.skipif(os.environ.get("TIGER_INTEGRATION") != "1", reason="set TIGER_INTEGRATION=1 to hit Tiger Data")
def test_tiger_integration_roundtrip_on_a_throwaway_table():
    table = f"scheduler_events_it_{uuid.uuid4().hex[:8]}"
    conn = td.connect(td.resolve_dsn()[0])
    try:
        td.apply_schema(conn, table)
        td.apply_schema(conn, table)  # idempotent
        events = events_of_one_run(meta={"manifest_id": "it", "measurement": "mock"})
        sink = td.TigerDataEventSink(table=table, batch_size=50, flush_interval_s=0.1)
        for e in events:
            sink.emit(e)
        sink.close()
        assert sink.rows_written == len(events) and sink.errors == 0
        assert td.table_status(conn, table)["rows"] == len(events)
        again = td.import_events(conn, [e.to_dict() for e in events], table)
        assert again == 0 and td.table_status(conn, table)["rows"] == len(events)  # duplicates ignored
        row = conn.execute(f'select measurement, manifest_id, data->>\'reason\' from "{table}" '
                           f"where event_type = 'request_selected' limit 1").fetchone()
        assert row[0] == "mock" and row[1] == "it" and row[2]
    finally:
        conn.execute(f'drop table if exists "{table}"')
        conn.close()
