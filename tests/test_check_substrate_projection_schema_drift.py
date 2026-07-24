from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import check_substrate_projection_schema_drift as gate  # noqa: E402


class FakeConn:
    """Fakes asyncpg's connection object across all seven known projection tables.

    `payloads` maps table name -> dict payload (row present, will be JSON-encoded
    like a real jsonb column) or None (no row). Any table not present in `payloads`
    is treated as "no row" by default. `missing_tables` simulates a table that does
    not exist yet (fresh/unmigrated deploy).
    """

    def __init__(self, payloads: dict[str, dict | None] | None = None, missing_tables: set[str] | None = None,
                 query_errors: dict[str, Exception] | None = None) -> None:
        self._payloads = payloads or {}
        self._missing_tables = missing_tables or set()
        self._query_errors = query_errors or {}
        self.closed = False

    def _table_for_query(self, query: str) -> str:
        for spec in gate.PROJECTIONS:
            if spec.table in query:
                return spec.table
        raise AssertionError(f"unrecognized query: {query}")

    async def fetchrow(self, query: str, *args):
        table = self._table_for_query(query)
        if table in self._query_errors:
            raise self._query_errors[table]
        if table in self._missing_tables:
            import asyncpg

            raise asyncpg.exceptions.UndefinedTableError(f'relation "{table}" does not exist')
        payload = self._payloads.get(table)
        if payload is None:
            return None
        return {"projection_json": json.dumps(payload)}

    async def close(self) -> None:
        self.closed = True


def _connect_returning(conn: FakeConn):
    return patch("asyncpg.connect", new=AsyncMock(return_value=conn))


# --- DB unavailability fallback (the branchable behavior CLAUDE.md flags as worth
# testing without a live DB) ---------------------------------------------------


def test_main_skips_with_exit_zero_when_no_postgres_uri(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    exit_code = gate.main([])
    assert exit_code == 0


def test_main_skips_with_exit_zero_on_blank_postgres_uri(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    exit_code = gate.main(["--postgres-uri", "   "])
    assert exit_code == 0


def test_main_json_shape_when_no_postgres_uri(monkeypatch, capsys):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    exit_code = gate.main(["--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "skipped"
    assert payload["reason"] == "no_postgres_uri"
    assert payload["checked"] == 0


def test_main_skips_with_exit_zero_when_connection_refused():
    with patch("asyncpg.connect", new=AsyncMock(side_effect=ConnectionRefusedError("refused"))):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 0


def test_main_json_shape_when_connection_fails(capsys):
    with patch("asyncpg.connect", new=AsyncMock(side_effect=OSError("no route to host"))):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "skipped"
    assert payload["reason"] == "connection_failed"


# --- real per-row checks (live-shaped: a real connection object, real pydantic
# models, no mocking of the validation itself) ----------------------------------


def test_main_exits_zero_when_every_row_is_absent():
    """Fresh deploy / self-healed after a stale-row reset: no rows anywhere is a
    healthy state, not a failure -- matches how the live 2026-07-24 incident was
    actually fixed (delete the one stale row, let it self-heal)."""
    conn = FakeConn(payloads={})
    with _connect_returning(conn):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 0
    assert conn.closed


def test_main_exits_zero_when_rows_validate():
    valid_transport = {
        "schema_version": "transport_bus.projection.v1",
        "updated_at": "2026-07-24T00:00:00Z",
        "projection_id": "active_transport_bus_projection",
        "buses": {
            "node-1": {
                "target_id": "t1",
                "node_id": "node-1",
                "sample_window_id": "w1",
                "source_trace_id": "s1",
                "stream_backlog_health": 0.5,
                "stream_backlog_pressure": 0.1,
            }
        },
    }
    conn = FakeConn(payloads={"substrate_transport_bus_projection": valid_transport})
    with _connect_returning(conn):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 0


def test_main_reproduces_the_live_incident_exit_one():
    """The exact 2026-07-24 shape: a persisted row still carrying the pre-rename
    field names (bus_health/transport_pressure) against the current, renamed
    TransportBusStateV1 model (extra='forbid'). This must FAIL, not silently pass --
    that gap is precisely what let the real incident crash-loop for ~10 hours."""
    stale_transport = {
        "schema_version": "transport_bus.projection.v1",
        "updated_at": "2026-07-24T00:00:00Z",
        "projection_id": "active_transport_bus_projection",
        "buses": {
            "node-1": {
                "target_id": "t1",
                "node_id": "node-1",
                "sample_window_id": "w1",
                "source_trace_id": "s1",
                "bus_health": 0.5,
                "transport_pressure": 0.1,
            }
        },
    }
    conn = FakeConn(payloads={"substrate_transport_bus_projection": stale_transport})
    with _connect_returning(conn):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 1


def test_main_reports_the_failing_table_in_output(capsys):
    stale_transport = {
        "schema_version": "transport_bus.projection.v1",
        "updated_at": "2026-07-24T00:00:00Z",
        "projection_id": "active_transport_bus_projection",
        "buses": {
            "node-1": {
                "target_id": "t1",
                "node_id": "node-1",
                "sample_window_id": "w1",
                "source_trace_id": "s1",
                "bus_health": 0.5,
                "transport_pressure": 0.1,
            }
        },
    }
    conn = FakeConn(payloads={"substrate_transport_bus_projection": stale_transport})
    with _connect_returning(conn):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "substrate_transport_bus_projection" in out
    assert "FAIL" in out


def test_main_json_shape_when_a_row_fails(capsys):
    stale_transport = {
        "schema_version": "transport_bus.projection.v1",
        "updated_at": "2026-07-24T00:00:00Z",
        "projection_id": "active_transport_bus_projection",
        "buses": {"node-1": {"target_id": "t1", "node_id": "node-1", "sample_window_id": "w1",
                              "source_trace_id": "s1", "bus_health": 0.5}},
    }
    conn = FakeConn(payloads={"substrate_transport_bus_projection": stale_transport})
    with _connect_returning(conn):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db", "--json"])
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ran"
    assert payload["failures"] == 1
    failing = [r for r in payload["results"] if r["status"] == "validation_failed"]
    assert failing and failing[0]["table"] == "substrate_transport_bus_projection"


def test_main_treats_missing_table_as_non_fatal():
    """An unmigrated/fresh deploy (table doesn't exist yet) is reported but must not
    fail the gate -- distinct from a row that exists but fails validation."""
    conn = FakeConn(payloads={}, missing_tables={"substrate_chat_session_projection"})
    with _connect_returning(conn):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 0


def test_main_exits_two_on_unrelated_query_error():
    conn = FakeConn(payloads={}, query_errors={
        "substrate_route_arbitration_projection": RuntimeError("syntax error at or near ...")
    })
    with _connect_returning(conn):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 2


def test_malformed_stored_payload_is_a_real_error_not_a_skip():
    """Regression: a corrupt/truncated stored JSON payload must surface as a real
    per-row error (exit 2), NOT get silently relabeled 'SKIPPED: could not connect
    to Postgres' -- caught in review, where an earlier version of this script
    routed any exception raised anywhere after a successful connect through the
    same broad except-block used for connection failures. A malformed row is
    arguably a worse version of the exact incident this gate exists to catch, so
    it must never produce the friendliest possible non-answer."""

    class _BadJSONConn(FakeConn):
        async def fetchrow(self, query: str, *args):
            table = self._table_for_query(query)
            if table == "substrate_transport_bus_projection":
                return {"projection_json": "{not valid json"}
            return await super().fetchrow(query, *args)

    conn = _BadJSONConn(payloads={})
    with _connect_returning(conn):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db", "--json"])
    assert exit_code == 2
    assert conn.closed


def test_malformed_stored_payload_reported_as_payload_error(capsys):
    class _BadJSONConn(FakeConn):
        async def fetchrow(self, query: str, *args):
            table = self._table_for_query(query)
            if table == "substrate_transport_bus_projection":
                return {"projection_json": "{not valid json"}
            return await super().fetchrow(query, *args)

    conn = _BadJSONConn(payloads={})
    with _connect_returning(conn):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db", "--json"])
    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ran"
    bad = [r for r in payload["results"] if r["status"] == "payload_error"]
    assert bad and bad[0]["table"] == "substrate_transport_bus_projection"


def test_unexpected_error_after_successful_connect_is_not_treated_as_skip(capsys):
    """A bug or unhandled exception raised after the connection already succeeded
    must never print 'SKIPPED: could not connect to Postgres' -- that would hide a
    real problem behind the friendliest possible non-answer. Simulated here via a
    conn whose close() raises: this happens in _run_all's `finally` block, after
    every per-row check already ran and returned cleanly, so it deliberately lands
    outside both `_check_projection`'s per-row try/except *and* the connect-only
    `_ConnectFailure` wrapping -- exactly the class of bug review found live."""

    class _CloseExplodingConn(FakeConn):
        async def close(self) -> None:
            raise SystemError("simulated unrelated internal bug during close")

    conn = _CloseExplodingConn(payloads={})
    with _connect_returning(conn):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert out["status"] == "error"
    assert out["reason"] == "unexpected_error"


def test_validation_failure_takes_priority_over_query_error():
    stale_transport = {
        "schema_version": "transport_bus.projection.v1",
        "updated_at": "2026-07-24T00:00:00Z",
        "projection_id": "active_transport_bus_projection",
        "buses": {"node-1": {"target_id": "t1", "node_id": "node-1", "sample_window_id": "w1",
                              "source_trace_id": "s1", "bus_health": 0.5}},
    }
    conn = FakeConn(
        payloads={"substrate_transport_bus_projection": stale_transport},
        query_errors={"substrate_route_arbitration_projection": RuntimeError("boom")},
    )
    with _connect_returning(conn):
        exit_code = gate.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 1


# --- spec-list integrity: catches the PROJECTIONS list itself going stale (e.g. a
# model renamed/removed without updating this gate) -----------------------------


def test_every_projection_spec_model_imports_today():
    for spec in gate.PROJECTIONS:
        model = gate._load_model(spec)
        assert model.model_config.get("extra") == "forbid", (
            f"{spec.label}: model {spec.model_name} no longer has extra='forbid' -- "
            "this gate assumes strict validation is what makes a stale field name a "
            "hard failure. If this is intentional, this gate needs a rethink, not a "
            "silent pass."
        )


def test_projection_specs_have_unique_tables():
    tables = [spec.table for spec in gate.PROJECTIONS]
    assert len(tables) == len(set(tables))
