from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from orion.schemas.causal_geometry import (
    CausalGeometryDivergenceEntryV1,
    CausalGeometryEdgeV1,
    CausalGeometrySnapshotV1,
)
from orion.substrate import causal_geometry_snapshot_store as store_module

BASE_TS = datetime(2026, 7, 16, tzinfo=timezone.utc)


def _snapshot() -> CausalGeometrySnapshotV1:
    return CausalGeometrySnapshotV1(
        snapshot_id="snap-1",
        generated_at=BASE_TS,
        window_start=BASE_TS,
        window_end=BASE_TS,
        edges=[
            CausalGeometryEdgeV1(
                source_id="cap:transport",
                target_id="cap:orchestration",
                lag_sec=0,
                strength=0.9,
                significance=0.01,
                n_samples=100,
                window_start=BASE_TS,
                window_end=BASE_TS,
            )
        ],
        designed_topology_version="field_lattice.v1.test",
        divergence=[
            CausalGeometryDivergenceEntryV1(
                source_id="cap:transport",
                target_id="cap:orchestration#orchestration",
                observed_strength=0.9,
                designed_weight=0.2,
                delta=0.7,
                status="both",
            )
        ],
        insufficient_data=False,
        notes=["1 significant edge found."],
    )


class _FakeCursor:
    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self._record.setdefault("executed_sql", []).append(sql)
        self._record.setdefault("executed_params", []).append(params)

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _FakeConn:
    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._record)

    def commit(self) -> None:
        self._record["committed"] = True

    def close(self) -> None:
        self._record["closed"] = True


class _FakePsycopg2Module:
    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record

    def connect(self, postgres_uri: str) -> _FakeConn:
        self._record["connect_uri"] = postgres_uri
        return _FakeConn(self._record)


class _BoomingPsycopg2Module:
    def connect(self, postgres_uri: str) -> Any:
        raise RuntimeError("postgres unreachable")


def test_persist_snapshot_executes_insert_with_conflict_clause(monkeypatch) -> None:
    record: dict[str, Any] = {}
    monkeypatch.setitem(
        __import__("sys").modules, "psycopg2", _FakePsycopg2Module(record)
    )

    result = store_module.persist_snapshot("postgresql://unused/unused", _snapshot())

    assert result == {"ok": True, "error": None}
    assert record["connect_uri"] == "postgresql://unused/unused"
    assert record["committed"] is True
    assert record["closed"] is True
    executed_sql = record["executed_sql"]
    assert any("causal_geometry_snapshots" in sql and "ON CONFLICT" in sql for sql in executed_sql)


def test_persist_snapshot_never_raises_on_connect_failure(monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "psycopg2", _BoomingPsycopg2Module())

    result = store_module.persist_snapshot("postgresql://unused/unused", _snapshot())

    assert result["ok"] is False
    assert "postgres unreachable" in result["error"]


def test_ensure_schema_ddl_contains_expected_table_and_type() -> None:
    assert "CREATE TABLE IF NOT EXISTS causal_geometry_snapshots" in store_module._CREATE_TABLE_SQL
    assert "JSONB" in store_module._CREATE_TABLE_SQL
