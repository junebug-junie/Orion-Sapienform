"""Column shape and routing for office.env.sample.v1 -> office_env_sample."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect as sa_inspect

from app import main
from app.models.office_env_sample import OfficeEnvSampleSQL
from app.settings import Settings
from app.worker import MODEL_MAP, _normalize_office_env_sample_payload

from orion.schemas.telemetry.office_env import (
    OfficeEnvMeasurementsV1,
    OfficeEnvSampleV1,
    OfficeEnvSensorV1,
    OfficeEnvStateV1,
)

NOW = datetime(2026, 10, 27, 20, 0, tzinfo=timezone.utc)


def _columns() -> dict:
    return {c.key: c for c in sa_inspect(OfficeEnvSampleSQL).columns}


def _write_keys(payload: dict) -> dict:
    cols = _columns()
    return {k: v for k, v in _normalize_office_env_sample_payload(payload).items() if k in cols}


def test_the_channel_is_actually_subscribed() -> None:
    example = Path(__file__).resolve().parents[1] / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert "orion:office:env:sample" in json.loads(raw)
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert "orion:office:env:sample" in stale.effective_subscribe_channels


def test_the_kind_routes_to_this_table() -> None:
    model_cls, schema_cls = MODEL_MAP["OfficeEnvSampleSQL"]
    assert model_cls.__tablename__ == "office_env_sample"
    assert schema_cls.__name__ == "OfficeEnvSampleV1"
    assert Settings().route_map.get("office.env.sample.v1") == "OfficeEnvSampleSQL"


def test_fresh_payload_round_trips_to_columns() -> None:
    sample = OfficeEnvSampleV1(
        ts=NOW,
        sensor=OfficeEnvSensorV1(id="sht40-office", online=True),
        measurements=OfficeEnvMeasurementsV1(office_temp_c=21.5, office_humidity_pct=30.0),
    )
    row = _write_keys(sample.model_dump(mode="json", by_alias=True))
    assert row["office_temp_c"] == 21.5
    assert row["office_humidity_pct"] == 30.0
    assert row["stale"] is False
    assert row["sensor_id"] == "sht40-office"
    assert row["sensor_online"] is True
    assert row["payload_json"]["schema"] == "office.env.sample.v1"


def test_stale_payload_stores_null_readings_not_zero() -> None:
    sample = OfficeEnvSampleV1(
        ts=NOW,
        sensor=OfficeEnvSensorV1(id="sht40-office", online=False),
        measurements=OfficeEnvMeasurementsV1(),
        state=OfficeEnvStateV1(stale=True),
    )
    row = _write_keys(sample.model_dump(mode="json", by_alias=True))
    assert "office_temp_c" not in row and "office_humidity_pct" not in row
    assert row["stale"] is True
    assert row["sensor_online"] is False


def test_air_quality_round_trips_to_columns() -> None:
    sample = OfficeEnvSampleV1(
        ts=NOW,
        sensor=OfficeEnvSensorV1(id="sht40-office", online=True),
        measurements=OfficeEnvMeasurementsV1(office_temp_c=21.5, office_eco2_ppm=640, office_tvoc_ppb=36),
    )
    row = _write_keys(sample.model_dump(mode="json", by_alias=True))
    assert row["office_eco2_ppm"] == 640
    assert row["office_tvoc_ppb"] == 36
    assert row["air_warming"] is False


def test_warming_air_stores_null_readings_and_the_flag() -> None:
    sample = OfficeEnvSampleV1(
        ts=NOW,
        sensor=OfficeEnvSensorV1(id="sht40-office", online=True),
        measurements=OfficeEnvMeasurementsV1(office_temp_c=21.5),
        state=OfficeEnvStateV1(air_warming=True),
    )
    row = _write_keys(sample.model_dump(mode="json", by_alias=True))
    assert "office_eco2_ppm" not in row and "office_tvoc_ppb" not in row
    assert row["air_warming"] is True


def test_boot_ddl_adds_every_air_column() -> None:
    """The live table predates these columns; create_all will not add them."""
    main_src = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text()
    for column in ("office_eco2_ppm", "office_tvoc_ppb", "air_warming"):
        assert column in _columns()
        assert f"office_env_sample ADD COLUMN IF NOT EXISTS {column}" in main_src


class _Rows:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple]:
        return self._rows


class _FakeConnection:
    def __init__(self, *, fail_alter: bool = False, column_shapes: dict | None = None) -> None:
        self.column_shapes = dict(column_shapes or {})
        self.fail_alter = fail_alter
        self.alter_attempts = 0

    def exec_driver_sql(self, sql: str) -> _Rows:
        from orion.db.guarded_ddl import COLUMNS_SQL, INDEXES_SQL, TABLES_SQL

        if sql == TABLES_SQL:
            return _Rows([("office_env_sample",)])
        if sql == COLUMNS_SQL:
            return _Rows([("office_env_sample", column) for column in sorted(self.column_shapes)])
        if sql == INDEXES_SQL:
            return _Rows([])
        if re.match(r"\s*ALTER TABLE", sql, re.IGNORECASE):
            self.alter_attempts += 1
            if self.fail_alter:
                raise TimeoutError("canceling statement due to lock timeout")
            column = re.search(r"ADD COLUMN IF NOT EXISTS (\w+)", sql, re.IGNORECASE)
            assert column is not None
            sql_type = re.search(r"ADD COLUMN IF NOT EXISTS \w+ (\w+)", sql, re.IGNORECASE)
            assert sql_type is not None
            data_type = {"INTEGER": "integer", "BOOLEAN": "boolean"}[sql_type.group(1).upper()]
            self.column_shapes[column.group(1).lower()] = (data_type, "YES")
            return _Rows([])
        if "FROM information_schema.columns" in sql:
            return _Rows(
                [(column, *self.column_shapes[column]) for column in sorted(self.column_shapes)]
            )
        raise AssertionError(f"unexpected SQL: {sql}")


class _Begin:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection

    def __enter__(self) -> _FakeConnection:
        return self.connection

    def __exit__(self, *_args) -> None:
        return None


class _FakeEngine:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection

    def begin(self) -> _Begin:
        return _Begin(self.connection)


def test_required_office_columns_are_established_in_an_isolated_gate(monkeypatch) -> None:
    connection = _FakeConnection()
    monkeypatch.setattr(main, "engine", _FakeEngine(connection))

    main._ensure_office_env_columns()

    assert connection.column_shapes == main._OFFICE_ENV_REQUIRED_COLUMN_SHAPES


def test_office_column_lock_timeout_stops_startup(monkeypatch) -> None:
    connection = _FakeConnection(fail_alter=True)
    monkeypatch.setattr(main, "engine", _FakeEngine(connection))

    with pytest.raises(TimeoutError, match="lock timeout"):
        main._ensure_office_env_columns()


def test_current_office_column_shape_takes_no_table_lock(monkeypatch) -> None:
    connection = _FakeConnection(
        fail_alter=True,
        column_shapes=main._OFFICE_ENV_REQUIRED_COLUMN_SHAPES,
    )
    monkeypatch.setattr(main, "engine", _FakeEngine(connection))

    main._ensure_office_env_columns()

    assert connection.alter_attempts == 0


def test_wrong_office_column_shape_stops_startup(monkeypatch) -> None:
    wrong_shapes = dict(main._OFFICE_ENV_REQUIRED_COLUMN_SHAPES)
    wrong_shapes["office_eco2_ppm"] = ("text", "YES")
    connection = _FakeConnection(column_shapes=wrong_shapes)
    monkeypatch.setattr(main, "engine", _FakeEngine(connection))

    with pytest.raises(RuntimeError, match="column shape mismatch"):
        main._ensure_office_env_columns()
