"""Column shape and routing for office.env.sample.v1 -> office_env_sample."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import inspect as sa_inspect

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
