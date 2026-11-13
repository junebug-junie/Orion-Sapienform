"""Column shape and routing for closet.env.sample.v1 -> closet_env_sample."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import inspect as sa_inspect

from app.models.closet_env_sample import ClosetEnvSampleSQL
from app.settings import Settings
from app.worker import MODEL_MAP, _normalize_closet_env_sample_payload

from orion.schemas.telemetry.closet_env import (
    ClosetEnvMeasurementsV1,
    ClosetEnvSampleV1,
    ClosetEnvSensorV1,
    ClosetEnvStateV1,
)

NOW = datetime(2026, 11, 4, 23, 0, tzinfo=timezone.utc)


def _write_keys(payload: dict) -> dict:
    cols = {c.key for c in sa_inspect(ClosetEnvSampleSQL).columns}
    return {k: v for k, v in _normalize_closet_env_sample_payload(payload).items() if k in cols}


def test_the_channel_is_actually_subscribed() -> None:
    example = Path(__file__).resolve().parents[1] / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert "orion:closet:env:sample" in json.loads(raw)
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert "orion:closet:env:sample" in stale.effective_subscribe_channels


def test_the_kind_routes_to_this_table() -> None:
    model_cls, schema_cls = MODEL_MAP["ClosetEnvSampleSQL"]
    assert model_cls.__tablename__ == "closet_env_sample"
    assert schema_cls is ClosetEnvSampleV1
    assert Settings().route_map.get("closet.env.sample.v1") == "ClosetEnvSampleSQL"


def test_fresh_payload_round_trips_to_columns() -> None:
    sample = ClosetEnvSampleV1(
        ts=NOW,
        sensor=ClosetEnvSensorV1(id="scd40-closet", online=True),
        measurements=ClosetEnvMeasurementsV1(closet_co2_ppm=640, closet_temp_c=29.4, closet_humidity_pct=23.5),
    )
    row = _write_keys(sample.model_dump(mode="json", by_alias=True))
    assert row["closet_co2_ppm"] == 640
    assert row["closet_temp_c"] == 29.4
    assert row["closet_humidity_pct"] == 23.5
    assert row["stale"] is False
    assert row["sensor_id"] == "scd40-closet"
    assert row["role"] == "closet_environment"
    assert row["payload_json"]["schema"] == "closet.env.sample.v1"


def test_stale_payload_stores_null_readings_not_zero() -> None:
    sample = ClosetEnvSampleV1(
        ts=NOW,
        sensor=ClosetEnvSensorV1(id="scd40-closet", online=False),
        measurements=ClosetEnvMeasurementsV1(),
        state=ClosetEnvStateV1(stale=True),
    )
    row = _write_keys(sample.model_dump(mode="json", by_alias=True))
    assert not {"closet_co2_ppm", "closet_temp_c", "closet_humidity_pct"} & row.keys()
    assert row["stale"] is True
