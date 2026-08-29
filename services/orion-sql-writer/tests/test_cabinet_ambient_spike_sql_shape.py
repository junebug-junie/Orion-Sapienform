"""Column shape and routing for cabinet.ambient.spike.v1 -> cabinet_ambient_spike."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import inspect as sa_inspect

from app.models.cabinet_ambient_spike import CabinetAmbientSpikeSQL
from app.settings import Settings

from orion.schemas.telemetry.cabinet_ambient_spike import CabinetAmbientSpikeV1


def _columns() -> dict:
    return {c.key: c for c in sa_inspect(CabinetAmbientSpikeSQL).columns}


def test_every_payload_field_has_a_column() -> None:
    cols = _columns()
    for field in CabinetAmbientSpikeV1.model_fields:
        assert field in cols, f"payload field {field!r} has no column"


def test_the_channel_is_actually_subscribed() -> None:
    example = Path(__file__).resolve().parents[1] / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert "orion:cabinet:ambient:spike" in json.loads(raw)

    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert "orion:cabinet:ambient:spike" in stale.effective_subscribe_channels


def test_the_kind_routes_to_this_table() -> None:
    from app.worker import MODEL_MAP

    model_cls, schema_cls = MODEL_MAP["CabinetAmbientSpikeSQL"]
    assert model_cls.__tablename__ == "cabinet_ambient_spike"
    assert schema_cls.__name__ == "CabinetAmbientSpikeV1"
    assert Settings().route_map.get("cabinet.ambient.spike.v1") == "CabinetAmbientSpikeSQL"


def test_a_real_payload_round_trips_through_the_column_filter() -> None:
    now = datetime(2026, 8, 28, 6, 20, tzinfo=timezone.utc)
    payload = CabinetAmbientSpikeV1(
        spike_id="spike-1",
        node="athena",
        timestamp=now,
        activity=0.41,
        rms=7340.0,
        peak=16213.0,
        activity_threshold=0.30,
        consecutive_ticks=2,
        source_service="orion-biometrics",
        source_node="athena",
    ).model_dump(mode="json")

    cols = set(_columns())
    kept = {k: v for k, v in payload.items() if k in cols}
    assert kept["spike_id"] == "spike-1"
    assert kept["activity"] == 0.41
    assert set(payload) - cols == set(), "payload fields would be silently dropped"
