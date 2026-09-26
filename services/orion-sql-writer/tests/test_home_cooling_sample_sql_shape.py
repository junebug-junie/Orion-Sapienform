"""Column shape and routing for home.cooling.sample.v1 -> home_cooling_sample."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import inspect as sa_inspect

from app.models.home_cooling_sample import HomeCoolingSampleSQL
from app.settings import Settings
from app.worker import MODEL_MAP, _normalize_home_cooling_sample_payload

from orion.schemas.telemetry.home_cooling import HomeCoolingSampleV1


def _columns() -> dict:
    return {c.key: c for c in sa_inspect(HomeCoolingSampleSQL).columns}


def test_flat_payload_fields_have_columns() -> None:
    cols = _columns()
    for field in (
        "ts",
        "node",
        "role",
        "cooling_watts",
        "cooling_volts",
        "cooling_amps",
        "switch_on",
        "zwave_node_id",
        "controller_ready",
        "device_online",
        "payload_json",
    ):
        assert field in cols, f"expected column {field!r} missing"


def test_the_channel_is_actually_subscribed() -> None:
    example = Path(__file__).resolve().parents[1] / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert "orion:home:cooling:sample" in json.loads(raw)

    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert "orion:home:cooling:sample" in stale.effective_subscribe_channels


def test_the_kind_routes_to_this_table() -> None:
    model_cls, schema_cls = MODEL_MAP["HomeCoolingSampleSQL"]
    assert model_cls.__tablename__ == "home_cooling_sample"
    assert schema_cls.__name__ == "HomeCoolingSampleV1"
    assert Settings().route_map.get("home.cooling.sample.v1") == "HomeCoolingSampleSQL"


def test_a_real_payload_round_trips_through_the_column_filter() -> None:
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    payload = HomeCoolingSampleV1(
        ts=now,
        node="athena",
        role="cabinet_cooling",
        controller={"ready": True, "driver": "zwave-js", "device_path": "/dev/zwave"},
        device={
            "id": "node-2",
            "name": "Cabinet AC",
            "product": "Shelly Wave Plug",
            "online": True,
        },
        measurements={"cooling_watts": 412.0, "cooling_volts": 120.1},
        state={"switch_on": True},
        provenance={"zwave_node_id": 2, "source": "zwave-js"},
    ).model_dump(mode="json")

    mapped = _normalize_home_cooling_sample_payload(payload)
    cols = set(_columns())
    kept = {k: v for k, v in mapped.items() if k in cols}
    assert kept["ts"] == now.isoformat().replace("+00:00", "Z")
    assert kept["node"] == "athena"
    assert kept["role"] == "cabinet_cooling"
    assert kept["cooling_watts"] == 412.0
    assert kept["cooling_volts"] == 120.1
    assert kept["cooling_amps"] is None
    assert kept["switch_on"] is True
    assert kept["zwave_node_id"] == 2
    assert kept["controller_ready"] is True
    assert kept["device_online"] is True
    assert kept["payload_json"] == payload
    assert set(mapped) - cols <= {"schema_name", "schema", "controller", "device", "measurements", "state", "provenance"}


def test_payload_json_sanitizes_datetime_from_plain_model_dump() -> None:
    """Regression: live inserts failed with TypeError datetime not JSON serializable."""
    now = datetime(2026, 9, 26, 4, 50, tzinfo=timezone.utc)
    # mode="python" (default) leaves datetime objects — matches sql-writer ingest path.
    payload = HomeCoolingSampleV1(
        ts=now,
        node="athena",
        role="cabinet_cooling",
        controller={"ready": True, "driver": "zwave-js"},
        device={"id": "node-2", "name": "Cabinet AC", "online": True},
        measurements={"cooling_watts": 33.11},
        state={"switch_on": True},
        provenance={"zwave_node_id": 2, "source": "zwave-js"},
    ).model_dump()

    assert isinstance(payload["ts"], datetime)
    mapped = _normalize_home_cooling_sample_payload(payload)
    blob = mapped["payload_json"]
    assert isinstance(blob, dict)
    assert isinstance(blob["ts"], str)
    json.dumps(blob)  # must not raise
