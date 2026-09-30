"""Compile-time shape checks for the transport_baseline_hourly write path (no Postgres
required): subscribed even under a stale operator channel list, routed, every schema
field lands on a column, append-only, and a real producer row validates and maps."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import inspect

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.registry import resolve  # noqa: E402
from orion.schemas.telemetry.transport_baseline_hourly import (  # noqa: E402
    TRANSPORT_BASELINE_HOURLY_CHANNEL,
    TRANSPORT_BASELINE_HOURLY_KIND,
    TransportBaselineHourlyV1,
)

from app.models.transport_baseline_hourly import TransportBaselineHourlySQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402
from app.worker import INSERT_ONLY_MODELS, MODEL_MAP  # noqa: E402


def _env_example(key: str):
    raw = next(line.split("=", 1)[1].strip() for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
               if line.startswith(f"{key}="))
    return json.loads(raw)


def test_the_channel_is_actually_subscribed() -> None:
    assert TRANSPORT_BASELINE_HOURLY_CHANNEL in _env_example("SQL_WRITER_SUBSCRIBE_CHANNELS")
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert TRANSPORT_BASELINE_HOURLY_CHANNEL in stale.effective_subscribe_channels


def test_route_map_and_model_map_agree() -> None:
    assert DEFAULT_ROUTE_MAP[TRANSPORT_BASELINE_HOURLY_KIND] == "TransportBaselineHourlySQL"
    assert _env_example("SQL_WRITER_ROUTE_MAP_JSON")[TRANSPORT_BASELINE_HOURLY_KIND] == "TransportBaselineHourlySQL"
    assert MODEL_MAP["TransportBaselineHourlySQL"] == (TransportBaselineHourlySQL, TransportBaselineHourlyV1)
    assert TransportBaselineHourlySQL in INSERT_ONLY_MODELS
    assert resolve("TransportBaselineHourlyV1") is TransportBaselineHourlyV1


def test_every_schema_field_lands_on_a_column() -> None:
    columns = {c.key for c in inspect(TransportBaselineHourlySQL).columns}
    for name in TransportBaselineHourlyV1.model_fields:
        assert name in columns, name


def test_a_producer_row_constructs_the_sql_model() -> None:
    row = TransportBaselineHourlyV1(
        summary_id="abc", service="cortex-exec", instance="chat",
        key="orion:exec:request:LLMGatewayService",
        hour_start=datetime(2026, 9, 29, 8, tzinfo=timezone.utc), flush_reason="hour_end",
        flushed_at=datetime(2026, 9, 29, 9, 1, 30, tzinfo=timezone.utc),
        windows_seen=120, windows_evaluated=40, z_p50=0.1, z_p90=1.2, saturation_ratio_p50=1.02,
        baseline_ms=9000.0, floor_ms_start=8800.0, floor_ms=8900.0, calls_per_min_mean=3.2,
        conditions_opened={"timeout": 1}, open_at_hour_end=["timeout"],
        would_emit_by_condition={"timeout:open": 1}, warm=True, config_fingerprint="f",
    )
    obj = TransportBaselineHourlySQL(**row.model_dump())
    assert obj.key.endswith("LLMGatewayService") and obj.would_emit_by_condition == {"timeout:open": 1}


def test_a_real_envelope_lands_through_the_worker_consume_path(monkeypatch) -> None:
    """Envelope -> handle_envelope -> route -> schema -> _write_row column filter ->
    insert-only branch, against sqlite. Catches a route/schema/column mismatch the
    direct-constructor test above cannot."""
    import asyncio
    from uuid import uuid4

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import app.worker as worker
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    TransportBaselineHourlySQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    holder: dict = {}

    def _get_session():
        holder.setdefault("s", Session())
        return holder["s"]

    def _remove_session():
        s = holder.pop("s", None)
        if s is not None:
            s.close()

    monkeypatch.setattr(worker, "get_session", _get_session)
    monkeypatch.setattr(worker, "remove_session", _remove_session)

    row = TransportBaselineHourlyV1(
        summary_id="row-1", service="orion-durable-runs", instance="main",
        key="orion:gpu_pool:lease:request#gpu_pool_lease",
        hour_start=datetime(2026, 9, 29, 9, tzinfo=timezone.utc), flush_reason="hour_end",
        flushed_at=datetime(2026, 9, 29, 10, 1, 30, tzinfo=timezone.utc),
        windows_seen=120, windows_evaluated=12, z_p50=-0.2, z_p90=0.9, saturation_ratio_p50=1.01,
        floor_ms_start=40.0, floor_ms=41.0, calls_per_min_mean=0.4,
        conditions_opened={}, open_at_hour_end=[], would_emit_by_condition={"timeout:open": 1},
        warm=True, warm_at_start=True, config_fingerprint="fp",
    )
    env = BaseEnvelope(
        kind=TRANSPORT_BASELINE_HOURLY_KIND, source=ServiceRef(name="orion-equilibrium-service"),
        correlation_id=uuid4(), payload=row.model_dump(mode="json"),
    )
    asyncio.run(worker.handle_envelope(env))

    s = Session()
    try:
        got = s.get(TransportBaselineHourlySQL, "row-1")
        assert got is not None, "the envelope did not reach transport_baseline_hourly"
        assert got.key == "orion:gpu_pool:lease:request#gpu_pool_lease"
        assert got.would_emit_by_condition == {"timeout:open": 1}
        assert got.warm_at_start is True and got.windows_evaluated == 12
    finally:
        s.close()
