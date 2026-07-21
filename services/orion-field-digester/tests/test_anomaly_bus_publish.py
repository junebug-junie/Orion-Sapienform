from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

import app.anomaly_bus_publish as anomaly_bus_publish
from app.anomaly_bus_publish import FIELD_CHANNEL_ANOMALY_SCORE_KIND, publish_anomaly_score
from orion.schemas.telemetry.field_channel_anomaly_score import FieldChannelAnomalyScoreV1


def _score() -> FieldChannelAnomalyScoreV1:
    return FieldChannelAnomalyScoreV1(
        correlation_id="00000000-0000-4000-8000-000000000001",
        encoder_id="mood-arc-encoder:test.v1",
        encoder_version="test.v1",
        recon_loss=0.02,
        recon_error_p95=0.0014,
        threshold_multiplier=3.0,
        threshold=0.0042,
        anomalous=True,
        window_start=datetime.now(timezone.utc),
        window_end=datetime.now(timezone.utc),
        window_size=30,
    )


@pytest.mark.asyncio
async def test_publishes_envelope_with_expected_shape(monkeypatch) -> None:
    mock_bus = AsyncMock()
    monkeypatch.setattr(anomaly_bus_publish, "OrionBusAsync", lambda *a, **kw: mock_bus)

    await publish_anomaly_score(
        bus_url="redis://unused/0",
        bus_enabled=True,
        channel="orion:field_channel:anomaly_score",
        score=_score(),
        service_name="orion-field-digester",
        service_version="0.1.0",
        node_name="athena",
    )

    mock_bus.connect.assert_awaited_once()
    mock_bus.publish.assert_awaited_once()
    channel_arg, env = mock_bus.publish.call_args[0]
    assert channel_arg == "orion:field_channel:anomaly_score"
    assert env.kind == FIELD_CHANNEL_ANOMALY_SCORE_KIND
    assert env.payload["recon_loss"] == 0.02
    assert env.payload["anomalous"] is True
    mock_bus.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_noop_when_bus_disabled(monkeypatch) -> None:
    mock_bus = AsyncMock()
    monkeypatch.setattr(anomaly_bus_publish, "OrionBusAsync", lambda *a, **kw: mock_bus)

    await publish_anomaly_score(
        bus_url="redis://unused/0",
        bus_enabled=False,
        channel="orion:field_channel:anomaly_score",
        score=_score(),
        service_name="orion-field-digester",
        service_version="0.1.0",
        node_name="athena",
    )

    mock_bus.connect.assert_not_awaited()
    mock_bus.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_never_raises_when_publish_fails(monkeypatch) -> None:
    mock_bus = AsyncMock()
    mock_bus.publish.side_effect = RuntimeError("boom")
    monkeypatch.setattr(anomaly_bus_publish, "OrionBusAsync", lambda *a, **kw: mock_bus)

    # Must not raise.
    await publish_anomaly_score(
        bus_url="redis://unused/0",
        bus_enabled=True,
        channel="orion:field_channel:anomaly_score",
        score=_score(),
        service_name="orion-field-digester",
        service_version="0.1.0",
        node_name="athena",
    )
    mock_bus.close.assert_awaited_once()
