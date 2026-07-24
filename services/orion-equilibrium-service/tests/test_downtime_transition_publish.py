from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import EquilibriumService, settings
from orion.schemas.telemetry.system_health import EquilibriumServiceState


def _service() -> EquilibriumService:
    svc = EquilibriumService()
    svc.bus = MagicMock()
    svc.bus.publish = AsyncMock()
    return svc


def _state(service: str, status: str) -> EquilibriumServiceState:
    now = datetime.now(timezone.utc)
    return EquilibriumServiceState(
        service=service,
        node="athena",
        status=status,
        last_seen_ts=now,
        heartbeat_interval_sec=10.0,
        down_for_ms=0,
        uptime_pct={"60": 1.0},
    )


@pytest.mark.asyncio
async def test_publish_service_transitions_publishes_real_transition_envelope(monkeypatch):
    monkeypatch.setattr(settings, "equilibrium_transition_publish_enable", True)
    monkeypatch.setattr(settings, "channel_equilibrium_transition", "orion:equilibrium:transition")
    svc = _service()
    now = datetime.now(timezone.utc)

    # First tick seeds tracking (no fire); second tick with a status change
    # must publish exactly one transition envelope.
    await svc._publish_service_transitions([_state("recall", "ok")], now)
    svc.bus.publish.assert_not_awaited()

    await svc._publish_service_transitions([_state("recall", "down")], now)

    svc.bus.publish.assert_awaited_once()
    channel, env = svc.bus.publish.call_args.args
    assert channel == "orion:equilibrium:transition"
    assert env.kind == "equilibrium.service.transition.v1"
    assert env.payload["service"] == "recall"
    assert env.payload["from_status"] == "ok"
    assert env.payload["to_status"] == "down"


@pytest.mark.asyncio
async def test_publish_service_transitions_repeated_same_status_does_not_publish(monkeypatch):
    monkeypatch.setattr(settings, "equilibrium_transition_publish_enable", True)
    svc = _service()
    now = datetime.now(timezone.utc)

    await svc._publish_service_transitions([_state("recall", "ok")], now)
    await svc._publish_service_transitions([_state("recall", "ok")], now)

    svc.bus.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_publish_service_transitions_respects_kill_switch(monkeypatch):
    monkeypatch.setattr(settings, "equilibrium_transition_publish_enable", False)
    svc = _service()
    now = datetime.now(timezone.utc)

    await svc._publish_service_transitions([_state("recall", "ok")], now)
    await svc._publish_service_transitions([_state("recall", "down")], now)

    svc.bus.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_publish_snapshot_calls_publish_service_transitions(monkeypatch):
    """`_publish_snapshot` is the single cadence-owning caller for transition
    detection (see DowntimeTransitionTracker's own docstring for why the
    other, higher-frequency `_calculate_metrics()` call sites must not also
    drive it).
    """
    svc = _service()
    svc._calculate_metrics = MagicMock(return_value=(0.0, 1.0, []))
    svc._publish_service_transitions = AsyncMock()

    await svc._publish_snapshot()

    svc._publish_service_transitions.assert_awaited_once()
