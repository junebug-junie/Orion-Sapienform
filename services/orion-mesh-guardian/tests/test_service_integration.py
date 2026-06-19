from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.probe import ProbeResult
from app.roster import ProbeConfig, ProbeMode, RosterDocument, RosterEntry
from app.service import MeshGuardianService
from app.settings import Settings
from app.state_machine import ServicePhase, ServiceState


def _entry(*, entry_id: str = "landing-pad") -> RosterEntry:
    return RosterEntry(
        id=entry_id,
        heartbeat_name=entry_id,
        compose_dir="orion-landing-pad",
        compose_service="orion-landing-pad",
        include_bus_env=False,
        auto_remediate=True,
        probe=ProbeConfig(
            mode=ProbeMode.redis_and_http,
            intake_channels=["orion:pad:rpc:request"],
            ready_url="http://orion-landing-pad:8370/ready",
        ),
    )


@pytest.mark.asyncio
async def test_probe_loop_publishes_attention_on_confirmed_unhealthy() -> None:
    settings = Settings(
        MESH_GUARDIAN_PROBE_INTERVAL_SEC=1,
        MESH_GUARDIAN_AUTO_REMEDIATE=False,
        MESH_GUARDIAN_CONSECUTIVE_PROBE_FAILS=2,
    )
    service = MeshGuardianService(settings)
    service.roster = RosterDocument(services=[_entry()])
    service.bus = MagicMock()
    service.bus.redis = AsyncMock()

    attention_events: list[dict] = []

    def _capture_attention(**kwargs: object) -> None:
        event = kwargs.get("event")
        if isinstance(event, dict):
            attention_events.append(event)

    service.attention.publish_transition = _capture_attention

    sleep_ticks = 0

    async def _stop_after_three_cycles(_delay: float) -> None:
        nonlocal sleep_ticks
        sleep_ticks += 1
        if sleep_ticks >= 3:
            service._stop.set()

    bad_probe = ProbeResult(status="probe_bad", reason="no_subscriber")
    with patch("app.service.run_probe", new=AsyncMock(return_value=bad_probe)), patch(
        "app.service.save_one",
        new=AsyncMock(),
    ), patch("app.service.asyncio.sleep", side_effect=_stop_after_three_cycles), patch(
        "app.service.asyncio.to_thread",
        new=AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
    ):
        await service._probe_loop()

    assert service.states["landing-pad"].phase == ServicePhase.attention_only
    assert any("unhealthy confirmed" in e.get("message", "") for e in attention_events)


@pytest.mark.asyncio
async def test_apply_transition_triggers_remediation_when_enabled() -> None:
    settings = Settings(MESH_GUARDIAN_AUTO_REMEDIATE=True)
    service = MeshGuardianService(settings)
    entry = _entry()
    service.roster = RosterDocument(services=[entry])
    service.bus = MagicMock()
    service.bus.redis = AsyncMock()
    service.states["landing-pad"] = ServiceState(phase=ServicePhase.unhealthy_confirmed)

    with patch(
        "app.service.execute_remediation",
        new=AsyncMock(return_value=MagicMock(ok=True, tier=1, command=[], exit_code=0, stderr_tail="")),
    ), patch("app.service.save_one", new=AsyncMock()), patch("app.service.asyncio.to_thread", new=AsyncMock()):
        await service._apply_transition(entry, probe_status="probe_bad", equilibrium_bad=True)

    assert service.states["landing-pad"].phase == ServicePhase.post_check_grace
