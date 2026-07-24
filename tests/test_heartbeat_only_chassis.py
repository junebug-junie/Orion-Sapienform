"""Regression coverage for `HeartbeatOnly` (orion/core/bus/bus_service_chassis.py).

Added alongside the pilot-5 service-heartbeat rollout (see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md). The 5 pilot
services (orion-hub, orion-harness-governor, orion-cortex-gateway, orion-field-digester,
orion-substrate-runtime) all construct a standalone `HeartbeatOnly` chassis rather than a bare
`BaseChassis`, because `BaseChassis._run()` is abstract (raises NotImplementedError) --
instantiating `BaseChassis` directly would make `_supervise_run()` treat every startup as an
immediate crash and busy-loop reconnect/retry forever with exponential backoff. This test
guards that specific footgun and confirms `start_background()` schedules the real heartbeat
loop (not just the supervise-crash-loop) for the shared base class every pilot service reuses.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly


def _cfg(**overrides) -> ChassisConfig:
    base = dict(
        service_name="test-heartbeat-only",
        service_version="0.1.0",
        node_name="node-a",
        bus_url="redis://localhost:6379/0",
        bus_enabled=True,
        heartbeat_interval_sec=0.01,
    )
    base.update(overrides)
    return ChassisConfig(**base)


@pytest.mark.asyncio
async def test_run_waits_for_stop_without_raising() -> None:
    """HeartbeatOnly._run() must not raise NotImplementedError (the BaseChassis default) --
    it should simply block until stop is requested, so _supervise_run() never treats a fresh
    heartbeat-only service as having crashed on its very first loop iteration."""
    chassis = HeartbeatOnly(_cfg())
    run_task = asyncio.create_task(chassis._run())
    await asyncio.sleep(0)
    assert not run_task.done(), "_run() must block on _stop, not return/raise immediately"

    chassis._stop.set()
    await asyncio.wait_for(run_task, timeout=1.0)
    assert run_task.exception() is None


@pytest.mark.asyncio
async def test_start_background_schedules_heartbeat_loop_not_crash_retry() -> None:
    """start_background() (the entrypoint every pilot-5 service's FastAPI lifespan calls)
    must schedule a real orion-heartbeat task, and _supervise_run() must not immediately log a
    crash for it -- confirming the real fix (HeartbeatOnly._run) is what's wired in, not a bare
    BaseChassis that would busy-loop-retry on NotImplementedError."""
    chassis = HeartbeatOnly(_cfg(heartbeat_interval_sec=100.0))
    chassis.bus.connect = AsyncMock()

    with patch.object(chassis, "_heartbeat_loop", new=AsyncMock()) as heartbeat_loop:
        await chassis.start_background()
        await asyncio.sleep(0)

        task_names = {t.get_name() for t in chassis._tasks}
        assert "orion-heartbeat" in task_names
        assert f"{chassis.cfg.service_name}-run" in task_names

        run_task = next(t for t in chassis._tasks if t.get_name() == f"{chassis.cfg.service_name}-run")
        assert not run_task.done(), "run task crashed immediately instead of waiting on stop"

        await chassis.stop()
        heartbeat_loop.assert_awaited()
