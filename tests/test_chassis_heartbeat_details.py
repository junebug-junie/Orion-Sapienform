"""Regression coverage for the optional `heartbeat_details` provider hook added to
`BaseChassis`/`Clock` (orion/core/bus/bus_service_chassis.py).

Added alongside orion-biometrics' real host disk-usage telemetry, piggybacked onto its
existing bus-native SystemHealthV1 heartbeat (see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md,
"Cross-node disk telemetry: bus-piggyback vs. SSH?"). `SystemHealthV1.details` was
previously always published as a hardcoded `{}` -- this hook lets a chassis owner
(any of the 25 services already using BaseChassis) supply extra free-form data
without touching `_heartbeat_loop()` itself. This is additive: omitting the kwarg
(as every pre-existing caller does) must reproduce the old `details={}` behavior
exactly.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Clock, HeartbeatOnly


def _cfg(**overrides) -> ChassisConfig:
    base = dict(
        service_name="test-heartbeat-details",
        service_version="0.1.0",
        node_name="node-a",
        bus_url="redis://localhost:6379/0",
        bus_enabled=True,
        heartbeat_interval_sec=0.01,
    )
    base.update(overrides)
    return ChassisConfig(**base)


async def _run_one_heartbeat(chassis) -> dict:
    """Run a single _heartbeat_loop iteration and return the published details dict."""
    publish = AsyncMock()
    chassis.bus = SimpleNamespace(publish=publish, reconnect=AsyncMock())
    chassis._source = lambda: ServiceRef(
        name=chassis.cfg.service_name, node=chassis.cfg.node_name, version=chassis.cfg.service_version
    )
    chassis._stop.clear()
    # Stop after the first publish so the while loop exits on the next check.
    orig_publish = publish

    async def _publish_then_stop(*args, **kwargs):
        chassis._stop.set()
        return await orig_publish(*args, **kwargs)

    chassis.bus.publish = _publish_then_stop
    await chassis._heartbeat_loop()
    assert publish.await_count == 1
    env = publish.await_args.args[1]
    return env.payload["details"]


@pytest.mark.asyncio
async def test_no_provider_publishes_empty_details_unchanged() -> None:
    """Omitting heartbeat_details (every pre-existing BaseChassis caller) must keep
    publishing details={} exactly as before this patch."""
    chassis = HeartbeatOnly(_cfg())
    details = await _run_one_heartbeat(chassis)
    assert details == {}


@pytest.mark.asyncio
async def test_provider_result_is_folded_into_details() -> None:
    chassis = HeartbeatOnly(_cfg(), heartbeat_details=lambda: {"disk_usage_pct": {"docker": 87.3}})
    details = await _run_one_heartbeat(chassis)
    assert details == {"disk_usage_pct": {"docker": 87.3}}


@pytest.mark.asyncio
async def test_provider_exception_does_not_block_heartbeat() -> None:
    """A broken details provider (e.g. a filesystem read failure) must never prevent
    the liveness pulse itself from publishing."""

    def _boom() -> dict:
        raise RuntimeError("disk read failed")

    chassis = HeartbeatOnly(_cfg(), heartbeat_details=_boom)
    details = await _run_one_heartbeat(chassis)
    assert details == {}


@pytest.mark.asyncio
async def test_provider_non_dict_result_is_ignored() -> None:
    chassis = HeartbeatOnly(_cfg(), heartbeat_details=lambda: "not-a-dict")
    details = await _run_one_heartbeat(chassis)
    assert details == {}


@pytest.mark.asyncio
async def test_clock_threads_heartbeat_details_to_base_chassis() -> None:
    """Clock (the chassis subclass orion-biometrics actually uses) must forward the
    kwarg through to BaseChassis, not just HeartbeatOnly."""

    async def _tick() -> None:
        return None

    clock = Clock(
        _cfg(),
        interval_sec=1.0,
        tick=_tick,
        heartbeat_details=lambda: {"disk_usage_pct": {"scripts": 14.1}},
    )
    details = await _run_one_heartbeat(clock)
    assert details == {"disk_usage_pct": {"scripts": 14.1}}
