"""Power-intent declaration pins.

This file exists because of a real, live-caught bug. `_publish_power_intent`
called `bus.publish(channel, kind, intent)` -- three positional arguments --
while `OrionBusAsync.publish` takes `(channel, msg)`. Every generation raised
`TypeError: OrionBusAsync.publish() takes 3 positional arguments but 4 were
given`, and the broad `except Exception` in `_publish_power_intent` swallowed
it into a WARNING. The result was the worst kind of failure: generation kept
working, the log looked like an ordinary hiccup, and `power_intent_settled`
sat at 0 rows from the day the table was created while the loop appeared
fully wired end to end.

The service had no power-intent test at all, which is why it shipped. The
key design choice here: the fake bus's `publish` is checked against the REAL
`OrionBusAsync.publish` signature via `inspect.signature`, so a hand-written
double can never drift back into accepting a call shape the real bus would
reject. A fake with a permissive `*args` signature would have passed happily
against the broken code -- that is exactly the hole this closes.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.power import PowerIntentV1

import app.main as main


class _RecordingBus:
    """Bus double whose publish() signature is pinned to the real one."""

    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, channel: str, msg) -> None:
        self.published.append((channel, msg))


class _Chassis:
    def __init__(self, bus) -> None:
        self.bus = bus


def test_recording_bus_matches_the_real_publish_signature():
    """Guard the guard: if OrionBusAsync.publish ever changes arity, this
    fails here rather than letting the double silently keep accepting a
    call shape production would reject."""
    real = inspect.signature(OrionBusAsync.publish)
    fake = inspect.signature(_RecordingBus.publish)
    assert [p for p in real.parameters if p != "self"] == [
        p for p in fake.parameters if p != "self"
    ]


def test_power_intent_is_published_as_an_envelope(monkeypatch):
    bus = _RecordingBus()
    monkeypatch.setattr(main, "_heartbeat_chassis", _Chassis(bus))
    monkeypatch.setattr(main.settings, "DIFFUSION_POWER_INTENT_ENABLED", True)

    asyncio.run(main._publish_power_intent())

    # The real bug: this list was empty on every generation.
    assert len(bus.published) == 1, "no intent was published"
    channel, env = bus.published[0]
    assert channel == "orion:power:intent"
    assert isinstance(env, BaseEnvelope), f"expected BaseEnvelope, got {type(env)}"
    assert env.kind == "power.intent.v1"
    # The settler validates the payload as PowerIntentV1 and drops anything
    # whose node is not its own, so both must survive the envelope round trip.
    intent = PowerIntentV1.model_validate(env.payload)
    assert intent.node == main.settings.NODE_NAME
    assert intent.gpu_index == main.settings.DIFFUSION_POWER_INTENT_GPU_INDEX
    assert intent.workload_kind == "reverie_diffusion"


def test_disabled_flag_publishes_nothing(monkeypatch):
    bus = _RecordingBus()
    monkeypatch.setattr(main, "_heartbeat_chassis", _Chassis(bus))
    monkeypatch.setattr(main.settings, "DIFFUSION_POWER_INTENT_ENABLED", False)

    asyncio.run(main._publish_power_intent())

    assert bus.published == []


def test_bus_failure_is_swallowed_so_generation_survives(monkeypatch):
    """The broad except is deliberate -- a failed declaration should cost a
    settlement, not an image. Pinning it so nobody 'fixes' it into a raise."""

    class _BrokenBus:
        async def publish(self, channel: str, msg) -> None:
            raise RuntimeError("bus down")

    monkeypatch.setattr(main, "_heartbeat_chassis", _Chassis(_BrokenBus()))
    monkeypatch.setattr(main.settings, "DIFFUSION_POWER_INTENT_ENABLED", True)

    asyncio.run(main._publish_power_intent())  # must not raise


def test_contradictory_config_is_loud(monkeypatch):
    """POWER_INTENT on + bus off is silently fatal to the loop, because
    publish() early-returns when disabled. Startup must say so.

    Calls the check directly rather than entering the lifespan via
    `with TestClient(app)`. Entering it fires _load_model_background(), which
    on any host with torch installed (Circe, or inside the container) starts a
    real multi-GB FLUX load onto the live GPU 2 and contends with the running
    service -- and its exit calls _gpu_executor.shutdown(), poisoning
    /generate for every later test in the same process. test_health.py's
    module docstring documents exactly this rule.
    """
    monkeypatch.setattr(main.settings, "DIFFUSION_POWER_INTENT_ENABLED", True)
    monkeypatch.setattr(main.settings, "ORION_BUS_ENABLED", False)

    records: list[str] = []
    handler_id = main.logger.add(lambda m: records.append(str(m)), level="ERROR")
    try:
        main.warn_on_contradictory_power_intent_config()
    finally:
        main.logger.remove(handler_id)

    assert any("power_intent_enabled_but_bus_disabled" in r for r in records), records


def test_no_warning_when_the_config_is_coherent(monkeypatch):
    monkeypatch.setattr(main.settings, "DIFFUSION_POWER_INTENT_ENABLED", True)
    monkeypatch.setattr(main.settings, "ORION_BUS_ENABLED", True)

    records: list[str] = []
    handler_id = main.logger.add(lambda m: records.append(str(m)), level="ERROR")
    try:
        main.warn_on_contradictory_power_intent_config()
    finally:
        main.logger.remove(handler_id)

    assert records == []


def test_an_absent_chassis_is_reported_not_silently_skipped(monkeypatch):
    """A failed chassis start is never retried, so this early return would
    otherwise mute the loop for the container's whole life."""
    monkeypatch.setattr(main, "_heartbeat_chassis", None)
    monkeypatch.setattr(main, "_power_intent_no_bus_warned", False)
    monkeypatch.setattr(main.settings, "DIFFUSION_POWER_INTENT_ENABLED", True)

    records: list[str] = []
    handler_id = main.logger.add(lambda m: records.append(str(m)), level="ERROR")
    try:
        asyncio.run(main._publish_power_intent())
        asyncio.run(main._publish_power_intent())  # latched: still one line
    finally:
        main.logger.remove(handler_id)

    hits = [r for r in records if "power_intent_no_bus" in r]
    assert len(hits) == 1, f"expected exactly one latched error, got {len(hits)}"
