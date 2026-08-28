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
