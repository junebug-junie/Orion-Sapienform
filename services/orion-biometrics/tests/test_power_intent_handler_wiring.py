"""Power-intent handler WIRING pins (distinct from the settlement arithmetic).

test_power_intent_settlement.py covers `settle()`/`summarize()` thoroughly --
9 tests, all against app.power_intent. None of them import app.main, so none
of them ever exercised the path that actually delivers a settlement. That gap
is not hypothetical: the handler's publish call referenced a bare `bus` name
that existed in no enclosing scope, and the arithmetic tests stayed green
through the entire outage.

The failure was maximally quiet. settle() ran its full sampling window against
a real GPU, logged `power_intent_settled ... outcome=settled samples=18
peak=48.8`, and only THEN raised NameError -- inside a fire-and-forget task
nobody awaited, so it surfaced as "Task exception was never retrieved" and
`power_intent_settled` stayed at 0 rows while every log line said the loop
worked.

These tests assert the settled event is actually handed to a bus.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SVC = Path(__file__).resolve().parents[1]
_REPO = _SVC.parents[1]
sys.path.insert(0, str(_SVC))
os.environ.setdefault(
    "NODE_CATALOG_PATH", str(_REPO / "config" / "biometrics" / "node_catalog.yaml")
)

import app.main as main  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.power import PowerIntentSettledV1, PowerIntentV1  # noqa: E402


class _RecordingBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []
        self.enabled = True

    async def publish(self, channel: str, msg) -> None:
        self.published.append((channel, msg))


def _intent(node: str) -> PowerIntentV1:
    now = datetime.now(timezone.utc)
    return PowerIntentV1(
        intent_id="test-intent",
        workload_kind="reverie_diffusion",
        node=node,
        gpu_index=2,
        expected_duration_sec=0.01,
        expected_watts=None,
        deadline=now + timedelta(seconds=1),
    )


def _envelope(intent: PowerIntentV1) -> BaseEnvelope:
    return BaseEnvelope(
        kind="power.intent.v1",
        source=ServiceRef(name="diffusion-host", version="0.2.0", node=intent.node),
        payload=intent.model_dump(mode="json"),
    )


async def _drain() -> None:
    """The handler fires the settlement into a detached task on purpose, so
    the test must await it explicitly. Gathering (rather than sleeping a
    fixed amount) also means a raised exception inside that task surfaces
    here instead of vanishing into "Task exception was never retrieved" --
    the exact way the original bug stayed invisible in production."""
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending)


def test_a_settled_intent_is_actually_published(monkeypatch):
    bus = _RecordingBus()
    monkeypatch.setattr(main.settings, "NODE_NAME", "circe")
    monkeypatch.setattr(main, "sample_gpu_watts", lambda idx: 100.0)
    monkeypatch.setattr(main.settings, "POWER_INTENT_SAMPLE_INTERVAL_SEC", 0.001)

    handler = main.make_power_intent_handler(lambda: bus)

    async def _go():
        await handler(_envelope(_intent("circe")))
        await _drain()

    asyncio.run(_go())

    # The bug: this list was empty while the log said outcome=settled.
    assert len(bus.published) == 1, "settlement was computed but never published"
    channel, env = bus.published[0]
    assert channel == main.settings.POWER_INTENT_SETTLED_CHANNEL
    assert env.kind == "power.intent.settled.v1"
    settled = PowerIntentSettledV1.model_validate(env.payload)
    assert settled.intent_id == "test-intent"
    assert settled.node == "circe"


def test_another_nodes_intent_is_ignored(monkeypatch):
    bus = _RecordingBus()
    monkeypatch.setattr(main.settings, "NODE_NAME", "circe")
    monkeypatch.setattr(main, "sample_gpu_watts", lambda idx: 100.0)
    monkeypatch.setattr(main.settings, "POWER_INTENT_SAMPLE_INTERVAL_SEC", 0.001)

    handler = main.make_power_intent_handler(lambda: bus)

    async def _go():
        await handler(_envelope(_intent("athena")))
        await _drain()

    asyncio.run(_go())
    assert bus.published == []


def test_the_bus_is_resolved_late_not_captured_at_build_time(monkeypatch):
    """The Hunter that owns the connection is constructed AFTER the handler,
    so the factory must take a callable. Capturing a bus eagerly would
    reintroduce the original ordering problem in a new shape."""
    monkeypatch.setattr(main.settings, "NODE_NAME", "circe")
    monkeypatch.setattr(main, "sample_gpu_watts", lambda idx: 100.0)
    monkeypatch.setattr(main.settings, "POWER_INTENT_SAMPLE_INTERVAL_SEC", 0.001)

    holder: dict[str, _RecordingBus] = {}
    handler = main.make_power_intent_handler(lambda: holder["bus"])
    # Bus does not exist yet at build time; only bound now.
    holder["bus"] = _RecordingBus()

    async def _go():
        await handler(_envelope(_intent("circe")))
        await _drain()

    asyncio.run(_go())
    assert len(holder["bus"].published) == 1
