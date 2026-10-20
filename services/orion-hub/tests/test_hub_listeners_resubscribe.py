"""Every long-lived hub bus listener must survive a Redis drop.

Live 2026-10-20: a bare ``async with bus.subscribe(...)`` loop ended on one redis
ConnectionError/TimeoutError and the process stayed up looking healthy. These
listeners now run under ``orion.core.bus.resilience.run_subscription``; each
test here makes the first subscribe raise and proves the listener opens a new
session and delivers the next message to the site's own handler.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
from typing import Any, Callable

import pytest

from orion.core.bus import resilience
from scripts import (
    biometrics_cache,
    bus_synaptic_trigger_notifier,
    cognition_trace_cache,
    curiosity_investigation,
    embodiment_outcome_cache,
    notification_cache,
    signals_inspect_cache,
    vision_frame_cache,
)


class _PubSub:
    def __init__(self, messages: list[dict]) -> None:
        self.messages = list(messages)

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        if self.messages:
            return self.messages.pop(0)
        await asyncio.sleep(0.01)
        return None


class _FlakyBus:
    """First subscribe raises like a dropped Redis; later ones deliver ``messages``."""

    enabled = True

    def __init__(self, messages: list[dict]) -> None:
        self.messages = messages
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def subscribe(self, *channels: str, patterns: bool = False):
        bus = self

        @contextlib.asynccontextmanager
        async def _cm():
            bus.calls.append((channels, patterns))
            if len(bus.calls) == 1:
                raise ConnectionError("Error 111 connecting to redis")
            yield _PubSub(bus.messages)

        return _cm()


def _msg(n: int, mtype: str = "message") -> dict:
    return {"type": mtype, "channel": b"c", "data": n}


def _bare(cls: type, **attrs: Any) -> Any:
    obj = cls.__new__(cls)  # skip settings-heavy constructors; set only what _run reads
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real run_subscription, with a 10 ms retry wait instead of 1 s."""
    fast = functools.partial(resilience.run_subscription, max_backoff_sec=0.01)
    for module in (
        biometrics_cache, bus_synaptic_trigger_notifier, cognition_trace_cache, curiosity_investigation,
        embodiment_outcome_cache, notification_cache, signals_inspect_cache, vision_frame_cache,
    ):
        monkeypatch.setattr(module, "run_subscription", fast, raising=False)


# (listener name, factory(seen) -> (object, loop coroutine fn), expected channels, patterns, mtype)
def _cache_case(module, cls_name, handler_attr, loop_attr, **attrs):
    def make(seen: list) -> tuple[Any, Callable]:
        obj = _bare(getattr(module, cls_name), **attrs)
        setattr(obj, handler_attr, seen.append)
        return obj, getattr(obj, loop_attr)

    return make


CASES = [
    ("hub.cognition_trace_cache", _cache_case(cognition_trace_cache, "CognitionTraceCache", "_handle_message", "_run", subscribe_channel="orion:cognition:trace"), ("orion:cognition:trace",), False, "message"),
    ("hub.signals_inspect_cache", _cache_case(signals_inspect_cache, "SignalsInspectCache", "_handle_message", "_run", subscribe_pattern="orion:signals:*"), ("orion:signals:*",), True, "pmessage"),
    ("hub.embodiment_outcome_cache", _cache_case(embodiment_outcome_cache, "EmbodimentOutcomeCache", "_handle_message", "_run", channel="orion:embodiment:outcome"), ("orion:embodiment:outcome",), False, "message"),
    ("hub.biometrics_cache", _cache_case(biometrics_cache, "BiometricsCache", "_handle_message", "_run"), ("orion:biometrics:summary", "orion:biometrics:induction", "orion:biometrics:cluster"), False, "message"),
    ("hub.bus_synaptic_trigger_notifier", _cache_case(bus_synaptic_trigger_notifier, "BusSynapticTriggerNotifier", "_handle_metacog_trigger", "_run", metacog_trigger_channel="orion:equilibrium:metacog:trigger"), ("orion:equilibrium:metacog:trigger",), False, "message"),
    ("hub.vision_frame_cache", _cache_case(vision_frame_cache, "VisionFrameCache", "_handle_message", "_run", channel="orion:vision:frames", stream_ids={"carbon"}), ("orion:vision:frames",), False, "message"),
    ("hub.notification_cache", _cache_case(notification_cache, "NotificationCache", "_handle_message", "_run", channel="orion:notify:in_app"), ("orion:notify:in_app",), False, "message"),
    ("hub.curiosity_run_state", _cache_case(curiosity_investigation, "CuriosityInvestigation", "_handle_run_state", "_run_state_loop", _stop=None), (curiosity_investigation.DURABLE_RUN_STATE_CHANNEL,), False, "message"),
]


@pytest.mark.parametrize("name,make,channels,patterns,mtype", CASES, ids=[c[0] for c in CASES])
def test_listener_resubscribes_after_the_first_subscribe_fails(name, make, channels, patterns, mtype) -> None:
    async def scenario() -> tuple[list, _FlakyBus, dict]:
        resilience._SUBSCRIPTION_STATES.pop(name, None)
        seen: list = []
        bus = _FlakyBus([_msg(1, mtype), _msg(2, mtype)])
        obj, loop = make(seen)
        obj._bus = bus
        if name == "hub.curiosity_run_state":
            obj._stop = asyncio.Event()
        task = asyncio.create_task(loop())
        for _ in range(500):
            if len(seen) == 2:
                break
            await asyncio.sleep(0.01)
        row = next((r for r in resilience.subscription_health()["listeners"] if r["name"] == name), None)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        resilience._SUBSCRIPTION_STATES.pop(name, None)
        return seen, bus, row

    seen, bus, row = asyncio.run(scenario())
    assert [m["data"] for m in seen] == [1, 2]
    assert bus.calls == [(channels, patterns), (channels, patterns)]
    assert row is not None, "listener never ran under run_subscription"
    assert row["failures_total"] == 1 and row["subscribed"] is True
    assert row["last_failure"].startswith("ConnectionError")


def test_curiosity_turn_requests_resubscribe_and_never_serialize_the_stream() -> None:
    """The turn handler starts a task and returns None, so a slow turn does not block the next."""
    name = "hub.curiosity_turn_request"

    async def scenario() -> tuple[list, _FlakyBus]:
        resilience._SUBSCRIPTION_STATES.pop(name, None)
        started: list = []
        release = asyncio.Event()

        async def slow_turn(msg: dict) -> None:
            started.append(msg["data"])
            await release.wait()

        bus = _FlakyBus([_msg(1), _msg(2)])
        stop = asyncio.Event()
        inv = _bare(
            curiosity_investigation.CuriosityInvestigation,
            _bus=bus, _stop=stop, _turn_tasks=set(), _serve_turn_request=slow_turn,
        )
        task = asyncio.create_task(inv._turn_request_loop())
        for _ in range(500):
            if len(started) == 2:
                break
            await asyncio.sleep(0.01)
        both_running = len(started) == 2 and len(inv._turn_tasks) == 2
        release.set()
        stop.set()  # the listener honors the investigation's stop event
        await asyncio.wait_for(task, timeout=5)
        await asyncio.gather(*list(inv._turn_tasks))
        resilience._SUBSCRIPTION_STATES.pop(name, None)
        assert both_running, "second turn waited on the first: the handler returned its task"
        return started, bus

    started, bus = asyncio.run(scenario())
    assert started == [1, 2]
    assert len(bus.calls) == 2


def test_hub_health_reads_degraded_while_a_listener_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """A listener stuck in backoff must show on /health, not hide behind status=ok."""
    from scripts import api_routes

    name = "hub.health_probe"
    state = resilience.SubscriptionState(name=name, channel="orion:test", subscribed=True)
    monkeypatch.setattr(resilience, "_SUBSCRIPTION_STATES", {name: state})
    up = api_routes.health()
    assert up["status"] == "ok" and up["listeners_ok"] is True
    state.subscribed = False
    down = api_routes.health()
    assert down["status"] == "degraded" and down["listeners_ok"] is False
    assert down["listeners_down"] == [name]


class _Codec:
    def decode(self, data):
        class _Decoded:
            pass

        out = _Decoded()
        out.ok = isinstance(data, dict)
        if out.ok:
            out.envelope = type("Env", (), {"payload": data})()
        return out


def test_embodiment_handler_records_outcome_fields_and_skips_undecodable() -> None:
    cache = embodiment_outcome_cache.EmbodimentOutcomeCache(enabled=True, channel="c")
    cache._bus = type("B", (), {"codec": _Codec()})()
    payload = {
        "intent_correlation_id": "corr-1", "source": "mc", "status": "done",
        "reason": "ok", "player_id": "p1", "extra": "ignored",
    }
    cache._handle_message({"type": "message", "data": payload})
    cache._handle_message({"type": "message", "data": b"not-an-envelope"})
    assert cache.recent() == [
        {"intent_correlation_id": "corr-1", "source": "mc", "status": "done", "reason": "ok", "player_id": "p1"}
    ]
