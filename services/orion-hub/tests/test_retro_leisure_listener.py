"""Orion plays Zelda, section E: Hub's side of the allocator's leisure start.

Pins:
- no runner (Lane B not wired) -> refused `runner_unavailable`, held 300 s, never a fake session;
- the runner's own gate decides, in any of its documented answer shapes, and a refusal keeps the
  runner's named reason and retry clock (acceptance check 5: arousal/heat, human present, cap, switch);
- a passed gate starts one session, opens the `play` act on that dispatch's key, and answers started;
- a replayed dispatch id gets its first answer and never a second session;
- the bus boundary answers only on the start reply prefix.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.retro_leisure import (
    RETRO_LEISURE_START_REPLY_PREFIX,
    RETRO_LEISURE_START_REQUEST_KIND,
    RetroLeisureStartRequestV1,
)
from scripts import act_emit
from scripts.retro_leisure_listener import UNAVAILABLE_HOLD_SEC, RetroLeisureListener, gate_verdict

SRC = ServiceRef(name="orion-hub")
DISPATCH = "dispatch:proposal:retro_leisure_session:t1"


class _Bus:
    def __init__(self):
        self.sent = []

    async def publish(self, channel, envelope):
        self.sent.append((channel, envelope))

    def events(self):
        return [GrammarEventV1.model_validate(e.payload) for ch, e in self.sent if ch == "orion:grammar:event"]


class _Runner:
    def __init__(self, verdict=True, *, fail=False, is_async=False):
        self.verdict, self.fail, self.starts = verdict, fail, []
        if is_async:
            async def should_start():
                return self.verdict

            async def start(**kw):
                return self._start(**kw)

            self.should_start, self.start = should_start, start

    def should_start(self):
        return self.verdict

    def start(self, **kw):
        return self._start(**kw)

    def _start(self, *, dispatch_id, correlation):
        if self.fail:
            raise RuntimeError("no save")
        self.starts.append(dispatch_id)
        return {"session_id": f"leisure-{len(self.starts)}"}


@pytest.fixture(autouse=True)
def _ledger_on(monkeypatch):
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_ACT_LEDGER_ENABLED", True, raising=False)


def _listener(runner):
    listener = RetroLeisureListener(lambda: runner, SRC)
    listener.bus = _Bus()
    return listener


def _req(dispatch_id=DISPATCH):
    return RetroLeisureStartRequestV1(dispatch_id=dispatch_id, correlation={"proposal_id": "p1"})


@pytest.mark.asyncio
async def test_no_runner_is_refused_and_held_never_faked() -> None:
    out = await _listener(None).start(_req())
    assert (out.state, out.refusal, out.retry_after_sec, out.session_id) == (
        "refused", "runner_unavailable", UNAVAILABLE_HOLD_SEC, None)


@pytest.mark.parametrize("verdict,expected", [
    (True, (True, None, None)),
    (False, (False, "unnamed_refusal", None)),
    ((False, "cabinet_hot", 600), (False, "cabinet_hot", 600.0)),
    ((False, "arousal_strained"), (False, "arousal_strained", None)),
    (SimpleNamespace(ok=False, reason="human_present", retry_after_sec=420.0), (False, "human_present", 420.0)),
    ({"allowed": False, "reason": "daily_cap", "retry_after_sec": 3600}, (False, "daily_cap", 3600.0)),
    ({"allowed": True, "reason": "ignored"}, (True, None, None)),
])
def test_gate_verdict_shapes(verdict, expected) -> None:
    assert gate_verdict(verdict) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["arousal_strained", "cabinet_hot", "human_present", "session_cap", "disabled"])
async def test_each_runner_refusal_is_passed_through_with_its_clock(reason) -> None:
    runner = _Runner((False, reason, 900))
    listener = _listener(runner)
    out = await listener.start(_req())
    assert (out.state, out.refusal, out.retry_after_sec) == ("refused", reason, 900.0)
    assert runner.starts == [] and listener.bus.events() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_a_passed_gate_starts_one_session_and_opens_the_play_act(is_async) -> None:
    runner = _Runner(True, is_async=is_async)
    listener = _listener(runner)
    out = await listener.start(_req())
    assert (out.state, out.session_id, out.object_ref) == (
        "started", "leisure-1", f"retro_leisure_dispatch:{DISPATCH}")
    (ev,) = listener.bus.events()
    assert ev.event_kind == "act_opened"
    act = ev.act
    assert (act.verb, act.actor, act.test_before, act.object_ref) == (
        "play", "hub.retro_leisure", f"retro_leisure_dispatch:{DISPATCH}", "retro_session:leisure-1")
    assert act.expect.signal_id == "artifact_produced" and act.expect.horizon_sec == act_emit.PLAY_HORIZON_SEC == 2400.0
    assert "1800s session cap" in act.expect.basis
    # Replay: same answer, no second session, no second act.
    again = await listener.start(_req())
    assert again == out and runner.starts == [DISPATCH] and len(listener.bus.events()) == 1


@pytest.mark.asyncio
async def test_a_start_that_raises_is_refused_start_failed() -> None:
    listener = _listener(_Runner(True, fail=True))
    out = await listener.start(_req())
    assert (out.state, out.refusal) == ("refused", "start_failed") and listener.bus.events() == []


@pytest.mark.asyncio
async def test_handle_answers_only_on_the_start_reply_prefix() -> None:
    listener = _listener(_Runner(True))
    good = BaseEnvelope(kind=RETRO_LEISURE_START_REQUEST_KIND, source=SRC, correlation_id="7d3c6f8e-1111-4222-8333-444455556666",
                        reply_to=f"{RETRO_LEISURE_START_REPLY_PREFIX}abc", payload=_req().model_dump(mode="json"))
    await listener.handle(good)
    replies = [(ch, e) for ch, e in listener.bus.sent if ch.startswith(RETRO_LEISURE_START_REPLY_PREFIX)]
    assert len(replies) == 1 and replies[0][1].payload["state"] == "started"
    bad = good.model_copy(update={"reply_to": "orion:some:other:channel"})
    await listener.handle(bad)
    assert not any(ch == "orion:some:other:channel" for ch, _ in listener.bus.sent)


@pytest.mark.asyncio
async def test_play_act_close_carries_the_same_act_id() -> None:
    bus = _Bus()
    assert await act_emit.publish_play_opened(bus, dispatch_id=DISPATCH, session_id="s")
    assert await act_emit.publish_play_closed(bus, dispatch_id=DISPATCH, session_id="s", produced=True, note="ok")
    opened, closed = bus.events()
    assert opened.act.act_id == closed.act.act_id and closed.act.outcome.value == 1.0
