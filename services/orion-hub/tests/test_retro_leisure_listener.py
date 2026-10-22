"""Orion plays Zelda, section E: Hub's consumer of the allocator's leisure start.

The runner here is a fake with Lane B's exact interface (scripts/retro_leisure.py):
``should_start(config) -> None | reason`` and ``run_leisure_session(config, act_id=) -> LeisureOutcome``.

Pins:
- no runner -> refused `runner_unavailable`, held 300 s, never a fake session;
- each gate refusal (acceptance check 5: arousal/heat, human present, cap, switch) is passed through
  by name and no act is opened;
- a refusal from the runner's own re-check inside the grace window is a refusal, not a start;
- a passed gate opens one `play` act keyed by the dispatch, answers started, and when the session ends
  the export writes the memory and closes that same act;
- a replayed dispatch id gets its first answer and never a second session; a second dispatch while
  one session runs is refused `leisure_already_running`;
- the bus boundary answers only on the start reply prefix.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.retro_leisure import (
    RETRO_LEISURE_START_REPLY_PREFIX,
    RETRO_LEISURE_START_REQUEST_KIND,
    RetroLeisureStartRequestV1,
)
from scripts import act_emit
from scripts import retro_leisure_listener as rl
from scripts.retro_leisure_listener import UNAVAILABLE_HOLD_SEC, RetroLeisureListener, gate_verdict

SRC = ServiceRef(name="orion-hub")
DISPATCH = "dispatch:proposal:retro_leisure_session:t1"
CONFIG = object()


@dataclass
class _Ended:
    """Lane B's RetroSessionEnded fields the export reads."""
    session_id: str = "retro-sess-1"
    lineage_id: str | None = "lin-1"
    leisure_id: str = "leis-1"
    game_id: str = "the-legend-of-zelda"
    started_at: float = 1792699320.0     # 2026-10-22T20:02:00Z
    ended_at: float = 1792700640.0       # +22 min
    end_reason: str = "session_cap"
    act_id: str | None = DISPATCH
    decisions_planned: int = 31
    verdicts: dict = field(default_factory=lambda: {"met": 14, "refuted": 12, "unknown": 3})

    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)


@dataclass
class _Outcome:
    started: bool
    refusal: str | None = None
    ended: _Ended | None = None


class _Runner:
    def __init__(self, gate=None, *, outcome=None, hold: asyncio.Event | None = None, raises=False):
        self.gate, self.outcome, self.hold, self.raises = gate, outcome, hold, raises
        self.runs: list[str] = []

    async def should_start(self, config):
        assert config is CONFIG
        return self.gate

    async def run_leisure_session(self, config, *, act_id=None):
        assert config is CONFIG
        self.runs.append(act_id)
        if self.raises:
            raise RuntimeError("retro down")
        if self.hold is not None:
            await self.hold.wait()
        return self.outcome if self.outcome is not None else _Outcome(True, ended=_Ended(act_id=act_id))


class _Bus:
    def __init__(self):
        self.sent = []

    async def publish(self, channel, envelope):
        self.sent.append((channel, envelope))

    def events(self):
        return [GrammarEventV1.model_validate(e.payload) for ch, e in self.sent if ch == "orion:grammar:event"]


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_ACT_LEDGER_ENABLED", True, raising=False)
    monkeypatch.setattr(rl, "START_GRACE_SEC", 0.05)


@pytest.fixture
def exported(monkeypatch):
    from scripts import retro_memory_export as rx

    calls = []

    async def fake_export(*, pool, bus, ended, enabled=None):
        calls.append(ended)
        await act_emit.publish_play_closed(bus, dispatch_id=ended.act_id, session_id=ended.session_id,
                                           produced=True, note="exported")
        return rx.ExportResult("exported", ended.session_id, card_id="c1")

    monkeypatch.setattr(rx, "export_leisure_outcome", fake_export)
    return calls


def _listener(runner):
    listener = RetroLeisureListener(SRC, config_provider=lambda: CONFIG, pool_provider=lambda: None,
                                    runner_provider=lambda: runner)
    listener.bus = _Bus()
    return listener


def _req(dispatch_id=DISPATCH):
    return RetroLeisureStartRequestV1(dispatch_id=dispatch_id, correlation={"proposal_id": "p1"})


async def _settle(listener):
    for _ in range(50):
        if not listener._followers:
            return
        await asyncio.sleep(0.01)


@pytest.mark.parametrize("raw,expected", [
    (None, (True, None)), (True, (True, None)), ("cabinet_hot", (False, "cabinet_hot")),
    ("", (False, "unnamed_refusal")), (False, (False, "unnamed_refusal")),
])
def test_gate_verdict(raw, expected) -> None:
    assert gate_verdict(raw) == expected


@pytest.mark.asyncio
async def test_no_runner_is_refused_and_held_never_faked() -> None:
    out = await _listener(None).start(_req())
    assert (out.state, out.refusal, out.retry_after_sec, out.session_id) == (
        "refused", "runner_unavailable", UNAVAILABLE_HOLD_SEC, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["arousal_strained", "cabinet_hot", "browser_connected", "human_recent",
                                    "cooldown", "daily_cap", "disabled"])
async def test_each_gate_refusal_is_passed_through_by_name(reason) -> None:
    runner = _Runner(reason)
    listener = _listener(runner)
    out = await listener.start(_req())
    assert (out.state, out.refusal) == ("refused", reason)
    assert runner.runs == [] and listener.bus.events() == []


@pytest.mark.asyncio
async def test_a_refusal_from_the_runners_own_recheck_is_a_refusal() -> None:
    runner = _Runner(None, outcome=_Outcome(False, refusal="human_recent"))
    listener = _listener(runner)
    out = await listener.start(_req())
    assert (out.state, out.refusal) == ("refused", "human_recent")
    assert runner.runs == [DISPATCH] and listener.bus.events() == []


@pytest.mark.asyncio
async def test_a_runner_that_raises_at_start_is_refused_start_failed() -> None:
    out = await _listener(_Runner(None, raises=True)).start(_req())
    assert (out.state, out.refusal) == ("refused", "start_failed")


@pytest.mark.asyncio
async def test_a_session_opens_the_play_act_and_its_end_exports_and_closes_it(exported) -> None:
    hold = asyncio.Event()
    runner = _Runner(None, hold=hold)
    listener = _listener(runner)
    out = await listener.start(_req())
    assert (out.state, out.object_ref) == ("started", f"retro_leisure_dispatch:{DISPATCH}")
    (opened,) = listener.bus.events()
    assert opened.event_kind == "act_opened"
    assert (opened.act.verb, opened.act.actor, opened.act.test_before) == (
        "play", "hub.retro_leisure", f"retro_leisure_dispatch:{DISPATCH}")
    assert opened.act.expect.horizon_sec == act_emit.PLAY_HORIZON_SEC == 2400.0
    # While it runs: a second dispatch is refused, a replay gets the first answer.
    busy = await listener.start(_req("dispatch:other"))
    assert (busy.state, busy.refusal) == ("refused", "leisure_already_running")
    assert await listener.start(_req()) == out and runner.runs == [DISPATCH]
    hold.set()
    await _settle(listener)
    assert [e.act_id for e in exported] == [DISPATCH]
    opened, closed = listener.bus.events()
    assert closed.event_kind == "act_closed" and closed.act.act_id == opened.act.act_id


@pytest.mark.asyncio
async def test_a_session_that_ends_without_data_closes_the_act_at_zero() -> None:
    hold = asyncio.Event()
    listener = _listener(_Runner(None, hold=hold, outcome=_Outcome(True, refusal=None, ended=None)))
    assert (await listener.start(_req())).state == "started"
    hold.set()
    await _settle(listener)
    _, closed = listener.bus.events()
    assert closed.act.outcome.value == 0.0 and closed.act.outcome.note.startswith("no_session")


@pytest.mark.asyncio
async def test_handle_answers_only_on_the_start_reply_prefix() -> None:
    listener = _listener(_Runner("cabinet_hot"))
    good = BaseEnvelope(kind=RETRO_LEISURE_START_REQUEST_KIND, source=SRC,
                        correlation_id="7d3c6f8e-1111-4222-8333-444455556666",
                        reply_to=f"{RETRO_LEISURE_START_REPLY_PREFIX}abc", payload=_req().model_dump(mode="json"))
    await listener.handle(good)
    replies = [(ch, e) for ch, e in listener.bus.sent if ch.startswith(RETRO_LEISURE_START_REPLY_PREFIX)]
    assert len(replies) == 1 and replies[0][1].payload["refusal"] == "cabinet_hot"
    await listener.handle(good.model_copy(update={"reply_to": "orion:some:other:channel"}))
    assert not any(ch == "orion:some:other:channel" for ch, _ in listener.bus.sent)


def test_the_real_runner_module_is_found_or_reported_missing() -> None:
    mod = rl.import_runner()
    # Lane B's module may not be merged yet; either way the listener never fakes a session.
    assert mod is None or (hasattr(mod, "should_start") and hasattr(mod, "run_leisure_session"))
