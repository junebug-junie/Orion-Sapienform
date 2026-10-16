"""World-pulse reading `look` act: opened once, at the moment the read's work starts.

Live 2026-10-16: the act was opened on every poll of a queued read (243 act_opened
grammar events for 10 acts), dated at submission, so 6 of the 7 reads that produced
were graded 0: their durable runs waited 0.5 to 3.3 h for a GPU, then read in 4 to
12 minutes. These drive the real _read_claimed -> _generate -> poll_turn seam with
a scripted durable-runs status.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reading_turn import ReadingTurnResultV1
from orion.substrate.store import InMemorySubstrateGraphStore
from orion.world_pulse_read.durable import ReadingPending
from scripts.world_pulse_read_pipeline import WorldPulseReadPipeline
from test_world_pulse_read_pipeline import (
    _FakeBus,
    _FakeConn,
    _handoff,
    _pipeline,
    _seed,
    _seed_queue,
)

pytestmark = pytest.mark.usefixtures("reading_dns")

STARTED = "2026-10-16T15:17:37.356183Z"
STARTED_AT = datetime(2026, 10, 16, 15, 17, 37, 356183, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _ledger_on(monkeypatch):
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_ACT_LEDGER_ENABLED", True, raising=False)


class _Conn(_FakeConn):
    async def execute(self, sql, *args):
        # release_claim (a still-pending durable turn hands the seed back to the queue).
        if "SET status='pending', claimed_at=NULL" in sql and args[0] in self.rows:
            self.rows[args[0]].update(status="pending", claimed_at=None)
            return "UPDATE 1"
        return await super().execute(sql, *args)


def _state(status: str, *, started: bool, error: str | None = None) -> dict:
    history = [{"event": "run.accepted", "generated_at": "2026-10-16T12:03:31.833202Z"},
               {"event": "run.waiting_resource", "generated_at": "2026-10-16T12:03:36.4Z"}]
    if started:
        history += [{"event": "run.resource_granted", "generated_at": "2026-10-16T15:17:37.12Z"},
                    {"event": "run.started", "generated_at": STARTED}]
    return {"run_id": "reading-run", "workflow_kind": "reading.turn", "status": status,
            "created_at": "2026-10-16T12:03:31.833202+00:00", "queue_wait_seconds": 11645.29,
            "history": history, "work_started": started, "error": error}


def _scripted(monkeypatch, states: list[dict]):
    """Patch bind_turn/poll_turn: each poll reports the next scripted status."""
    script = list(states)

    async def bind(conn, brief, correlation_id):
        return SimpleNamespace(run_id="reading-run", brief=SimpleNamespace(prompt=brief.prompt))

    async def poll(request, base_url, *, on_state=None):
        state = script.pop(0)
        if on_state is not None:
            on_state(state)
        if state["status"] == "completed":
            return ReadingTurnResultV1(run_id="reading-run", correlation_id="c", ok=True,
                                       text="read", source_fetches=[])
        if state["status"] == "failed":
            if not state["work_started"]:
                raise ValueError(f"turn_deferred:reading_admission:{state['error']}")
            raise ValueError(state["error"])
        raise ReadingPending(f"durable:reading-run:{state['status']}")

    g = WorldPulseReadPipeline._generate.__globals__
    monkeypatch.setitem(g, "bind_turn", bind)
    monkeypatch.setitem(g, "poll_turn", poll)
    return script


def _wire(pipe: WorldPulseReadPipeline) -> None:
    handoff = _handoff()

    async def stage1(seed):
        await pipe._generate("Read source", "corr", seed_id=seed.seed_id)
        return handoff

    pipe._stage1_read = stage1  # type: ignore[method-assign]


def _acts(bus: _FakeBus) -> list[GrammarEventV1]:
    return [GrammarEventV1.model_validate(e.payload) for ch, e in bus.published if ch == "orion:grammar:event"]


def _ticks(pipe, conn, n: int) -> list:
    async def run():
        await _seed_queue(conn)
        return [await pipe.tick(force=True) for _ in range(n)]

    return asyncio.run(run())


def test_queued_read_opens_once_at_work_start_and_closes_at_one(monkeypatch):
    bus, conn = _FakeBus(), _Conn()
    pipe = _pipeline(bus, conn, InMemorySubstrateGraphStore())
    _wire(pipe)
    _scripted(monkeypatch, [
        _state("waiting_resource", started=False),
        _state("waiting_resource", started=False),
        _state("running", started=True),
        _state("running", started=True),
        _state("completed", started=True),
    ])

    outcomes = _ticks(pipe, conn, 5)

    assert outcomes[-1] is None and conn.rows["finding:r1:x"]["status"] == "done"
    acts = _acts(bus)
    opens = [e for e in acts if e.event_kind == "act_opened"]
    closes = [e for e in acts if e.event_kind == "act_closed"]
    assert len(opens) == 1, "one open per act, not one per poll"
    # The writer dates opened_at (and so the horizon) from observed_at.
    assert opens[0].observed_at == STARTED_AT, "the clock starts when the read starts, not in the GPU queue"
    assert [c.act.outcome.value for c in closes] == [1.0]
    assert acts.index(opens[0]) < acts.index(closes[0])


def test_read_refused_before_work_is_not_an_act(monkeypatch):
    bus, conn = _FakeBus(), _Conn()
    pipe = _pipeline(bus, conn, InMemorySubstrateGraphStore(), max_attempts=1)
    _wire(pipe)
    _scripted(monkeypatch, [
        _state("waiting_resource", started=False),
        _state("failed", started=False, error="deadline"),
    ])

    _ticks(pipe, conn, 2)

    assert _acts(bus) == [], "an admission refusal neither opens nor closes (would be graded 0)"


def test_read_that_starts_then_fails_opens_once_and_closes_at_zero(monkeypatch):
    bus, conn = _FakeBus(), _Conn()
    pipe = _pipeline(bus, conn, InMemorySubstrateGraphStore(), max_attempts=1)
    _wire(pipe)
    # Started and failed between two polls: the failing poll is the first to see the start.
    _scripted(monkeypatch, [
        _state("waiting_resource", started=False),
        _state("failed", started=True, error="empty_learning"),
    ])

    _ticks(pipe, conn, 2)

    acts = _acts(bus)
    assert [e.event_kind for e in acts] == ["act_opened", "act_closed"]
    assert acts[0].observed_at == STARTED_AT
    assert acts[1].act.outcome.value == 0.0
    assert pipe._read_act_state() == ({}, set()), "terminal close forgets the seed"
