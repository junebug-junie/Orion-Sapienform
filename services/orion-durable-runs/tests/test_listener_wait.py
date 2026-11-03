"""A step whose RPC reached no listener waits out the deploy and replays, no attempt spent.

Live 2026-10-31: 16 reading runs failed with ``RpcNoListener`` in four bursts (04:28, 07:58,
22:29, 23:20 UTC), each a service redeploy. Only the publish-time case (``RpcPublishedToNobody``:
no worker received the request) is replayed; a request delivered and then orphaned
("while awaiting its reply") keeps the normal attempt path, since its listener may have done work.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from test_admitted_graph import CFG, World, initial
from app.admitted_graph import (
    LISTENER_WAIT_HOLD_KEY, WAITING, AdmissionDeps, build_admitted_graph, listener_wait, published_to_nobody,
)
from app.graph import Deps
from app.reading_graph import build_reading_graph
from app.self_sense_graph import SelfSenseAskFailed
from orion.core.bus.async_service import RpcNoListener, RpcPublishedToNobody
from orion.schemas.reading_turn import ReadingTurnResultV1

NOBODY = "no listener on orion:reading:turn:request: request published to nobody"
AWAITING = "no listener on orion:reading:turn:request while awaiting its reply"


class ParkingWorld(World):
    """World whose register honours the listener park the way AdmissionRuntime.register does."""

    async def register(self, state):
        hold = state.get("hold") or {}
        if hold.get(LISTENER_WAIT_HOLD_KEY) and not hold.get("request_id") \
                and self.now < datetime.fromisoformat(hold["retry_at"]):
            return {"status": "waiting_resource", "hold": hold, "hold_seq": int(state.get("hold_seq") or 0)}
        if hold.get(LISTENER_WAIT_HOLD_KEY):
            state = {**state, "hold": None}
        return await super().register(state)

    async def lease(self, state):
        # AdmissionRuntime.lease: a hold with no lease_id at the pool is simply still waiting.
        if not (state.get("hold") or {}).get("lease_id"):
            return WAITING, {"status": "waiting_resource", "lease": None}
        return await super().lease(state)


def deps(world, **kw):
    return AdmissionDeps(world.register, world.lease, world.execute, world.release, world.event,
                         now=lambda: world.now, **kw)


def reading_state():
    value = initial()
    value["brief"].update(seed_id="reading:seed", stage=1, timeout_sec=600)
    return value


def reading_turn(failures):
    """A Hub reading turn that raises the queued exceptions first, then answers."""
    calls = []

    async def turn(request):
        calls.append(request)
        if failures:
            raise failures.pop(0)
        return ReadingTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id, ok=True,
                                   text="Grounded result", source_fetches=[])

    return turn, calls


# --- classification -----------------------------------------------------------------------------
def test_only_the_publish_time_no_listener_is_replay_safe():
    assert published_to_nobody(RpcPublishedToNobody(NOBODY)) is not None
    # Delivered, then orphaned: the listener may have started work.
    assert published_to_nobody(RpcNoListener(AWAITING)) is None
    assert published_to_nobody(TimeoutError("slow")) is None
    assert published_to_nobody(ConnectionError("redis")) is None


def test_a_wrapped_publish_time_no_listener_is_still_found():
    # self-sense wraps the transport error: SelfSenseAskFailed(...) from exc.
    try:
        try:
            raise RpcPublishedToNobody(NOBODY)
        except RpcPublishedToNobody as inner:
            raise SelfSenseAskFailed("RpcPublishedToNobody") from inner
    except SelfSenseAskFailed as outer:
        assert published_to_nobody(outer) is not None


# --- reading: the incident ----------------------------------------------------------------------
def test_reading_turn_published_to_nobody_waits_out_the_deploy_and_completes():
    async def run():
        world, saver = ParkingWorld(), InMemorySaver()
        turn, calls = reading_turn([RpcPublishedToNobody(NOBODY)])
        graph = build_reading_graph(turn, deps(world), saver)
        await graph.ainvoke(reading_state(), CFG)
        world.grant()
        parked = await graph.ainvoke(Command(resume=True), CFG)

        assert parked["status"] == "waiting_resource", "not failed: the deploy is waited out"
        assert parked.get("attempt", 0) == 0, "no attempt spent"
        assert parked["listener_waits"] == 1
        assert parked["hold"][LISTENER_WAIT_HOLD_KEY] == 1
        assert parked["hold"]["retry_at"] == (world.now + timedelta(seconds=60)).isoformat()
        assert world.releases == ["listener_absent"], "the hold is handed back during the wait"
        assert ("run.listener_wait", {"waits": 1, "retry_at": parked["hold"]["retry_at"], "error": NOBODY}) \
            in world.events
        snap = await graph.aget_state(CFG)
        assert snap.next == ("resource_wait",)

        # Inside the wait: no new pool request, no turn.
        asked = list(world.requests)
        world.now += timedelta(seconds=59)
        await graph.ainvoke(Command(resume=True), CFG)
        assert world.requests == asked and len(calls) == 1

        # Wait over: a fresh hold, the turn replays and the reading completes.
        world.now += timedelta(seconds=2)
        world.pool_status = "queued"
        await graph.ainvoke(Command(resume=True), CFG)
        world.grant()
        done = await graph.ainvoke(Command(resume=True), CFG)
        assert done["status"] == "completed" and len(calls) == 2
        assert done["listener_waits"] == 1

    asyncio.run(run())


def test_reading_listener_waits_are_bounded_then_fail_as_today():
    async def run():
        world = World()
        world.grant()
        turn, calls = reading_turn([RpcPublishedToNobody(NOBODY)])
        state = {**reading_state(), "listener_waits": 2}
        result = await build_reading_graph(turn, deps(world, listener_wait_max_tries=2), InMemorySaver()) \
            .ainvoke(state, CFG)
        assert result["status"] == "failed"
        assert result["last_error"].startswith("turn_exception:RpcPublishedToNobody: no listener")
        assert world.releases == ["attempt_failed"] and len(calls) == 1
        assert not any(name == "run.listener_wait" for name, _ in world.events)

    asyncio.run(run())


def test_reading_turn_orphaned_while_awaiting_its_reply_is_not_replayed():
    async def run():
        world = World()
        world.grant()
        turn, calls = reading_turn([RpcNoListener(AWAITING)])
        result = await build_reading_graph(turn, deps(world), InMemorySaver()).ainvoke(reading_state(), CFG)
        assert result["status"] == "failed" and len(calls) == 1
        assert result["last_error"].startswith("turn_exception:RpcNoListener")
        assert not result.get("listener_waits")

    asyncio.run(run())


def test_listener_wait_off_keeps_todays_behaviour():
    async def run():
        world = World()
        world.grant()
        turn, _ = reading_turn([RpcPublishedToNobody(NOBODY)])
        result = await build_reading_graph(turn, deps(world, listener_wait_max_tries=0), InMemorySaver()) \
            .ainvoke(reading_state(), CFG)
        assert result["status"] == "failed"

    asyncio.run(run())


def test_a_wait_past_the_deadline_is_not_taken():
    async def run():
        world = World()
        state = {**reading_state(), "admission": {"resource": "llm.route.agent",
                                                  "deadline_at": (world.now + timedelta(seconds=300)).isoformat()}}
        update = await listener_wait(deps(world), state, RpcPublishedToNobody(NOBODY),
                                     waiting={"status": "waiting_resource"})
        assert update is None, "60 s wait + a 600 s attempt cannot finish inside 300 s"

    asyncio.run(run())


def test_backoff_doubles_and_caps():
    async def run():
        world = World()
        gaps = []
        for waits in range(5):
            update = await listener_wait(deps(world), {**reading_state(), "listener_waits": waits},
                                         RpcPublishedToNobody(NOBODY), waiting={"status": "waiting_resource"})
            gaps.append((datetime.fromisoformat(update["hold"]["retry_at"])
                         - world.now).total_seconds())
        assert gaps == [60.0, 120.0, 240.0, 300.0, 300.0]

    asyncio.run(run())


# --- curiosity (shared failed_attempt path) ------------------------------------------------------
def test_curiosity_turn_published_to_nobody_waits_without_spending_an_attempt():
    async def scenario():
        world, saver = World(), InMemorySaver()
        failures = [RpcPublishedToNobody("no listener on orion:curiosity:turn:request: request published to nobody")]

        async def turn(req):
            if failures:
                raise failures.pop(0)
            return await world.turn(req)

        def graph():
            return build_admitted_graph(Deps(turn, world.read, world.row, world.journal),
                                        AdmissionDeps(world.register, world.lease, world.execute, world.release,
                                                      world.event, now=lambda: world.now, max_attempts=2,
                                                      guard=world.guard, keep_for_outreach=world.keep), saver)

        state = {**initial(), "brief": {**initial()["brief"], "timeout_sec": 600}}
        await graph().ainvoke(state, CFG)
        world.grant()
        waiting = await graph().ainvoke(Command(resume=True), CFG)
        assert waiting["status"] == "retrying" and waiting["attempt"] == 0, "max_attempts=2 is untouched"
        assert waiting["listener_waits"] == 1
        assert waiting["retry_at"] == (world.now + timedelta(seconds=60)).isoformat()
        assert (await graph().aget_state(CFG)).next == ("retry_wait",)

        world.now += timedelta(seconds=61)
        await graph().ainvoke(Command(resume=True), CFG)
        world.grant()
        done = await graph().ainvoke(Command(resume=True), CFG)
        assert done["status"] == "completed" and world.calls == [1]

    asyncio.run(scenario())


# --- runner: the admitted Hub turn RPC raises, the legacy path keeps ok=False ----------------------
def _runner(exc, admitted):
    from test_runner_turn_contract import runner_for, turn_request

    request = turn_request(admitted=admitted)
    runner, bus = runner_for(request)
    bus.rpc_request = AsyncMock(side_effect=exc)
    return runner, request


def test_admitted_turn_published_to_nobody_raises_for_listener_wait():
    runner, request = _runner(RpcPublishedToNobody(NOBODY), admitted=True)
    with pytest.raises(RpcPublishedToNobody):
        asyncio.run(runner._run_turn(request))


@pytest.mark.parametrize("exc,admitted", [(RpcPublishedToNobody(NOBODY), False), (RpcNoListener(AWAITING), True)])
def test_other_no_listener_turns_still_come_back_as_failed_results(exc, admitted):
    runner, request = _runner(exc, admitted)
    result = asyncio.run(runner._run_turn(request))
    assert not result.ok and result.error == "rpc:RpcNoListener"


# --- every other graph whose work step makes a require_listener RPC ------------------------------
def _parked_after_one_unheard_call(build, initial_state):
    """Drive one granted step that raises RpcPublishedToNobody; the checkpointed state after it."""
    async def run():
        saver = InMemorySaver()
        result = await build(saver).ainvoke(initial_state, CFG)
        return result, (await build(saver).aget_state(CFG))
    return asyncio.run(run())


def test_journal_compose_unheard_compose_waits_without_spending_an_attempt():
    from test_journal_compose_graph import Journal, graph, initial as journal_initial

    world = World()
    world.grant()
    journal = Journal(world)
    real = journal.compose

    async def compose(*args, **kw):
        if not journal.composes:
            journal.composes.append("unheard")
            raise RpcPublishedToNobody("no listener on orion:cortex:request: request published to nobody")
        return await real(*args, **kw)

    journal.compose = compose
    result, snap = _parked_after_one_unheard_call(lambda saver: graph(world, saver, journal), journal_initial())
    assert result["status"] == "retrying" and result.get("attempt", 0) == 0
    assert result["listener_waits"] == 1 and snap.next == ("retry_wait",)
    assert result["retry_at"] == (world.now + timedelta(seconds=60)).isoformat()
    assert world.releases == ["listener_absent"]


def test_compactor_digest_unheard_call_waits_without_spending_an_attempt():
    from test_compactor_digest_graph import Orch, graph, initial as digest_initial

    world = ParkingWorld()
    world.grant()
    orch = Orch(world)
    real = orch.rpc
    unheard = [RpcPublishedToNobody("no listener on orion:cortex:request: request published to nobody")]

    async def rpc(payload, **kw):
        if unheard:
            raise unheard.pop(0)
        return await real(payload, **kw)

    orch.rpc = rpc
    result, snap = _parked_after_one_unheard_call(lambda saver: graph(world, saver, orch), digest_initial())
    assert result["status"] == "waiting_resource" and result.get("attempt", 0) == 0
    assert result["listener_waits"] == 1 and snap.next == ("resource_wait",)
    assert result["hold"][LISTENER_WAIT_HOLD_KEY] == 1 and world.releases == ["listener_absent"]
    assert orch.calls == [], "no digest call was made during the wait"


def test_episode_distill_unheard_call_waits_without_spending_an_attempt():
    from test_episode_distill_graph import Distiller, graph, initial as distill_initial

    world = World()
    world.grant()
    d = Distiller(world, [])

    async def call_llm(prompt, **kw):
        raise RpcPublishedToNobody("no listener on orion:exec:request:LLMGatewayService: request published to nobody")

    d.call_llm = call_llm
    result, snap = _parked_after_one_unheard_call(lambda saver: graph(world, saver, d), distill_initial())
    assert result["status"] == "retrying" and result.get("attempt", 0) == 0
    assert result["listener_waits"] == 1 and snap.next == ("retry_wait",)


def test_orion_day_unheard_note_waits_then_completes_without_spending_an_attempt():
    from test_orion_day_graph import World as DayWorld, drive
    from orion.schemas.orion_day import ORION_DAY_NOTE_VERB

    async def scenario():
        world, saver = DayWorld(max_attempts=1), InMemorySaver()
        world.replies[ORION_DAY_NOTE_VERB] = [
            RpcPublishedToNobody("no listener on orion:cortex:request: request published to nobody")]
        snap = await drive(world.graph(saver), world)
        # max_attempts=1: had the unheard call spent an attempt, the run would have failed.
        assert snap.values["status"] == "completed"
        assert snap.values["listener_waits"] == 1
        assert "listener_absent" in world.releases

    asyncio.run(scenario())


def test_settings_default_the_wait_on():
    from app.settings import Settings

    s = Settings(_env_file=None, POSTGRES_URI="postgresql://unused", ORION_BUS_ENABLED=False)
    assert (s.listener_wait_max_tries, s.listener_wait_base_sec, s.listener_wait_max_sec) == (6, 60.0, 300.0)
