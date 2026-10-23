"""Body economy A5: want-family durable runs open an act at their first grant and close it at terminal.

Two lanes:
* pure (no database): the family map, the event shapes, the V1-only cost, horizons, the gates;
* real Postgres + the real in-process GPU pool (ORION_ADMISSION_TEST_DSN, as the other admission
  tests): open once and close once per terminal status, a resume and a re-grant never re-open, and
  the act's correlation_id is the turn_correlation_id the run's hold lease carries.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

from app import run_acts
from app.run_acts import FAMILIES, RunActEmitter, closed_event, opened_event, run_act_family
from orion.grammar.acts import stable_act_id
from orion.schemas.grammar import ActCostV1, GrammarEventV1

T0 = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)


def _state(workflow="curiosity.investigate", line="investigate", **extra):
    brief = {"prompt": "p", "session_id": "orion_curiosity", "line": line}
    brief.update(extra.pop("brief", {}))
    return {"run_id": "run-1", "correlation_id": "6f1e2a3b-0000-4000-8000-000000000001", "workflow": workflow,
            "brief": brief, "admission": extra.pop("admission", {"deadline_at": None}), **extra}


# --- pure lane -----------------------------------------------------------------------------------

def test_workflow_strings_match_the_runtime():
    from app.admission_runtime import DEFAULT_WORKFLOW, SELF_SENSE_WORKFLOW

    assert run_acts.CURIOSITY_WORKFLOW == DEFAULT_WORKFLOW
    assert run_acts.SELF_SENSE_WORKFLOW == SELF_SENSE_WORKFLOW


@pytest.mark.parametrize("workflow,line,actor,verb", [
    ("curiosity.investigate", "investigate", "durable.curiosity_investigate", "infer"),
    ("curiosity.investigate", None, "durable.curiosity_investigate", "infer"),
    ("curiosity.investigate", "self_inquiry", "durable.self_inquiry", "ask"),
    ("self_sense_eval", "self_sense_eval", "durable.self_sense_eval", "sense"),
    ("journal.compose", None, "durable.journal_compose", "make"),
])
def test_want_families_map_to_their_fixed_verb(workflow, line, actor, verb):
    family = run_act_family(_state(workflow, line))
    assert (family.actor, family.verb) == (actor, verb)


@pytest.mark.parametrize("state", [
    _state(brief={"urgent": {"subject": "cabinet_ac", "trigger": "cooling"}}),   # incident reflex
    _state("reading.turn", None), _state("reverie.visual", None), _state("dream.carry", None),
    _state("memory.episode_distill", None), _state("curiosity.compaction", None),
    _state("compactor.digest", None), _state("orion_day.letter", None),
    _state("self_study.reflect", "reflect"), _state("curiosity.investigate", "some_new_line"),
])
def test_non_want_runs_have_no_family(state):
    assert run_act_family(state) is None


def test_open_and_close_events_carry_identity_correlation_and_v1_cost_only():
    family = FAMILIES["curiosity_investigate"]
    state = _state()
    opened = opened_event(family, state, opened_at=T0, grant_entry_id="granted:L1:1")
    closed = closed_event(family, state, status="completed", opened_at=T0, grant_entry_id="granted:L1:1",
                          closed_at=T0 + timedelta(seconds=1234.5))
    for event in (opened, closed):
        GrammarEventV1.model_validate(event.model_dump(mode="json"))   # round-trips the bus shape
        assert event.provenance.source_service == "orion-durable-runs"
        assert event.correlation_id == state["correlation_id"]
        assert event.session_id == "orion_curiosity"
        assert event.trace_id == f"act:{stable_act_id('durable.curiosity_investigate', 'infer', 'run-1')}"
        assert event.act.expect.signal_id == "artifact_produced"
        assert event.act.expect.horizon_sec == 86400.0
        assert event.act.test_before == "gpu_pool_grant:granted:L1:1"
    assert opened.event_kind == "act_opened" and opened.observed_at == T0 and opened.act.outcome is None
    assert closed.event_kind == "act_closed"
    assert closed.act.outcome.value == 1.0 and closed.act.outcome.source == "producer"
    assert type(closed.act.cost) is ActCostV1                       # never an ActCostV2 on the bus
    assert closed.act.cost.model_dump() == {"seconds": 1234.5, "gpu_seconds": None, "tokens": None, "watts": None}


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_a_non_completed_terminal_closes_at_zero_with_the_reason(status):
    state = _state(last_error="gpu_pool_unavailable:deadline")
    event = closed_event(FAMILIES["self_inquiry"], state, status=status, opened_at=T0, grant_entry_id="g",
                         closed_at=T0 + timedelta(seconds=5))
    assert event.act.outcome.value == 0.0
    assert event.act.outcome.note == f"{status}: gpu_pool_unavailable:deadline"


def test_a_run_deadline_later_than_the_family_horizon_extends_it():
    deadline = T0 + timedelta(hours=18)
    state = _state("journal.compose", None, admission={"deadline_at": deadline.isoformat().replace("+00:00", "Z")})
    act = opened_event(FAMILIES["journal_compose"], state, opened_at=T0, grant_entry_id="g").act
    assert act.expect.horizon_sec == 18 * 3600 + run_acts.DEADLINE_HORIZON_MARGIN_SEC
    # A deadline sooner than the family horizon never shortens it.
    state = _state(admission={"deadline_at": (T0 + timedelta(minutes=20)).isoformat()})
    assert opened_event(FAMILIES["curiosity_investigate"], state, opened_at=T0, grant_entry_id="g").act.expect.horizon_sec == 86400.0


class _Store:
    def __init__(self, first=None):
        self.first = first

    async def first_event(self, run_id, event):
        assert event == "run.resource_granted"
        return self.first


def _emitter(store, *, enabled=True, sent=None):
    sent = [] if sent is None else sent

    async def publish(channel, kind, model, corr):
        sent.append((channel, kind, model, corr))
        return True
    return RunActEmitter(enabled=enabled, store=store, publish=publish, corr_for=lambda c: c,
                         now=lambda: T0 + timedelta(seconds=60)), sent


def test_emitter_opens_only_on_the_call_that_inserted_the_first_grant():
    store = _Store(("granted:L1:1", T0))
    emitter, sent = _emitter(store)
    inserted = {"entry_id": "granted:L1:1", "generated_at": T0.isoformat()}
    reread = {"entry_id": "granted:L1:1", "generated_at": (T0 + timedelta(seconds=9)).isoformat()}  # DO NOTHING
    regrant = {"entry_id": "granted:L1:2", "generated_at": (T0 + timedelta(seconds=99)).isoformat()}
    assert asyncio.run(emitter.on_grant(_state(), reread)) is False
    assert asyncio.run(emitter.on_grant(_state(), regrant)) is False
    assert asyncio.run(emitter.on_grant(_state(), inserted)) is True
    assert asyncio.run(emitter.on_grant(_state(), inserted)) is False   # same process: once
    [(channel, kind, event, corr)] = sent
    assert (channel, kind) == ("orion:grammar:event", "grammar.event.v1")
    assert event.event_kind == "act_opened" and corr == _state()["correlation_id"]


def test_emitter_closes_only_runs_that_were_granted_and_only_on_a_real_terminal():
    emitter, sent = _emitter(_Store(None))
    assert asyncio.run(emitter.on_terminal(_state(), "failed")) is False        # refused before any grant
    emitter, sent = _emitter(_Store(("granted:L1:1", T0)))
    assert asyncio.run(emitter.on_terminal(_state(), None)) is False            # paused: not terminal
    assert asyncio.run(emitter.on_terminal(_state(), "completed")) is True
    [(_, _, event, _)] = sent
    assert event.act.outcome.value == 1.0 and event.act.cost.seconds == 60.0


def test_emitter_off_publishes_nothing_and_never_raises():
    emitter, sent = _emitter(_Store(("g", T0)), enabled=False)
    assert asyncio.run(emitter.on_grant(_state(), {"entry_id": "g", "generated_at": T0.isoformat()})) is False
    assert asyncio.run(emitter.on_terminal(_state(), "completed")) is False
    assert sent == []

    class Broken:
        async def first_event(self, *a):
            raise RuntimeError("db down")
    emitter, sent = _emitter(Broken())
    assert asyncio.run(emitter.on_grant(_state(), {"entry_id": "g", "generated_at": T0.isoformat()})) is False
    assert asyncio.run(emitter.on_terminal(_state(), "completed")) is False


def test_flag_ships_on():
    from app.settings import Settings

    assert Settings.model_fields["durable_act_emit_enabled"].default is True
    example = (Path(__file__).resolve().parents[1] / ".env_example").read_text()
    assert "\nORION_DURABLE_ACT_EMIT_ENABLED=true\n" in example


# --- real Postgres + real pool lane --------------------------------------------------------------

from test_admission_runtime_postgres import DSN, Runner, request, runtime, until, with_database  # noqa: E402
from pool_fixture import CFG, InProcessPool  # noqa: E402
from orion.schemas.durable_run import CuriosityTurnResultV1  # noqa: E402
from app.graph import Deps  # noqa: E402

pg = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


def acts(rt, run_id=None):
    out = [m for k, m in rt.runner.events if k == "grammar.event.v1"]
    return [e for e in out if run_id is None or e.act.object_ref == f"durable_run:{run_id}"]


def _req(run_id, line="investigate", **admission):
    req = request(run_id, **admission)
    return req.model_copy(update={"brief": req.brief.model_copy(update={"line": line})})


@pg
@pytest.mark.parametrize("line,actor,verb", [("investigate", "durable.curiosity_investigate", "infer"),
                                              ("self_inquiry", "durable.self_inquiry", "ask")])
def test_completed_run_opens_once_closes_once_and_joins_its_hold_lease(line, actor, verb):
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        req = _req("act-done-"+line, line)
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        await rt.reconcile()     # a later tick re-reads nothing new
        opened, closed = acts(rt)
        assert [opened.event_kind, closed.event_kind] == ["act_opened", "act_closed"]
        assert (opened.act.actor, opened.act.verb) == (actor, verb)
        assert opened.act.act_id == closed.act.act_id == stable_act_id(actor, verb, req.run_id)
        assert closed.act.outcome.value == 1.0 and closed.act.cost.seconds >= 0
        first = await store.first_event(req.run_id, "run.resource_granted")
        assert opened.observed_at == first[1]                       # opened at the first grant
        # The act-ledger GPU cost fill joins grammar_acts.correlation_id to the hold's lease.
        [hold] = gpu.leases(holder=f"durable-runs:{req.run_id}")
        assert hold["turn_correlation_id"] == req.correlation_id == opened.correlation_id == closed.correlation_id
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_cancelled_while_running_closes_at_zero_once():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        block = asyncio.Event()
        rt = runtime(pool, saver, store, block, gpu=gpu)
        req = _req("act-cancel")
        await rt.submit(req)
        await rt.reconcile()
        await until(lambda: rt.runner.calls)
        await rt.control(req.run_id, "cancel")
        await rt._drive(await store.get_run(req.run_id))
        await rt._drive(await store.get_run(req.run_id))   # a terminal run is not driven again
        assert (await store.get_run(req.run_id))["terminal"] == "cancelled"
        assert [e.event_kind for e in acts(rt)] == ["act_opened", "act_closed"]
        closed = acts(rt)[1]
        assert closed.act.outcome.value == 0.0 and closed.act.outcome.note.startswith("cancelled")
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_failed_after_work_closes_at_zero_once(monkeypatch):
    real = Runner._curiosity_deps

    def failing(self):
        deps = real(self)

        async def turn(req):
            self.calls.append(req)
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, ok=False,
                                         error="hub_turn_failed")
        return Deps(turn, deps.read_turn_result, deps.publish_attention_row, deps.publish_journal)
    monkeypatch.setattr(Runner, "_curiosity_deps", failing)

    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, DURABLE_RUNS_RETRY_MAX_ATTEMPTS=1)
        req = _req("act-fail")
        await rt.submit(req)
        for _ in range(4):
            await rt._drive(await store.get_run(req.run_id))
            if (await store.get_run(req.run_id))["terminal"]:
                break
        assert (await store.get_run(req.run_id))["terminal"] == "failed"
        assert [e.event_kind for e in acts(rt)] == ["act_opened", "act_closed"]
        assert acts(rt)[1].act.outcome.value == 0.0
        assert acts(rt)[1].act.outcome.note.startswith("failed")
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_refused_before_any_grant_opens_and_closes_nothing():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        req = request("act-refused", preferred_lane="no-such-route", resource="llm.route.no-such-route")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "failed"
        assert acts(rt) == []
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_restart_mid_turn_does_not_reopen_and_closes_once():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        block = asyncio.Event()
        dead = runtime(pool, saver, store, block, gpu=gpu)
        req = _req("act-restart")
        await dead.submit(req)
        driver = asyncio.create_task(dead._drive(await store.get_run(req.run_id)))
        await until(lambda: dead.runner.calls)

        async def crashed(*args, **kwargs):
            return None
        dead._cancel_harness = crashed
        driver.cancel()
        await asyncio.gather(driver, return_exceptions=True)
        assert [e.event_kind for e in acts(dead)] == ["act_opened"]

        block.set()
        alive = runtime(pool, saver, store, gpu=gpu)
        await alive._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert [e.event_kind for e in acts(alive)] == ["act_closed"]   # the resume re-reads the grant, no re-open
        assert acts(alive)[0].act.act_id == acts(dead)[0].act.act_id
        await alive.close()
    asyncio.run(with_database(scenario))


@pg
def test_lost_hold_regrant_does_not_reopen():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        block = asyncio.Event()
        rt = runtime(pool, saver, store, block, gpu=gpu)
        req = _req("act-regrant")
        await rt.submit(req)
        driver = asyncio.create_task(rt._drive(await store.get_run(req.run_id)))
        await until(lambda: rt.runner.calls)
        await gpu.later(CFG.defaults.hold_lease_ttl_sec + 1)
        await asyncio.wait_for(driver, 5)
        block.set()
        await gpu.later(60)
        await rt.on_pool_event({"holder": f"durable-runs:{req.run_id}", "event": "granted"})
        await rt._drive(await store.get_run(req.run_id))
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        grants = [e for e in await store.history(req.run_id) if e["event"] == "run.resource_granted"]
        assert len(grants) == 2
        assert [e.event_kind for e in acts(rt)] == ["act_opened", "act_closed"]
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_flag_off_and_non_want_runs_emit_nothing():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        off = runtime(pool, saver, store, gpu=gpu, ORION_DURABLE_ACT_EMIT_ENABLED=False)
        req = _req("act-off")
        await off.submit(req)
        await off._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert acts(off) == []
        await off.close()
        on = runtime(pool, saver, store, gpu=gpu)
        reflect = request("act-reflect", workflow="self_study.reflect")
        await on.submit(reflect)
        await on._drive(await store.get_run(reflect.run_id))
        assert acts(on) == []
        await on.close()
    asyncio.run(with_database(scenario))
