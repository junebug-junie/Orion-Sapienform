"""Orion Prime 2A resident loop (app/prime_loop.py) with a fake bus, model and market. No network."""

from __future__ import annotations

import asyncio
import json
import urllib.error
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.prime_loop import STATE_KEY, JuniperTurns, PrimeRuntime
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.introspect.transport import QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL, RESULT_KIND
from orion.prime import acts as pa
from orion.prime.state import PrimeStateV1
from orion.schemas.introspect import IntrospectResultV1

T0 = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)

BODY = {"ok": True, "age_sec": 30, "strain": {"value": 0.3, "constraint": "lane:agent", "why": "lane:agent at 0.3"},
        "prices": {"heat": {"value": 0.05, "why": "cabinet 24 C"}}}
TEMPORAL = {"day_phase": "afternoon", "active_arc": {"arc_id": "arc1", "subject_label": "heat", "kind": "attention"},
            "arcs_today": [{"arc_id": "arc1", "kind": "attention", "subject_label": "heat", "status": "open"}]}


def curiosity(runs=(), pending=None):
    return {"runs": list(runs), "schedule": {"lines": {"investigate": {"runs_today": 1, "daily_cap": 7}},
                                             "pending_focus": {"available": True, "focus": pending}}}


class Redis:
    def __init__(self):
        self.kv = {}

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value):
        self.kv[key] = value


class Codec:
    def decode(self, data):
        return SimpleNamespace(ok=True, envelope=BaseEnvelope.model_validate(data))


class Bus:
    def __init__(self, focus_reply=None):
        self.redis = Redis()
        self.codec = Codec()
        self.published = []
        self.rpcs = []
        self.focus_reply = focus_reply or {"ok": True}

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
        self.rpcs.append((channel, envelope))
        if self.focus_reply["ok"]:
            result = IntrospectResultV1(ok=True, operation="queue_curiosity_focus", as_of=T0, total_available=0)
        else:
            result = IntrospectResultV1(ok=False, operation="queue_curiosity_focus", as_of=T0,
                                        error=self.focus_reply["error"])
        reply = BaseEnvelope(kind=RESULT_KIND, correlation_id=envelope.correlation_id, source=ServiceRef(name="orion-hub"),
                             payload=result.model_dump(mode="json"))
        return {"data": reply.model_dump(mode="json")}

    def acts(self):
        return [(e.payload["event_kind"], e.payload["act"]) for c, e in self.published if c == "orion:grammar:event"]


class LLM:
    def __init__(self, answer, on_call=None):
        self.answer = answer
        self.calls = []
        self.on_call = on_call

    def request_json(self, **kw):
        self.calls.append(kw)
        if self.on_call:
            self.on_call()
        return self.answer, None, {}


SETTINGS = SimpleNamespace(
    SERVICE_NAME="orion-mind", SERVICE_VERSION="t", NODE_NAME="test", PRIME_TICK_SEC=15, PRIME_REFRACTORY_SEC=120,
    PRIME_SETTLE_INTERVAL_SEC=300, PRIME_LLM_ROUTE="prime", PRIME_BID_LANE="chat", PRIME_MAX_TOKENS=700,
    PRIME_LLM_TIMEOUT_SEC=60, PRIME_HOLDBACK_FRACTION=0.0, PRIME_DURABLE_RUNS_URL="http://dr:8121",
    PRIME_HUB_URL="http://hub:8080",
)


def runtime(*, answer, bus=None, outcome="funded", runs=(), pending=None, holdback=0.0, turns=None, on_call=None,
            now=T0, extra_pages=None):
    clock = {"now": now}
    pages = {"http://dr:8121/physiology/state": BODY, "http://dr:8121/temporal-self/frame": TEMPORAL,
             **(extra_pages or {})}

    def get_json(url):
        if url.startswith("http://hub:8080/curiosity/api/runs"):
            return curiosity(runs, pending)
        if url not in pages:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        return pages[url]

    bids, released = [], []

    async def bid(want):
        bids.append(want)
        return None if outcome is None else SimpleNamespace(outcome=outcome, reason="net_positive", binding_constraint=None)

    async def release(want_id):
        released.append(want_id)

    rt = PrimeRuntime(settings=SimpleNamespace(**{**vars(SETTINGS), "PRIME_HOLDBACK_FRACTION": holdback}),
                      bus=bus or Bus(), llm=LLM(answer, on_call), get_json=get_json, bid=bid, release=release,
                      clock=lambda: clock["now"], turns=turns or JuniperTurns())
    rt.bids, rt.released, rt.clock_box = bids, released, clock
    return rt


MOVE = {"reading": "the cabinet is warming", "concerns": {"add": [{"text": "afternoon heat", "arc_id": "arc1"}]},
        "move": {"kind": "curiosity_focus", "question": "What heats the cabinet after 14:00?", "why": "afternoon heat"}}


def test_a_funded_thought_opens_and_closes_its_act_and_a_treated_question_is_queued():
    rt = runtime(answer=MOVE)
    asyncio.run(rt.tick())
    [want] = rt.bids
    assert want.family == "prime" and want.tier == 1 and want.verb == "infer" and want.lanes == ["chat"]
    # Its own cost ceiling, never the market's one-whole-window default for an unmeasured family.
    assert want.cost.gpu_slot_s_by_lane == {"chat": 60.0}
    assert rt.released == [want.want_id]
    [call] = rt.llm.calls
    assert call["route"] == "prime" and call["thinking"] is False and "first wake" in call["user_prompt"]
    kinds = [(k, a["verb"]) for k, a in rt.bus.acts()]
    # The concern the thought added opens its own act (verb predict) once the thought has closed.
    assert kinds == [("act_opened", "infer"), ("act_opened", "ask"), ("act_closed", "infer"), ("act_opened", "predict")]
    closed = rt.bus.acts()[2][1]
    assert closed["outcome"]["value"] == 1.0 and "move curiosity_focus" in closed["outcome"]["note"]
    [(channel, env)] = rt.bus.rpcs
    assert channel == QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL
    assert env.payload["binding"]["invocation_context"] == "prime" and env.payload["args"]["origin"] == "self"
    [nom] = rt.state.open_nominations
    assert nom.arm == "treated" and nom.queued is True
    assert nom.focus_id == pa.prime_focus_id(question=MOVE["move"]["question"], why=MOVE["move"]["why"],
                                             thought_id=nom.thought_id)
    assert [c.text for c in rt.state.concerns] == ["afternoon heat"] and rt.state.pending_changes == []
    # Persisted: a restart resumes the same concerns and the open question.
    saved = PrimeStateV1.model_validate_json(rt.bus.redis.kv[STATE_KEY])
    assert saved.open_nominations[0].focus_id == nom.focus_id


def test_a_held_back_question_is_an_act_but_never_sent():
    rt = runtime(answer=MOVE, holdback=1.0)
    asyncio.run(rt.tick())
    assert rt.bus.rpcs == []
    ask = [a for k, a in rt.bus.acts() if a["verb"] == "ask"][0]
    assert ask["test_before"] == "holdback:control"
    assert rt.state.open_nominations[0].arm == "control" and rt.state.open_nominations[0].queued is None


@pytest.mark.parametrize("outcome", ["deferred", None])
def test_no_thought_unless_the_bid_is_funded(outcome):
    rt = runtime(answer=MOVE, outcome=outcome)
    asyncio.run(rt.tick())
    assert rt.llm.calls == [] and rt.bus.acts() == [] and rt.state.pending_changes  # still hungry


def test_no_bid_while_a_juniper_turn_runs():
    turns = JuniperTurns()
    rt = runtime(answer=MOVE, turns=turns)
    with turns.turn():
        asyncio.run(rt.tick())
    assert rt.bids == [] and rt.llm.calls == []


def test_a_thought_overlapping_a_juniper_turn_is_preempted_and_discarded():
    turns = JuniperTurns()

    def juniper_arrives():
        with turns.turn():
            pass

    rt = runtime(answer=MOVE, turns=turns, on_call=juniper_arrives)
    asyncio.run(rt.tick())
    closed = [a for k, a in rt.bus.acts() if k == "act_closed"]
    assert closed[0]["outcome"]["value"] == 0.0 and "preempted" in closed[0]["outcome"]["note"]
    assert rt.bus.rpcs == [] and rt.state.concerns == [] and rt.state.pending_changes  # nothing applied


def test_an_unparseable_answer_is_a_failed_thought_and_the_refractory_holds():
    rt = runtime(answer=None)
    asyncio.run(rt.tick())
    closed = [a for k, a in rt.bus.acts() if k == "act_closed"]
    assert closed[0]["outcome"]["value"] == 0.0 and "failed" in closed[0]["outcome"]["note"]
    rt.clock_box["now"] = T0 + timedelta(seconds=30)
    asyncio.run(rt.tick())
    assert len(rt.bids) == 1  # inside the refractory interval: no second bid


def test_a_pending_focus_refuses_the_move_without_waking_prime_again():
    rt = runtime(answer=MOVE, pending={"question": "from chat", "origin": "self"})
    asyncio.run(rt.tick())
    assert rt.bus.rpcs == [] and rt.state.open_nominations == []
    assert rt.state.pending_changes == [] and "not sent" in rt.state.recent_outcomes[-1]


def test_hub_refusal_is_recorded_and_the_question_closes_zero_on_settle():
    bus = Bus(focus_reply={"ok": False, "error": "a focus is already waiting"})
    rt = runtime(answer=MOVE, bus=bus)
    asyncio.run(rt.tick())
    nom = rt.state.open_nominations[0]
    assert nom.queued is False and "already waiting" in nom.queue_error
    rt.clock_box["now"] = T0 + timedelta(minutes=10)
    asyncio.run(rt.tick())
    closes = [a for k, a in rt.bus.acts() if k == "act_closed" and a["verb"] == "ask"]
    assert closes[0]["act_id"] == nom.act_id and closes[0]["outcome"]["value"] == 0.0
    assert any("not queued" in o for o in rt.state.recent_outcomes)
    # The refusal is news, so the same tick thought again (and here asked again, into a new act).
    assert len(rt.llm.calls) == 2 and "not queued" in rt.llm.calls[1]["user_prompt"]
    assert [n.act_id for n in rt.state.open_nominations] != [nom.act_id]


def test_a_treated_question_settles_when_its_run_ends_and_prime_hears_about_it():
    rt = runtime(answer=MOVE)
    asyncio.run(rt.tick())
    nom = rt.state.open_nominations[0]
    run = {"run_id": "r1", "line": "investigate", "started_at_iso": (T0 + timedelta(hours=1)).isoformat(),
           "outcome_kind": "finished", "focus": {"focus_id": nom.focus_id, "disposition": "explored"}}
    rt2 = runtime(answer={"reading": "ok"}, bus=rt.bus, runs=[run], now=T0 + timedelta(hours=3))
    asyncio.run(rt2.load())
    asyncio.run(rt2.tick())
    ask_close = [a for k, a in rt2.bus.acts() if k == "act_closed" and a["verb"] == "ask"][0]
    assert ask_close["act_id"] == nom.act_id and ask_close["outcome"]["value"] == 1.0
    assert "disposition=explored" in ask_close["outcome"]["note"]
    assert rt2.state.open_nominations == []
    assert any("explored" in o for o in rt2.state.recent_outcomes)
    # News of its own question is a change: the same tick bid for a thought about it.
    assert rt2.bids and "explored" in rt2.llm.calls[0]["user_prompt"]


def test_a_control_outcome_is_never_described_to_prime_as_curiositys_answer():
    rt = runtime(answer=MOVE, holdback=1.0)
    asyncio.run(rt.tick())
    run = {"run_id": "r1", "line": "investigate", "started_at_iso": (T0 + timedelta(hours=1)).isoformat(),
           "outcome_kind": "finished"}
    rt2 = runtime(answer={"reading": "ok"}, bus=rt.bus, runs=[run], now=T0 + timedelta(hours=3))
    asyncio.run(rt2.load())
    asyncio.run(rt2.tick())
    assert any("held back as a control" in o for o in rt2.state.recent_outcomes)
    note = [a for k, a in rt2.bus.acts() if k == "act_closed" and a["verb"] == "ask"][0]["outcome"]["note"]
    assert note.startswith("control run r1")


def test_mind_run_marks_only_juniper_turns():
    """main.mind_run wraps run_mind in JUNIPER_TURNS only for a positively marked Juniper turn."""
    import app.main as main

    seen = []
    orig = main.run_mind
    main.run_mind = lambda body, **kw: seen.append(main.JUNIPER_TURNS.active()) or "ok"
    try:
        for origin, expected in (("juniper", True), ("orion", False), (None, False)):
            body = SimpleNamespace(trigger="user_turn", utterance_origin=origin)
            assert asyncio.run(main.mind_run(body)) == "ok"
            assert seen[-1] is expected
    finally:
        main.run_mind = orig


def test_lifespan_starts_prime_on_its_own_bus_without_blocking_and_stops_it(monkeypatch):
    """A bus that cannot connect at boot delays Prime (retry in the task); it never blocks the
    lifespan or run_mind. Off: no bus, no task."""
    import app.main as main
    from fastapi import FastAPI

    made = []

    class FailingBus:
        def __init__(self, **kw):
            made.append(self)
            self.closed = False

        async def connect(self):
            raise ConnectionError("bus down")

        async def close(self):
            self.closed = True

    monkeypatch.setattr(main, "OrionBusAsync", FailingBus)
    monkeypatch.setattr(main, "build_heartbeat_chassis", lambda: SimpleNamespace(
        start_background=_anoop, stop=_anoop))
    monkeypatch.setattr(main.settings, "RPC_HEALTH_PUBLISH_ENABLED", False)
    monkeypatch.setattr(main.settings, "ORION_BUS_ENABLED", True)
    monkeypatch.setattr(main.settings, "PRIME_ENABLED", True)

    async def go():
        async with main.lifespan(FastAPI()):
            assert main.prime_runtime is not None
            await asyncio.sleep(0)
        assert main.prime_runtime is None and made[0].closed

    asyncio.run(go())
    monkeypatch.setattr(main.settings, "PRIME_ENABLED", False)
    made.clear()

    async def off():
        async with main.lifespan(FastAPI()):
            assert main.prime_runtime is None and made == []

    asyncio.run(off())


async def _anoop(*a, **k):
    return None


def test_a_focus_replaced_in_hub_settles_after_two_reads_and_frees_prime():
    rt = runtime(answer=MOVE)
    asyncio.run(rt.tick())
    nom = rt.state.open_nominations[0]
    # A chat focus replaced Prime's: the slot holds another id and no run took Prime's.
    rt2 = runtime(answer={"reading": "ok"}, bus=rt.bus, pending={"focus_id": "chat-focus", "question": "from chat"},
                  now=T0 + timedelta(minutes=10))
    asyncio.run(rt2.load())
    asyncio.run(rt2.tick())
    assert rt2.state.open_nominations[0].gone_reads == 1
    rt2.clock_box["now"] = T0 + timedelta(minutes=16)
    asyncio.run(rt2.tick())
    closes = [a for k, a in rt2.bus.acts() if k == "act_closed" and a["verb"] == "ask"]
    assert closes[0]["act_id"] == nom.act_id and closes[0]["outcome"]["value"] == 0.0
    assert "left Hub's slot" in closes[0]["outcome"]["note"] and rt2.state.open_nominations == []


def test_a_crashing_thought_closes_its_act_and_still_waits_out_the_refractory(monkeypatch):
    rt = runtime(answer=MOVE)
    rt.llm.request_json = lambda **kw: (_ for _ in ()).throw(AssertionError("unreachable"))
    # The loop module's own globals (conftest re-imports `app`, so sys.modules may hold another copy).
    monkeypatch.setitem(PrimeRuntime._think.__globals__, "build_frame",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad payload")))
    asyncio.run(rt.tick())
    rt.clock_box["now"] = T0 + timedelta(seconds=30)
    asyncio.run(rt.tick())
    closed = [a for k, a in rt.bus.acts() if k == "act_closed"]
    assert len(closed) == 1 and closed[0]["outcome"]["value"] == 0.0 and "bad payload" in closed[0]["outcome"]["note"]
    assert len(rt.bids) == 1   # no hot loop: the second tick is inside the refractory interval


def test_the_market_prices_prime_at_its_ceiling_not_a_whole_window():
    """The live failure (10-30): no history -> one 3600 s window charged -> deferred forever."""
    from orion.physiology.appetite import estimate_cost
    from orion.physiology.families import load_families

    from pathlib import Path

    rt = runtime(answer=MOVE)
    repo = Path(__file__).resolve().parents[3]
    families, err = load_families(repo / "config/physiology/families.yaml")
    assert err is None
    cfg = families.families["prime"]
    cost = estimate_cost(rt._cost_ceiling().model_dump(), cfg, SimpleNamespace(cost_q={}, lane_slot_s_q={},
                                                                                total_slot_s_q=None))
    assert cost.one_of_slot_s == 60.0 and "gpu_slot_s_by_lane" not in cost.unpriced


# --- concerns as graded acts (Prime step 1) --------------------------------------------------------

LIVE_STATE = (
    '{"schema_version":"prime.state.v1","concerns":[{"concern_id":"777925ef197d44b1","text":"agent lane congestion",'
    '"arc_id":null,"opened_at":"2026-10-30T11:04:46Z","last_evidence_at":"2026-10-30T11:39:35Z"}],'
    '"open_nominations":[],"pending_changes":[],"recent_outcomes":[],"last_thought_at":"2026-10-30T11:00:00Z",'
    '"last_reading":"","thoughts_total":21,"seen_active_arc_id":"arc1","seen_binding":[]}'
)


def _concern_acts(bus):
    return [(k, a) for k, a in bus.acts() if a["verb"] == "predict"]


def test_a_concern_held_before_concern_acts_is_adopted_once_on_the_first_tick():
    bus = Bus()
    bus.redis.kv[STATE_KEY] = LIVE_STATE
    rt = runtime(answer={"reading": "ok"}, bus=bus)
    asyncio.run(rt.load())
    asyncio.run(rt.tick())
    [(kind, act)] = _concern_acts(bus)
    assert kind == "act_opened" and act["act_id"] == pa.concern_act("777925ef197d44b1").act_id
    assert act["actor"] == "prime" and act["goal_ref"] is None and act["object_ref"] == "prime_concern:777925ef197d44b1"
    assert rt.state.concerns[0].act_opened_at == T0
    rt.clock_box["now"] = T0 + timedelta(seconds=15)
    asyncio.run(rt.tick())
    assert len(_concern_acts(bus)) == 1  # never re-opened


def test_dropping_a_concern_closes_its_act_once_even_across_a_restart():
    bus = Bus()
    bus.redis.kv[STATE_KEY] = LIVE_STATE
    rt = runtime(answer={"reading": "ok", "concerns": {"drop": [{"concern_id": "777925ef197d44b1", "why": "moved on"}]}},
                 bus=bus)
    asyncio.run(rt.load())
    rt.state = rt.state.model_copy(update={"pending_changes": ["attention moved"]})
    asyncio.run(rt.tick())
    closes = [a for k, a in _concern_acts(bus) if k == "act_closed"]
    assert len(closes) == 1 and closes[0]["outcome"]["value"] == 0.0 and "dropped unresolved" in closes[0]["outcome"]["note"]
    [rec] = PrimeStateV1.model_validate_json(bus.redis.kv[STATE_KEY]).concern_record
    assert rec.published and rec.end == "dropped"
    # A restart reloads the saved state: nothing is re-sent.
    rt2 = runtime(answer={"reading": "ok"}, bus=bus, now=T0 + timedelta(minutes=1))
    asyncio.run(rt2.load())
    asyncio.run(rt2.tick())
    assert len([a for k, a in _concern_acts(bus) if k == "act_closed"]) == 1


def test_a_failed_close_publish_stays_in_the_outbox_and_is_retried_with_the_same_act_id():
    bus = Bus()
    bus.redis.kv[STATE_KEY] = LIVE_STATE
    rt = runtime(answer={"reading": "ok", "concerns": {"drop": [{"concern_id": "777925ef197d44b1", "why": "x"}]}},
                 bus=bus)
    asyncio.run(rt.load())
    rt.state = rt.state.model_copy(update={"pending_changes": ["attention moved"]})
    real = bus.publish

    async def flaky(channel, envelope):
        if envelope.payload.get("event_kind") == "act_closed" and envelope.payload["act"]["verb"] == "predict":
            raise ConnectionError("bus blip")
        await real(channel, envelope)

    bus.publish = flaky
    asyncio.run(rt.tick())
    assert not rt.state.concern_record[0].published
    bus.publish = real
    rt.clock_box["now"] = T0 + timedelta(seconds=15)
    asyncio.run(rt.tick())
    [close] = [a for k, a in _concern_acts(bus) if k == "act_closed"]
    assert close["act_id"] == pa.concern_act("777925ef197d44b1").act_id and rt.state.concern_record[0].published


def test_a_treated_question_that_produced_settles_its_concern_at_one():
    move = {**MOVE, "concerns": {"add": [{"text": "afternoon heat"}]},
            "move": {**MOVE["move"], "concern_id": "new"}}
    rt = runtime(answer=move)
    asyncio.run(rt.tick())
    nom = rt.state.open_nominations[0]
    concern_id = rt.state.concerns[0].concern_id
    assert nom.concern_id == concern_id
    run = {"run_id": "r1", "line": "investigate", "started_at_iso": (T0 + timedelta(hours=1)).isoformat(),
           "outcome_kind": "finished", "focus": {"focus_id": nom.focus_id, "disposition": "explored"}}
    rt2 = runtime(answer={"reading": "ok", "concerns": {"keep": [concern_id]}}, bus=rt.bus, runs=[run],
                  now=T0 + timedelta(hours=3))
    asyncio.run(rt2.load())
    asyncio.run(rt2.tick())
    [close] = [a for k, a in _concern_acts(rt.bus) if k == "act_closed"]
    assert close["act_id"] == pa.concern_act(concern_id).act_id and close["outcome"]["value"] == 1.0
    assert nom.act_id in close["outcome"]["note"]
    assert rt2.state.concerns[0].graded == "settled"  # still held: Prime decides whether to keep it
    assert any("concern settled" in o for o in rt2.state.recent_outcomes)
    assert "1 settled" in rt2.llm.calls[0]["user_prompt"]


# --- references on the move ------------------------------------------------------------------------

ARC = "8d87adeca2ce06a5"
ARC_PAGE = {f"http://dr:8121/temporal-self/arcs/{ARC}": {"arc_id": ARC, "kind": "concern", "status": "open",
                                                         "day_id": "2026-10-30"}}


def _ref_move(refs, question=f"What drives arc {ARC}?"):
    return {**MOVE, "move": {**MOVE["move"], "question": question, "refs": refs}}


def test_a_resolvable_ref_travels_to_hub_on_the_request():
    rt = runtime(answer=_ref_move([{"kind": "arc", "id": ARC}]), extra_pages=ARC_PAGE)
    asyncio.run(rt.tick())
    [(_, env)] = rt.bus.rpcs
    assert env.payload["args"]["refs"] == [{"kind": "arc", "id": ARC}]


def test_an_unresolvable_ref_refuses_the_move_and_tells_prime_without_waking_it():
    rt = runtime(answer=_ref_move([{"kind": "arc", "id": ARC}]))  # durable-runs answers 404
    asyncio.run(rt.tick())
    assert rt.bus.rpcs == [] and rt.state.open_nominations == []
    assert not [a for k, a in rt.bus.acts() if a["verb"] == "ask"]
    assert f"arc {ARC}: not found" in rt.state.recent_outcomes[-1] and rt.state.pending_changes == []


def test_a_question_with_no_refs_sends_no_refs_key_so_an_older_hub_accepts_it():
    rt = runtime(answer=MOVE)
    asyncio.run(rt.tick())
    [(_, env)] = rt.bus.rpcs
    assert "refs" not in env.payload["args"]
