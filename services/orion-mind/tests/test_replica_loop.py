"""The curiosity replica (app/replica_loop.py): loop-library instance #2, with a fake bus, model, market and
Postgres. Design: docs/superpowers/specs/2026-11-04-curiosity-replica-design.md sections 1-4, 7, 8."""

from __future__ import annotations

import ast
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from app.replica_loop import (
    DIGEST_SQL,
    FALLBACK_COUNTS_SQL,
    OUTCOMES_SQL,
    PLAN_ROWS_SQL,
    RECENT_PLANS_SQL,
    STATE_KEY,
    ReplicaRuntime,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity.replica.state import CuriosityReplicaStateV1
from orion.mindloop.turns import JuniperTurns
from orion.physiology import prime_charge as pc
from orion.prime.state import PRIME_CONCERNS_KEY, PrimeConcernsViewV1
from orion.schemas.curiosity_replica import (
    PLAN_REQUEST_KIND,
    PLAN_RESULT_CHANNEL_PREFIX,
    PLAN_RESULT_KIND,
    CuriosityOfferCandidateV1,
    CuriosityOfferPlanRequestV1,
    CuriosityOfferPlanResultV1,
)

T0 = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
IDS = [f"self:prior_{i}" for i in range(8)]
PLAN = {"reading": "curiosity keeps testing street priors and moving little",
        "picks": [{"prior_id": IDS[3], "why": "near even and never tested"},
                  {"prior_id": IDS[0], "why": "moved last time"},
                  {"prior_id": IDS[5], "why": "Juniper talked about it yesterday"}],
        "focus_stance": None, "signal_stance": None, "abstain": False}


class Redis:
    def __init__(self):
        self.kv, self.h = {}, {}

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, ex=None):
        self.kv[key] = value

    async def hgetall(self, key):
        return dict(self.h.get(key, {}))


class Bus:
    def __init__(self):
        self.redis = Redis()
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))

    def acts(self):
        return [(e.payload["event_kind"], e.payload["act"]) for c, e in self.published if c == "orion:grammar:event"]

    def results(self):
        return [(c, e) for c, e in self.published if c.startswith(PLAN_RESULT_CHANNEL_PREFIX)]


class LLM:
    def __init__(self, answer, *, err=None, gateway_error=None):
        self.answer, self.err, self.gw, self.calls = answer, err, gateway_error, []
        self.on_call = None

    def request_json(self, **kw):
        self.calls.append(kw)
        if self.on_call:
            self.on_call()
        return self.answer, self.err, ({"gateway_error": self.gw} if self.gw else {})


SETTINGS = SimpleNamespace(
    SERVICE_NAME="orion-mind", SERVICE_VERSION="t", NODE_NAME="test", CURIOSITY_REPLICA_TICK_SEC=15,
    CURIOSITY_REPLICA_REFRACTORY_SEC=0, CURIOSITY_REPLICA_SETTLE_INTERVAL_SEC=300, CURIOSITY_REPLICA_LLM_ROUTE="prime",
    CURIOSITY_REPLICA_SEAT_WAIT_SEC=45, CURIOSITY_REPLICA_BID_LANE="chat", CURIOSITY_REPLICA_MAX_TOKENS=500,
    CURIOSITY_REPLICA_LLM_TIMEOUT_SEC=40, CURIOSITY_REPLICA_HUB_URL="http://hub:8080", CURIOSITY_REPLICA_PG_DSN="",
)


class PG:
    """SQL -> rows, by the module's own query constants."""

    def __init__(self, plan_rows=None, fail=False):
        self.plan_rows, self.fail, self.calls = plan_rows or [], fail, []

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))
        if self.fail:
            raise RuntimeError("could not connect")
        if sql == PLAN_ROWS_SQL:
            return [r for r in self.plan_rows if r["run_id"] in params["ids"]]
        if sql == RECENT_PLANS_SQL:
            return list(self.plan_rows)
        if sql == FALLBACK_COUNTS_SQL:
            return [{"status": "ok", "n": 2}]
        if sql == OUTCOMES_SQL:
            return [{"run_id": "older1", "turn_ok": True, "realized_nats": 0.01,
                     "per_prior": [{"kind": "tested", "prior_id": IDS[1], "nats": 0.01}]}]
        if sql == DIGEST_SQL:
            return [{"summary": "Juniper and Orion talked about the walkway camera.", "title": "Walkway day",
                     "window_end": "2026-11-03T06:00:00+00:00", "created_at": T0 - timedelta(hours=23),
                     "index": "chat_compactor:day:2026-11-02"}]
        raise AssertionError(f"unexpected SQL {sql[:60]}")


def runtime(answer=PLAN, *, outcome="funded", err=None, gateway_error=None, pg=None, turns=None, bus=None,
            reason="net_positive", net=None):
    clock = {"now": T0}
    bids, released = [], []

    async def bid(want):
        bids.append(want)
        return SimpleNamespace(outcome=outcome, reason=reason, net=net, binding_constraint=None)

    async def release(want_id):
        released.append(want_id)

    def get_json(url):
        assert url.startswith("http://hub:8080/curiosity/api/runs")
        return {"runs": [{"run_id": "older1", "started_at": 1, "started_at_iso": "2026-11-04T08:00:00+00:00",
                          "outcome_kind": "finished", "finding_text": "street rhythm held"}]}

    rt = ReplicaRuntime(settings=SimpleNamespace(**vars(SETTINGS)), bus=bus or Bus(),
                        llm=LLM(answer, err=err, gateway_error=gateway_error), get_json=get_json,
                        pg_rows=pg or PG(), bid=bid, release=release, clock=lambda: clock["now"],
                        turns=turns or JuniperTurns())
    rt.bids, rt.released, rt.clock_box = bids, released, clock
    return rt


def advance(rt, sec):
    rt.clock_box["now"] += timedelta(seconds=sec)


def request_envelope(run_id="run000000001", deadline_sec=90):
    req = CuriosityOfferPlanRequestV1(
        run_id=run_id, correlation_id=str(uuid4()), deadline_at=T0 + timedelta(seconds=deadline_sec),
        candidates=[CuriosityOfferCandidateV1(prior_id=i, claim=f"claim {i}", confidence=0.5) for i in IDS],
        stale_ids=["self:stale_1"])
    corr = uuid4()
    return BaseEnvelope(kind=PLAN_REQUEST_KIND, correlation_id=corr, reply_to=f"{PLAN_RESULT_CHANNEL_PREFIX}{corr}",
                        source=ServiceRef(name="orion-hub"), payload=req.model_dump(mode="json"))


def result_of(rt):
    [(channel, env)] = rt.bus.results()
    assert env.kind == PLAN_RESULT_KIND
    return channel, env, CuriosityOfferPlanResultV1.model_validate(env.payload)


def test_a_funded_request_bids_as_the_replica_answers_hub_and_holds_its_plan_act(caplog):
    caplog.set_level(logging.INFO, logger="orion-mind.curiosity_replica")
    rt = runtime()
    env = request_envelope()
    assert rt.accept(env) and rt._wake.is_set()
    asyncio.run(rt.tick())
    [want] = rt.bids
    assert (want.family, want.producer, want.tier, want.verb, want.lanes) == (
        "curiosity.replica", "orion-mind.curiosity_replica", 2, "infer", ["chat"])
    assert want.object_ref == "curiosity_run:run000000001" and want.cost.gpu_slot_s_by_lane == {"chat": 40.0}
    [call] = rt.llm.calls
    assert call["route"] == "prime" and call["max_tokens"] == 500 and call["thinking"] is False
    # Seat wait cut so seat wait + 40 s generation end 5 s before the 90 s deadline.
    assert call["pool_wait_sec"] == 45.0
    frame = call["user_prompt"]
    assert IDS[3] in frame and "Walkway day" in frame and "street rhythm held" in frame
    channel, out, result = result_of(rt)
    assert channel == env.reply_to and out.correlation_id == env.correlation_id
    assert result.status == "ok" and [p.prior_id for p in result.picks] == [IDS[3], IDS[0], IDS[5]]
    assert result.frame_tokens and result.frame_tokens > 0
    # The plan act opened and is held open: it closes on the run's score.
    acts = rt.bus.acts()
    assert [(k, a["actor"], a["verb"]) for k, a in acts] == [("act_opened", "curiosity.replica", "infer")]
    assert acts[0][1]["act_id"] == result.act_id and acts[0][1]["goal_ref"] == "curiosity_run:run000000001"
    [held] = rt.state.open_plans
    assert held.run_id == "run000000001" and held.picks == [IDS[3], IDS[0], IDS[5]] and rt.requests == []
    # Its own key only.
    assert set(rt.bus.redis.kv) == {STATE_KEY}
    assert CuriosityReplicaStateV1.model_validate_json(rt.bus.redis.kv[STATE_KEY]).plans_total == 1
    assert any("curiosity_replica_plan run=run000000001 status=ok" in r.getMessage() for r in caplog.records)


def _plan_row(run_id, **over):
    row = {"run_id": run_id, "requested_at": T0, "status": "ok", "fallback_reason": None,
           "picks": [{"prior_id": IDS[3], "why": "w"}], "act_id": "a", "turn_ok": None, "realized_nats": None,
           "unknown_reason": None, "per_prior": []}
    row.update(over)
    return row


def test_the_plan_act_closes_on_the_runs_score():
    pg = PG()
    rt = runtime(pg=pg)
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    act_id = rt.state.open_plans[0].act_id
    pg.plan_rows = [_plan_row("run000000001")]  # Hub recorded it; the run has not been scored yet
    advance(rt, 301)
    asyncio.run(rt.tick())
    assert len(rt.state.open_plans) == 1 and len(rt.bus.acts()) == 1
    pg.plan_rows = [_plan_row("run000000001", turn_ok=True, realized_nats=0.02,
                              per_prior=[{"kind": "tested", "prior_id": IDS[3], "nats": 0.02}])]
    advance(rt, 301)
    asyncio.run(rt.tick())
    kind, act = rt.bus.acts()[-1]
    assert kind == "act_closed" and act["act_id"] == act_id
    assert act["outcome"]["value"] == 1.0
    assert "the run moved 0.020 nats" in act["outcome"]["note"] and rt.state.open_plans == []


def test_a_plan_hub_did_not_use_closes_zero_and_an_unknown_score_is_never_a_zero():
    for row, closes in ((_plan_row("run000000001", status="timeout", fallback_reason="timeout"), True),
                        (_plan_row("run000000001", turn_ok=True, realized_nats=None, unknown_reason="no_snapshot"), False)):
        rt = runtime(pg=PG(plan_rows=[row]))
        rt.accept(request_envelope())
        asyncio.run(rt.tick())
        advance(rt, 301)
        asyncio.run(rt.tick())
        assert rt.state.open_plans == []
        closed = [a for k, a in rt.bus.acts() if k == "act_closed"]
        assert bool(closed) is closes
        if closes:
            assert closed[0]["outcome"]["value"] == 0.0
            assert "plan not used: timeout" in closed[0]["outcome"]["note"]


def test_not_funded_rebids_until_there_is_no_time_then_answers_unfunded(caplog):
    caplog.set_level(logging.INFO, logger="orion-mind.curiosity_replica")
    rt = runtime(outcome="deferred")
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    advance(rt, 15)
    asyncio.run(rt.tick())
    assert len(rt.bids) == 2 and rt.bus.results() == [] and rt.llm.calls == []
    advance(rt, 31)  # 46 s in: 39 s left, less than one 40 s generation
    asyncio.run(rt.tick())
    _, _, result = result_of(rt)
    assert result.status == "unfunded" and result.act_id is None and rt.bus.acts() == []
    assert len(rt.bids) == 2 and rt.requests == []
    assert "could not plan run run000000001 in time (unfunded: bid deferred" in rt.state.notes[-1]


def test_a_juniper_turn_yields_and_the_give_up_says_so():
    turns = JuniperTurns()
    rt = runtime(turns=turns)
    rt.accept(request_envelope(deadline_sec=60))
    with turns.turn():
        asyncio.run(rt.tick())
        advance(rt, 20)
        asyncio.run(rt.tick())
    _, _, result = result_of(rt)
    assert result.status == "unfunded" and rt.bids == []
    assert "a Juniper turn was running" in rt.state.notes[-1]


def test_a_try_that_never_reached_the_model_is_retried_not_answered():
    rt = runtime(answer=None, err="empty_llm_response", gateway_error={"error": "gpu_pool_unavailable"})
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    assert rt.bus.results() == [] and rt.bus.acts() == [] and len(rt.requests) == 1
    assert rt.requests[0].miss_status == "unfunded"  # no seat: no capacity
    rt.llm.answer, rt.llm.err, rt.llm.gw = PLAN, None, None
    advance(rt, 15)
    asyncio.run(rt.tick())
    assert result_of(rt)[2].status == "ok"
    # 30 s left of seat wait budget at 15 s in (90 - 15 - 5 - 40).
    assert rt.llm.calls[-1]["pool_wait_sec"] == 30.0


def test_an_unparseable_or_abstaining_answer_closes_its_act_at_once():
    for answer, status in (({"reading": "x", "picks": "not a list"}, "unparseable"),
                           ({"reading": "nothing is worth reordering", "picks": [], "abstain": True}, "abstain"),
                           ({"reading": "x", "picks": []}, "unparseable")):
        rt = runtime(answer=answer)
        rt.accept(request_envelope())
        asyncio.run(rt.tick())
        assert result_of(rt)[2].status == status
        assert [k for k, _ in rt.bus.acts()] == ["act_opened", "act_closed"]
        assert rt.bus.acts()[1][1]["outcome"]["value"] == 0.0
        assert rt.state.open_plans == [] and rt.requests == []


def test_a_plan_that_fails_its_own_check_is_still_graded_from_hubs_row():
    """The act never guesses Hub's verdict: it is held, and Hub's invalid:<reason> row closes it 0."""
    bad = {**PLAN, "picks": [{"prior_id": IDS[0], "why": "a"}, {"prior_id": "not-a-candidate", "why": "b"},
                             {"prior_id": IDS[2], "why": "c"}]}
    pg = PG()
    rt = runtime(answer=bad, pg=pg)
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    assert result_of(rt)[2].status == "ok" and len(rt.state.open_plans) == 1
    pg.plan_rows = [_plan_row("run000000001", status="invalid:not_a_candidate:2", fallback_reason="not_a_candidate:2")]
    advance(rt, 301)
    asyncio.run(rt.tick())
    closed = rt.bus.acts()[-1][1]
    assert closed["outcome"]["value"] == 0.0 and "plan not used: invalid:not_a_candidate:2" in closed["outcome"]["note"]
    assert rt.state.open_plans == []


def test_an_answer_just_past_the_deadline_is_held_and_graded_on_what_hub_did():
    """Review finding: Hub's timer starts after it builds the request, so an answer a moment past
    deadline_at can still be used. The act follows Hub's row either way."""
    pg = PG()
    rt = runtime(pg=pg)
    rt.llm.on_call = lambda: advance(rt, 91)
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    assert result_of(rt)[2].status == "ok" and len(rt.state.open_plans) == 1
    pg.plan_rows = [_plan_row("run000000001", turn_ok=True, realized_nats=0.004,
                              per_prior=[{"kind": "tested", "prior_id": IDS[3], "nats": 0.004}])]
    advance(rt, 301)
    asyncio.run(rt.tick())
    assert rt.bus.acts()[-1][1]["outcome"]["value"] == 1.0


def test_frame_reads_that_eat_the_budget_make_no_model_call():
    rt = runtime()
    real = rt.get_json

    def slow(url):
        advance(rt, 46)  # 90 - 46 - 5 = 39 s left: less than one 40 s generation
        return real(url)

    rt.get_json = slow
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    assert rt.llm.calls == [] and rt.bus.acts() == [] and rt.bus.results() == []
    asyncio.run(rt.tick())
    result = result_of(rt)[2]
    assert result.status == "unfunded" and "no time left after the frame reads" in rt.state.notes[-1]


def test_the_seat_wait_is_never_zero():
    """0 would send no seat wait at all, and the gateway would then wait its own 300 s for a seat."""
    rt = runtime()
    rt.accept(request_envelope())
    advance(rt, 44.8)  # 40.2 s left: one generation fits, the seat wait is cut to its floor
    asyncio.run(rt.tick())
    assert rt.llm.calls[0]["pool_wait_sec"] == 0.5


def test_the_5xx_retry_runs_only_if_it_can_still_end_in_time():
    five = (None, "empty_llm_response", {"gateway_error": {"error": "upstream_http_5xx", "reason": None}})
    # Start at 0: 20 s in, 65 s left, a second 40 s call fits. Start at 30: 50 s in, 35 s left, it does not.
    for start, calls in ((0, 2), (30, 1)):
        rt = runtime()
        replies = [five, (PLAN, None, {})]
        rt.llm.request_json = lambda replies=replies, rt=rt, **kw: (rt.llm.calls.append(kw), advance(rt, 20),
                                                                    replies.pop(0))[2]  # each try takes 20 s
        rt.accept(request_envelope())
        advance(rt, start)
        asyncio.run(rt.tick())
        assert len(rt.llm.calls) == calls, start
    # The late one was not retried: it never reached the model, and is given up as model_failed.
    assert rt.bus.results() == [] and rt.requests[0].miss_status == "model_failed"


def test_a_thought_that_crashes_answers_model_failed_at_once():
    rt = runtime()

    async def boom(req, now):
        raise RuntimeError("frame bug")

    rt._read_frame_inputs = boom
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    assert result_of(rt)[2].status == "model_failed"
    assert [k for k, _ in rt.bus.acts()] == ["act_opened", "act_closed"] and rt.requests == []


def test_a_request_that_arrives_while_another_is_given_up_is_kept():
    rt = runtime(outcome="deferred")
    rt.accept(request_envelope(run_id="runA00000001", deadline_sec=40))
    fresh = request_envelope(run_id="runB00000001")
    publish = rt.bus.publish

    async def publish_and_accept(channel, envelope):
        await publish(channel, envelope)
        if channel.startswith(PLAN_RESULT_CHANNEL_PREFIX):
            rt.accept(fresh)

    rt.bus.publish = publish_and_accept
    asyncio.run(rt.tick())
    assert result_of(rt)[2].run_id == "runA00000001"
    assert [r.request.run_id for r in rt.requests] == ["runB00000001"]


def test_a_second_request_for_the_run_being_planned_is_ignored():
    rt = runtime()
    env = request_envelope()
    rt.accept(env)
    rt.llm.on_call = lambda: rt.accept(request_envelope())
    asyncio.run(rt.tick())
    assert len(rt.llm.calls) == 1 and len(rt.bus.results()) == 1 and rt.requests == []


def test_a_failed_turn_is_final_only_after_the_retry_grace():
    pg = PG()
    rt = runtime(pg=pg)
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    pg.plan_rows = [_plan_row("run000000001", turn_ok=False, realized_nats=0.0, completed_at=T0 + timedelta(minutes=30))]
    advance(rt, 3600)
    asyncio.run(rt.tick())
    assert len(rt.state.open_plans) == 1  # a durable retry may still replace the failure
    advance(rt, 6 * 3600)
    asyncio.run(rt.tick())
    closed = rt.bus.acts()[-1][1]
    assert closed["outcome"]["value"] == 0.0 and "the run failed" in closed["outcome"]["note"]


def test_the_listener_decodes_real_bus_bytes():
    """The _listen path with the real codec and raw pubsub messages (accept() alone skips the decode)."""
    from contextlib import asynccontextmanager

    from orion.core.bus.codec import OrionCodec

    env = request_envelope()
    rt = runtime()

    class PubBus(Bus):
        codec = OrionCodec()

        @asynccontextmanager
        async def subscribe(self, channel):
            assert channel == "orion:curiosity:replica:plan:request"
            yield object()

        async def iter_messages(self, pubsub):
            yield {"type": "message", "data": self.codec.encode(env)}
            yield {"type": "message", "data": b"not json"}
            rt._stop.set()

    rt.bus = PubBus()
    asyncio.run(rt._listen())
    assert [r.request.run_id for r in rt.requests] == ["run000000001"]
    assert rt.requests[0].correlation_id == env.correlation_id and rt.requests[0].reply_to == env.reply_to


def test_stances_are_sent_and_recorded_only():
    sig = pc.PrimeRegulationV1(signal_id="sig1", act_id="a1", family="durable.curiosity_investigate",
                               direction="slow", arm="control", magnitude=0.5, why="too many runs",
                               opened_at=T0 - timedelta(hours=1), expires_at=T0 + timedelta(hours=2))
    bus = Bus()
    bus.redis.h[pc.key("durable.curiosity_investigate")] = {pc.SIGNAL_FIELD: pc.dumps(sig)}
    answer = {**PLAN, "signal_stance": {"signal_id": "sig1", "stance": "challenge", "reason": "runs are not crowding"},
              "focus_stance": {"stance": "support", "reason": "it is a good question"}}
    rt = runtime(answer=answer, bus=bus)
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    frame = rt.llm.calls[0]["user_prompt"]
    assert "slow (signal_id sig1" in frame and "control" not in frame  # the arm is hidden
    result = result_of(rt)[2]
    assert result.signal_stance.stance == "challenge" and result.focus_stance.stance == "support"
    assert len(rt.state.open_plans) == 1  # a valid stance does not change the plan
    # Nothing was written back to Prime's signal.
    assert set(bus.redis.h[pc.key("durable.curiosity_investigate")]) == {pc.SIGNAL_FIELD}


def test_reads_that_fail_are_gaps_in_the_frame_not_a_dead_loop():
    rt = runtime(pg=PG(fail=True))
    rt.get_json = lambda url: (_ for _ in ()).throw(OSError("refused"))
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    frame = rt.llm.calls[0]["user_prompt"]
    assert frame.count("(unavailable:") >= 4 and "Prime has published no concerns view" in frame
    assert result_of(rt)[2].status == "ok"


def test_it_reads_prime_only_through_the_published_view():
    """Design section 7: no reference to Prime's state, key or runtime anywhere in the instance."""
    bus = Bus()
    view = PrimeConcernsViewV1(as_of=T0, concerns=[{"concern_id": "c1", "text": "heat after 14:00", "age_h": 2.0}])
    bus.redis.kv[PRIME_CONCERNS_KEY] = view.model_dump_json()
    bus.redis.kv["orion:prime:state"] = "{\"secret\": \"Prime's private state\"}"
    rt = runtime(bus=bus)
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    frame = rt.llm.calls[0]["user_prompt"]
    assert "heat after 14:00" in frame and "private state" not in frame
    src = (Path(__file__).resolve().parents[1] / "app" / "replica_loop.py").read_text()
    tree = ast.parse(src)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not imported & {"PrimeStateV1", "PrimeRuntime", "STATE_KEY_PRIME"} and "prime_loop" not in src
    assert "orion:prime:state" not in src
    assert not any(isinstance(n, ast.ImportFrom) and (n.module or "").endswith("prime_loop") for n in ast.walk(tree))


def test_a_request_that_is_not_one_is_ignored():
    rt = runtime()
    bad = request_envelope()
    assert not rt.accept(bad.model_copy(update={"kind": "something.else"}))
    assert not rt.accept(bad.model_copy(update={"payload": {"run_id": "x"}}))
    assert rt.requests == []


def test_beside_prime_on_one_bus_neither_touches_the_other():
    from test_prime_loop import MOVE
    from test_prime_loop import runtime as prime_runtime

    bus = Bus()
    prime = prime_runtime(answer=MOVE, bus=bus)
    rt = runtime(bus=bus, turns=prime.turns)
    asyncio.run(prime.tick())
    prime_saved = bus.redis.kv["orion:prime:state"]
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    assert bus.redis.kv["orion:prime:state"] == prime_saved
    assert {"orion:prime:state", PRIME_CONCERNS_KEY, STATE_KEY} == set(bus.redis.kv)
    assert [w.family for w in prime.bids] == ["prime"] and [w.family for w in rt.bids] == ["curiosity.replica"]
    # The replica read the concerns view Prime just published.
    assert "PRIME'S CONCERNS (Prime's published view" in rt.llm.calls[0]["user_prompt"]


def test_lifespan_starts_the_replica_on_its_own_bus_and_stops_it(monkeypatch):
    """A bus that cannot connect at boot delays the replica (retry in its task); it never blocks the lifespan.
    Off: no bus, no task. Prime is off here (its own lifespan test is in test_prime_loop.py)."""
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

    async def _noop():
        return None

    monkeypatch.setattr(main, "OrionBusAsync", FailingBus)
    monkeypatch.setattr(main, "build_heartbeat_chassis", lambda: SimpleNamespace(start_background=_noop, stop=_noop))
    monkeypatch.setattr(main.settings, "RPC_HEALTH_PUBLISH_ENABLED", False)
    monkeypatch.setattr(main.settings, "PRIME_CONSENT_ENABLED", False)
    monkeypatch.setattr(main.settings, "ORION_BUS_ENABLED", True)
    monkeypatch.setattr(main.settings, "PRIME_ENABLED", False)
    monkeypatch.setattr(main.settings, "CURIOSITY_REPLICA_ENABLED", True)

    async def go():
        async with main.lifespan(FastAPI()):
            assert isinstance(main.replica_runtime, main.ReplicaRuntime)
            await asyncio.sleep(0)
        assert main.replica_runtime is None and made[0].closed

    asyncio.run(go())
    monkeypatch.setattr(main.settings, "CURIOSITY_REPLICA_ENABLED", False)
    made.clear()

    async def off():
        async with main.lifespan(FastAPI()):
            assert main.replica_runtime is None and made == []

    asyncio.run(off())


def test_a_capacity_deferral_worth_its_price_plans_within_the_deadline():
    # Lane full, net positive: the thought queues at the GPU pool; the seat wait still ends before the deadline.
    rt = runtime(outcome="deferred", reason="capacity_reserved", net=0.5)
    rt.accept(request_envelope())
    asyncio.run(rt.tick())
    [want] = rt.bids
    [call] = rt.llm.calls
    assert call["pool_wait_sec"] == 45.0 and rt.released == [want.want_id]
    _, _, result = result_of(rt)
    assert result.status == "ok"
