"""The curiosity replica answers Prime (patch 4, docs/superpowers/specs/2026-11-04-curiosity-replica-design.md
section 9): support / challenge / refuse, each an act; refusals take effect; held-back ones do not; Prime hears
only the delivered ones. Fake bus, Redis, model, market, Postgres and Hub."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

from app.replica_loop import FOCUS_REFUSED_SQL, PENDING_FOCUS_SQL, REPLY_RETRY_SEC, ReplicaRuntime
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity.replica.answers import question_act_id
from orion.mindloop.turns import JuniperTurns
from orion.physiology import prime_charge as pc
from orion.schemas.curiosity_replica import (
    FOCUS_REFUSAL_REQUEST_CHANNEL,
    FOCUS_REFUSAL_RESULT_KIND,
    PLAN_REQUEST_KIND,
    PLAN_RESULT_CHANNEL_PREFIX,
    REPLIES_KEY,
    CuriosityFocusRefusalRequestV1,
    CuriosityFocusRefusalResultV1,
    CuriosityOfferCandidateV1,
    CuriosityOfferPlanRequestV1,
    CuriosityPendingFocusV1,
    CuriosityReplicaRepliesViewV1,
)
from test_replica_loop import IDS, LLM, PG, PLAN, SETTINGS, T0, Bus, Redis

CURIOSITY = "durable.curiosity_investigate"
FOCUS_ID = "11111111-2222-4333-8444-555555555555"
REFUSE = {"reading": "curiosity keeps failing on street priors", "stance": "refuse",
          "reason": "the last three runs died on this kind of question; it would waste the run"}
CHALLENGE = {"reading": "r", "stance": "challenge", "reason": "this repeats a question curiosity set aside yesterday"}
SUPPORT = {"reading": "r", "stance": "support", "reason": "it names a real gap"}


class ARedis(Redis):
    def __init__(self):
        super().__init__()
        self.ttls: dict[str, int] = {}
        self.fail_hset = False

    async def hset(self, key, field, value):
        if self.fail_hset:
            raise ConnectionError("redis down")
        self.h.setdefault(key, {})[field] = value

    async def ttl(self, key):
        return self.ttls.get(key, -1) if key in self.h else -2

    async def expire(self, key, sec):
        self.ttls[key] = sec


class Codec:
    def decode(self, data):
        return SimpleNamespace(ok=True, envelope=BaseEnvelope.model_validate(data))


class ABus(Bus):
    """Hub's refusal responder, faked: answers ok (or with ``hub_error``), or not at all (``hub_silent``)."""

    def __init__(self, *, hub_error=None, hub_silent=False):
        super().__init__()
        self.redis = ARedis()
        self.codec = Codec()
        self.rpcs = []
        self.timeline = []
        self.hub_error, self.hub_silent = hub_error, hub_silent

    async def publish(self, channel, envelope):
        self.timeline.append(("publish", channel))
        await super().publish(channel, envelope)

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec, require_listener=False):
        self.timeline.append(("rpc", channel))
        self.rpcs.append((channel, envelope, timeout_sec))
        assert require_listener, "a refusal RPC must fail fast when Hub is not listening"
        if self.hub_silent:
            raise TimeoutError("no reply")
        req = CuriosityFocusRefusalRequestV1.model_validate(envelope.payload)
        result = CuriosityFocusRefusalResultV1(focus_id=req.focus_id, ok=self.hub_error is None, error=self.hub_error,
                                               refused_at=T0 if self.hub_error is None else None)
        return {"data": BaseEnvelope(kind=FOCUS_REFUSAL_RESULT_KIND, correlation_id=envelope.correlation_id,
                                     source=ServiceRef(name="orion-hub"), payload=result.model_dump(mode="json")
                                     ).model_dump(mode="json")}


class APG(PG):
    def __init__(self, focus_rows=None, refused_act_id=None, **kw):
        super().__init__(**kw)
        self.focus_rows = focus_rows or []
        self.refused_act_id = refused_act_id

    def __call__(self, sql, params=None):
        if sql == PENDING_FOCUS_SQL:
            self.calls.append((sql, params))
            return list(self.focus_rows)
        if sql == FOCUS_REFUSED_SQL:
            self.calls.append((sql, params))
            return [{"refused_act_id": self.refused_act_id}]
        return super().__call__(sql, params)


def prime_focus_row(chat_corr="prime:thought0001"):
    return {"focus_id": FOCUS_ID, "origin": "self", "question": "What heats the cabinet after 14:00?",
            "chat_correlation_id": chat_corr, "created_at": T0, "expires_at": T0 + timedelta(hours=48)}


def answers_runtime(answer=REFUSE, *, refusal_holdback=0.0, challenge_holdback=0.0, per_day=3, refractory=1800,
                    bus=None, pg=None, outcome="funded", **settings):
    clock = {"now": T0}
    bids, released = [], []

    async def bid(want):
        bids.append(want)
        return SimpleNamespace(outcome=outcome, reason="net_positive", binding_constraint=None)

    async def release(want_id):
        released.append(want_id)

    def get_json(url):
        return {"runs": [{"run_id": "older1", "started_at": 1, "started_at_iso": "2026-11-04T08:00:00+00:00",
                          "outcome_kind": "died", "finding_text": "died at turn 3"}]}

    cfg = {**vars(SETTINGS), "CURIOSITY_REPLICA_ANSWERS_ENABLED": True,
           "CURIOSITY_REPLICA_REFUSAL_HOLDBACK_FRACTION": refusal_holdback,
           "CURIOSITY_REPLICA_CHALLENGE_HOLDBACK_FRACTION": challenge_holdback,
           "CURIOSITY_REPLICA_ANSWERS_PER_DAY": per_day, "CURIOSITY_REPLICA_ANSWER_REFRACTORY_SEC": refractory,
           **settings}
    rt = ReplicaRuntime(settings=SimpleNamespace(**cfg), bus=bus or ABus(), llm=LLM(answer), get_json=get_json,
                        pg_rows=pg or APG(), bid=bid, release=release, clock=lambda: clock["now"], turns=JuniperTurns())
    rt.bids, rt.released, rt.clock_box = bids, released, clock
    return rt


def put_signal(rt, *, signal_id="sig1", act_id="act_prime_regulate", direction="slow", arm="treated", hours=3.0):
    sig = pc.PrimeRegulationV1(signal_id=signal_id, act_id=act_id, family=CURIOSITY, direction=direction, arm=arm,
                               magnitude=0.5, why="curiosity ran five times this morning", opened_at=T0 - timedelta(minutes=1),
                               expires_at=T0 + timedelta(hours=hours))
    rt.bus.redis.h.setdefault(pc.key(CURIOSITY), {})[pc.SIGNAL_FIELD] = pc.dumps(sig)
    rt.bus.redis.ttls[pc.key(CURIOSITY)] = int(hours * 3600 + pc.GRACE_SEC)
    return sig


def answer_acts(rt):
    return [(k, a) for k, a in rt.bus.acts() if a["verb"] == "ask"]


def view(rt):
    raw = rt.bus.redis.kv.get(REPLIES_KEY)
    return CuriosityReplicaRepliesViewV1.model_validate_json(raw) if raw else None


def advance(rt, sec):
    rt.clock_box["now"] += timedelta(seconds=sec)


# --- a regulate signal, between runs --------------------------------------------------------------------------

def test_a_new_prime_signal_wakes_one_reply_thought_and_a_refusal_lifts_the_charge():
    rt = answers_runtime(REFUSE)
    put_signal(rt)
    asyncio.run(rt.tick())
    # Woken by the signal, no plan request: one bid as curiosity.replica, one model call with the reply frame.
    [want] = rt.bids
    assert want.family == "curiosity.replica" and want.object_ref == "prime_act:act_prime_regulate"
    [call] = rt.llm.calls
    assert "WHAT PRIME DID: sent a slow signal to curiosity (signal_id sig1" in call["user_prompt"]
    assert '"stance": "support" | "challenge" | "refuse"' in call["system_prompt"]
    assert call["pool_wait_sec"] == 20.0          # a reply waits briefly for a seat: a plan may come meanwhile
    # The refusal is written next to the signal, keyed to it; the key keeps Prime's expiry.
    ref = pc.parse_refusal(rt.bus.redis.h[pc.key(CURIOSITY)][pc.REFUSAL_FIELD])
    assert ref.signal_id == "sig1" and ref.by == "curiosity.replica" and ref.reason == REFUSE["reason"]
    assert rt.bus.redis.ttls[pc.key(CURIOSITY)] == int(3 * 3600 + pc.GRACE_SEC)
    # The answer is an act parented to Prime's regulate act, closed at once with the stance first.
    [(k1, opened), (k2, closed)] = answer_acts(rt)
    assert (k1, k2) == ("act_opened", "act_closed") and opened["actor"] == "curiosity.replica"
    assert opened["parent_act_id"] == "act_prime_regulate" and opened["test_before"] == "holdback:treated"
    assert opened["object_ref"] == "prime_signal:sig1" and ref.act_id == opened["act_id"]
    assert closed["outcome"]["note"].startswith("refuse: the last three runs died")
    assert closed["outcome"]["note"].endswith("| delivered") and closed["outcome"]["value"] == 1.0
    # Prime hears it: the replies view carries the reason.
    [reply] = view(rt).replies
    assert (reply.stance, reply.prime_act_id, reply.reason) == ("refuse", "act_prime_regulate", REFUSE["reason"])
    # Wakes once: the next tick neither bids nor thinks again.
    advance(rt, 15)
    asyncio.run(rt.tick())
    assert len(rt.bids) == 1 and len(rt.llm.calls) == 1 and rt.state.pending_targets == []


def test_a_held_back_refusal_is_recorded_but_never_applied_or_shown_to_prime():
    rt = answers_runtime(REFUSE, refusal_holdback=1.0)
    put_signal(rt)
    asyncio.run(rt.tick())
    assert pc.REFUSAL_FIELD not in rt.bus.redis.h[pc.key(CURIOSITY)]      # the charge still applies
    [(_, opened), (_, closed)] = answer_acts(rt)
    assert opened["test_before"] == "holdback:control"
    assert view(rt).replies == []                                          # never shown to Prime
    [rec] = rt.state.answers
    assert (rec.stance, rec.arm, rec.delivered) == ("refuse", "control", False)


def test_a_held_back_challenge_is_not_shown_to_prime_and_a_delivered_one_is():
    for holdback, shown in ((1.0, 0), (0.0, 1)):
        rt = answers_runtime(CHALLENGE, challenge_holdback=holdback)
        put_signal(rt)
        asyncio.run(rt.tick())
        assert len(view(rt).replies) == shown
        assert pc.REFUSAL_FIELD not in rt.bus.redis.h[pc.key(CURIOSITY)]   # a challenge never lifts the charge


def test_a_support_is_an_act_but_never_reaches_prime():
    rt = answers_runtime(SUPPORT, challenge_holdback=1.0, refusal_holdback=1.0)
    put_signal(rt)
    asyncio.run(rt.tick())
    [(_, opened), _] = answer_acts(rt)
    assert opened["test_before"] == "holdback:treated" and view(rt).replies == []


def test_a_signal_replaced_before_the_refusal_lands_is_not_delivered_and_closes_0():
    rt = answers_runtime(REFUSE)
    put_signal(rt)
    real = rt.llm.request_json

    def replace_then_answer(**kw):
        put_signal(rt, signal_id="sig2", act_id="act_other")   # Prime sent a new signal meanwhile
        return real(**kw)

    rt.llm.request_json = replace_then_answer
    asyncio.run(rt.tick())
    assert pc.REFUSAL_FIELD not in rt.bus.redis.h[pc.key(CURIOSITY)]
    [rec] = rt.state.answers
    assert rec.delivered is False and "no longer in force" in rec.delivery_note
    [_, (_, closed)] = answer_acts(rt)
    assert closed["outcome"]["value"] == 0.0 and view(rt).replies == []


# --- a Prime focus, between runs -------------------------------------------------------------------------------

def test_a_new_prime_focus_wakes_a_reply_and_a_refusal_asks_hub_to_refuse_it():
    pg = APG(focus_rows=[prime_focus_row()])
    rt = answers_runtime(REFUSE, pg=pg)
    asyncio.run(rt.tick())
    [(channel, env, timeout)] = rt.bus.rpcs
    req = CuriosityFocusRefusalRequestV1.model_validate(env.payload)
    assert channel == FOCUS_REFUSAL_REQUEST_CHANNEL and timeout == 10.0
    assert req.focus_id == FOCUS_ID and req.prime_act_id == question_act_id(FOCUS_ID) and req.reason == REFUSE["reason"]
    [(_, opened), _] = answer_acts(rt)
    assert opened["parent_act_id"] == question_act_id(FOCUS_ID) and req.act_id == opened["act_id"]
    assert opened["object_ref"] == f"curiosity_focus:{FOCUS_ID}"
    [reply] = view(rt).replies
    assert reply.target_kind == "focus" and reply.about == "What heats the cabinet after 14:00?"


def test_juniper_focus_or_orions_own_never_wakes_the_replica():
    for corr in (None, "chat-corr-123"):
        pg = APG(focus_rows=[{**prime_focus_row(), "origin": "juniper" if corr is None else "self",
                              "chat_correlation_id": corr}])
        rt = answers_runtime(REFUSE, pg=pg)
        asyncio.run(rt.tick())
        assert rt.bids == [] and rt.bus.rpcs == [] and answer_acts(rt) == []


def test_hub_refusing_the_refusal_or_not_answering_is_not_delivered_and_prime_is_not_told():
    for bus, words in ((ABus(hub_error="the focus is taken, not waiting in the slot"), "Hub refused: the focus is taken"),
                       (ABus(hub_silent=True), "the focus row shows no refusal of mine")):
        rt = answers_runtime(REFUSE, pg=APG(focus_rows=[prime_focus_row()]), bus=bus)
        asyncio.run(rt.tick())
        [rec] = rt.state.answers
        assert rec.delivered is False and words in rec.delivery_note and view(rt).replies == []


def test_a_lost_reply_is_settled_from_the_focus_row_and_the_request_carries_not_after():
    """Review finding: Hub could apply a refusal whose reply was lost; Prime then never heard it. The request now
    carries not_after (Hub writes nothing past it) and the replica reads the row back."""
    pg = APG(focus_rows=[prime_focus_row()])
    rt = answers_runtime(REFUSE, pg=pg, bus=ABus(hub_silent=True))
    act_id = None

    real_open = rt._open

    async def remember(act, component, *, at=None):
        nonlocal act_id
        if act.verb == "ask":
            act_id = act.act_id
            pg.refused_act_id = act_id          # Hub applied it before the reply was lost
        return await real_open(act, component, at=at)

    rt._open = remember
    asyncio.run(rt.tick())
    [(_, env, timeout)] = rt.bus.rpcs
    req = CuriosityFocusRefusalRequestV1.model_validate(env.payload)
    assert req.not_after is not None and abs((req.not_after - T0).total_seconds() - timeout) < 1.0
    [rec] = rt.state.answers
    assert rec.delivered is True and len(view(rt).replies) == 1


# --- the caps -------------------------------------------------------------------------------------------------

def test_one_answer_per_prime_act_a_refractory_and_a_daily_cap():
    rt = answers_runtime(CHALLENGE, per_day=2, refractory=1800)
    put_signal(rt)
    asyncio.run(rt.tick())
    assert len(rt.state.answers) == 1
    # A second Prime act 10 minutes later waits out the refractory (it stays queued, no thought).
    advance(rt, 600)
    put_signal(rt, signal_id="sig2", act_id="act_r2")
    asyncio.run(rt.tick())
    assert len(rt.llm.calls) == 1 and [t.prime_act_id for t in rt.state.pending_targets] == ["act_r2"]
    advance(rt, 1300)
    asyncio.run(rt.tick())
    assert len(rt.state.answers) == 2 and len(rt.llm.calls) == 2
    # The daily cap (2): a third act within 24 h is not answered.
    advance(rt, 1900)
    put_signal(rt, signal_id="sig3", act_id="act_r3")
    asyncio.run(rt.tick())
    assert len(rt.state.answers) == 2 and len(rt.llm.calls) == 2
    assert "at most 2" in rt._blocker(rt.state.pending_targets[0], rt.clock()).why


def test_never_an_answer_to_an_act_whose_parent_is_an_answer():
    rt = answers_runtime(CHALLENGE)
    put_signal(rt)
    asyncio.run(rt.tick())
    mine = rt.state.answers[0].answer_act_id
    from orion.curiosity.replica.answers import TargetV1

    loop = TargetV1(kind="signal", target_id="sigX", prime_act_id="act_prime_x", parent_act_id=mine)
    assert rt._blocker(loop, rt.clock() + timedelta(hours=2)).code == "answer_parent"


def test_an_unfunded_reply_backs_off_instead_of_bidding_every_tick():
    """Review must-fix: a deferred reply bid re-bid every 15 s for as long as Prime's act was live."""
    rt = answers_runtime(REFUSE, outcome="deferred")
    put_signal(rt)
    for _ in range(10):
        asyncio.run(rt.tick())
        advance(rt, 15)
    assert len(rt.bids) == 1 and rt.llm.calls == []
    advance(rt, REPLY_RETRY_SEC)
    asyncio.run(rt.tick())
    assert len(rt.bids) == 2


def test_a_reply_refused_by_the_act_gate_backs_off_too():
    rt = answers_runtime(REFUSE)
    put_signal(rt)

    from orion.mindloop.runtime import MindLoopRuntime

    async def gate_refuses(self, now):
        return False

    orig = MindLoopRuntime._thought_gate_admits
    MindLoopRuntime._thought_gate_admits = gate_refuses
    try:
        for _ in range(5):
            asyncio.run(rt.tick())
            advance(rt, 15)
    finally:
        MindLoopRuntime._thought_gate_admits = orig
    assert rt.bids == [] and rt._reply_retry_at is not None and rt._reply_retry_at > rt.clock()


def test_a_plan_request_that_arrives_during_a_reply_names_the_reply_if_it_runs_out_of_time():
    rt = answers_runtime(REFUSE)
    put_signal(rt)
    env = plan_request(from_prime=False, origin="juniper")
    real = rt.llm.request_json

    def arrive_then_answer(**kw):
        rt.accept(env)
        return real(**kw)

    rt.llm.request_json = arrive_then_answer
    asyncio.run(rt.tick())
    [pending] = rt.requests
    assert pending.miss_why == "a reply thought to Prime was running when it arrived"


def test_answers_off_means_no_watch_no_reply_no_view():
    rt = answers_runtime(REFUSE, CURIOSITY_REPLICA_ANSWERS_ENABLED=False, pg=APG(focus_rows=[prime_focus_row()]))
    put_signal(rt)
    asyncio.run(rt.tick())
    assert rt.bids == [] and REPLIES_KEY not in rt.bus.redis.kv and rt.state.pending_targets == []
    assert not any(sql == PENDING_FOCUS_SQL for sql, _ in rt.pg_rows.calls)


# --- inside a plan ---------------------------------------------------------------------------------------------

def plan_request(*, from_prime=True, origin="self", now=T0):
    req = CuriosityOfferPlanRequestV1(
        run_id="run000000009", correlation_id=str(uuid4()), deadline_at=now + timedelta(seconds=90),
        candidates=[CuriosityOfferCandidateV1(prior_id=i, claim=f"claim {i}", confidence=0.5) for i in IDS],
        stale_ids=[], pending_focus=CuriosityPendingFocusV1(focus_id=FOCUS_ID, origin=origin,
                                                           question="What heats the cabinet after 14:00?",
                                                           from_prime=from_prime))
    corr = uuid4()
    return BaseEnvelope(kind=PLAN_REQUEST_KIND, correlation_id=corr, reply_to=f"{PLAN_RESULT_CHANNEL_PREFIX}{corr}",
                        source=ServiceRef(name="orion-hub"), payload=req.model_dump(mode="json"))


def test_a_plan_that_refuses_primes_focus_refuses_it_before_answering_hub():
    plan = {**PLAN, "focus_stance": {"stance": "refuse", "reason": "the run should test priors, not the cabinet again"}}
    rt = answers_runtime(plan)
    rt.accept(plan_request())
    asyncio.run(rt.tick())
    kinds = [k for k, c in rt.bus.timeline if k == "rpc" or c.startswith(PLAN_RESULT_CHANNEL_PREFIX)]
    # Hub takes the focus after the plan answer, so the refusal must land first.
    assert kinds == ["rpc", "publish"]
    [(_, env, timeout)] = rt.bus.rpcs
    assert 1.0 <= timeout <= 10.0
    [rec] = rt.state.answers
    assert (rec.source, rec.stance, rec.delivered, rec.prime_act_id) == ("plan", "refuse", True, question_act_id(FOCUS_ID))


def test_a_plan_stance_on_juniper_focus_is_recorded_on_the_plan_row_only():
    plan = {**PLAN, "focus_stance": {"stance": "refuse", "reason": "no"}}
    rt = answers_runtime(plan)
    rt.accept(plan_request(from_prime=False, origin="juniper"))
    asyncio.run(rt.tick())
    assert rt.bus.rpcs == [] and answer_acts(rt) == [] and rt.state.answers == []


def test_one_plan_answers_both_the_focus_and_the_signal():
    """Review finding: the second stance of one plan was dropped by the 30-minute refractory."""
    plan = {**PLAN, "focus_stance": {"stance": "challenge", "reason": "it repeats yesterday's question"},
            "signal_stance": {"signal_id": "sig1", "stance": "refuse", "reason": "curiosity is not running hot"}}
    rt = answers_runtime(plan)
    put_signal(rt)
    rt.accept(plan_request())
    asyncio.run(rt.tick())
    assert sorted((a.kind, a.stance) for a in rt.state.answers) == [("focus", "challenge"), ("signal", "refuse")]
    assert pc.parse_refusal(rt.bus.redis.h[pc.key(CURIOSITY)][pc.REFUSAL_FIELD]).signal_id == "sig1"
    assert len({a.thought_id for a in rt.state.answers}) == 1


def test_the_replies_view_expires_with_its_window():
    rt = answers_runtime(CHALLENGE)
    put_signal(rt)
    seen = {}
    real = rt.bus.redis.set

    async def spy(key, value, ex=None):
        seen[key] = ex
        return await real(key, value, ex=ex)

    rt.bus.redis.set = spy
    asyncio.run(rt.tick())
    assert seen[REPLIES_KEY] == 48 * 3600


def test_a_plan_signal_stance_on_the_live_signal_is_an_answer_to_prime():
    rt = answers_runtime({**PLAN, "signal_stance": {"signal_id": "sig1", "stance": "challenge",
                                                    "reason": "runs this morning found things"}})
    put_signal(rt)
    rt.accept(plan_request(from_prime=False, origin="juniper"))
    asyncio.run(rt.tick())
    # The plan thought answered it, so the signal does not wake a second (reply) thought.
    [rec] = rt.state.answers
    assert (rec.source, rec.kind, rec.prime_act_id, rec.arm) == ("plan", "signal", "act_prime_regulate", "treated")
    advance(rt, 15)
    asyncio.run(rt.tick())
    assert len(rt.llm.calls) == 1


def test_the_plan_frame_shows_the_replicas_refusal_rate():
    rt = answers_runtime(REFUSE)
    put_signal(rt)
    asyncio.run(rt.tick())
    advance(rt, 3600)
    rt.llm.answer = PLAN
    rt.accept(plan_request(from_prime=False, origin="juniper", now=rt.clock()))
    asyncio.run(rt.tick())
    assert len(rt.llm.calls) == 2
    frame = rt.llm.calls[-1]["user_prompt"]
    assert "YOUR ANSWERS TO PRIME" in frame and "1 of your last 1 answers to Prime were refusals (100%)" in frame
