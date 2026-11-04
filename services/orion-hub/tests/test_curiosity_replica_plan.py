"""Hub's side of the curiosity replica (scripts/curiosity_replica_plan.py, wired into _investigate): ask for a
plan on the replica arm, check it, use it or fall back, and record one plan row either way.

Design: docs/superpowers/specs/2026-11-04-curiosity-replica-design.md sections 1, 4, 5 and 7. Drives the real
loop with the spend-log fakes; the replica's answer comes from a fake bus RPC."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity.kickoff_prompt import ORDER_SENTENCE
from orion.curiosity.value import ARM_REPLICA_ORDER, ARM_UNCERTAINTY_ORDER, ARM_VALUE_ORDER, draw_offer_arm
from orion.schemas.curiosity_replica import (
    PLAN_REQUEST_CHANNEL,
    PLAN_REQUEST_KIND,
    PLAN_RESULT_KIND,
    CuriosityOfferPlanRequestV1,
    CuriosityOfferPlanResultV1,
)
from scripts.curiosity_offer_decisions import INSERT_PLAN_SQL
from test_curiosity_investigation import _FakeBus
from test_curiosity_spend_log import _SpendConn, _moves_during_turn, _reader, _state

WHY = "REPLICA-REASON-NEVER-SHOWN"


class _Codec:
    def decode(self, data):
        return SimpleNamespace(ok=True, envelope=BaseEnvelope.model_validate(data))


class _PlanBus(_FakeBus):
    """``answer(request) -> payload dict | Exception``: what the replica sends back."""

    def __init__(self, answer):
        super().__init__()
        self.codec = _Codec()
        self.answer = answer
        self.requests: list[tuple[str, BaseEnvelope, str, float]] = []

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
        self.requests.append((channel, envelope, reply_channel, timeout_sec))
        req = CuriosityOfferPlanRequestV1.model_validate(envelope.payload)
        out = self.answer(req)
        if isinstance(out, BaseException):
            raise out
        kind = out.pop("_kind", PLAN_RESULT_KIND)
        reply = BaseEnvelope(kind=kind, correlation_id=envelope.correlation_id, source=ServiceRef(name="orion-mind"),
                             payload=out)
        return {"data": reply.model_dump(mode="json")}


class _PlanConn(_SpendConn):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.plans: dict[str, dict] = {}

    async def execute(self, sql, *args):
        if sql == INSERT_PLAN_SQL:
            keys = ("run_id", "plan_id", "requested_at", "answered_at", "status", "fallback_reason", "picks",
                    "focus_stance", "signal_stance", "act_id", "latency_ms", "frame_tokens")
            row = dict(zip(keys, args))
            row["picks"] = json.loads(row["picks"])
            self.plans.setdefault(row["run_id"], row)
            return "OK"
        return await super().execute(sql, *args)


STATES = [_state(f"p{i}", c, t) for i, (c, t) in enumerate(
    [(0.5, 0), (0.52, 1), (0.6, 0), (0.7, 1), (0.8, 2), (0.9, 0), (0.45, 1), (0.3, 0), (0.95, 2), (0.2, 1),
     (0.55, 4)])]  # p10 is stale (tested 4 >= 3)


def _ok(picks):
    def answer(req):
        return CuriosityOfferPlanResultV1(
            run_id=req.run_id, plan_id="plan1", status="ok", act_id="act-1", latency_ms=1200, frame_tokens=3100,
            picks=[{"prior_id": p, "why": f"{WHY} {p}"} for p in picks]).model_dump(mode="json")
    return answer


def _run(answer, **loop_over):
    from test_curiosity_investigation import _graph_loop

    bus = _PlanBus(answer)
    conn = _PlanConn()
    reader = _reader(STATES)
    kw = dict(replica_arm_enabled=True, replica_arm_propensity=1.0, value_order_enabled=False)
    kw.update(loop_over)
    loop = _graph_loop(bus, reader=reader, conn=conn, **kw)
    _moves_during_turn(loop, reader, lambda rid: STATES)
    assert asyncio.run(loop.tick()) is None
    ((run_id, decision),) = conn.decisions.items()
    return loop, bus, conn, run_id, decision


def _offered(decision):
    return [o["prior_id"] for o in decision["offered"]]


def _ordered(decision):
    return [o["prior_id"] for o in decision["offered"] if o["slot"] == "order"]


def test_a_good_plan_sets_the_top_of_the_offer_and_its_reasons_never_reach_orion(caplog):
    caplog.set_level(logging.INFO, logger="orion-hub.curiosity_replica_plan")
    loop, bus, conn, run_id, decision = _run(_ok(["p7", "p3", "p9"]))
    [(channel, env, reply, timeout)] = bus.requests
    assert channel == PLAN_REQUEST_CHANNEL and env.kind == PLAN_REQUEST_KIND and timeout == 90.0
    req = CuriosityOfferPlanRequestV1.model_validate(env.payload)
    assert req.run_id == run_id and reply.endswith(str(env.correlation_id))
    assert {c.prior_id for c in req.candidates} == {f"p{i}" for i in range(10)} and req.stale_ids == ["p10"]
    assert decision["arm"] == ARM_REPLICA_ORDER
    assert decision["arm_propensities"] == {ARM_VALUE_ORDER: 0.0, ARM_REPLICA_ORDER: 1.0, ARM_UNCERTAINTY_ORDER: 0.0}
    assert _offered(decision)[:3] == ["p7", "p3", "p9"] and len(decision["offered"]) == 8
    plan = conn.plans[run_id]
    assert plan["status"] == "ok" and plan["fallback_reason"] is None and plan["act_id"] == "act-1"
    assert [p["prior_id"] for p in plan["picks"]] == ["p7", "p3", "p9"] and WHY in plan["picks"][0]["why"]
    prompt = loop.seen_prompt
    assert WHY not in prompt and "plan1" not in prompt
    assert prompt.index("claim p7") < prompt.index("claim p3") < prompt.index("claim p9")
    assert ORDER_SENTENCE in prompt
    assert any(r.getMessage().startswith(f"curiosity_replica_plan_used run={run_id}") for r in caplog.records)


@pytest.mark.parametrize("answer,status,reason", [
    (lambda req: TimeoutError("RPC timeout"), "timeout", "timeout"),
    (lambda req: ConnectionError("bus down"), "timeout", "request_failed: ConnectionError"),
    (_ok(["p7", "p3", "nope"]), "invalid:not_a_candidate:3", "not_a_candidate:3"),
    (_ok(["p7", "p10", "p3"]), "invalid:not_a_candidate:2", "not_a_candidate:2"),  # a stale prior
    (_ok(["p7", "p3"]), "invalid:too_few_picks:2", "too_few_picks:2"),
    (lambda req: CuriosityOfferPlanResultV1(run_id=req.run_id, plan_id="x", status="abstain",
                                            latency_ms=5).model_dump(mode="json"), "abstain", "abstain"),
    (lambda req: CuriosityOfferPlanResultV1(run_id=req.run_id, plan_id="x", status="unfunded",
                                            latency_ms=5).model_dump(mode="json"), "unfunded", "unfunded"),
    (lambda req: {"_kind": "something.else"}, "invalid:malformed_reply", None),
    (lambda req: CuriosityOfferPlanResultV1(run_id="another-run", plan_id="x", status="abstain",
                                            latency_ms=5).model_dump(mode="json"), "invalid:malformed_reply", None),
])
def test_every_fallback_keeps_the_rule_order_and_records_why(answer, status, reason, caplog):
    caplog.set_level(logging.INFO, logger="orion-hub.curiosity_replica_plan")
    loop, bus, conn, run_id, decision = _run(answer)
    plan = conn.plans[run_id]
    assert plan["status"] == status and plan["fallback_reason"]
    if reason is not None:
        assert plan["fallback_reason"] == reason
    assert decision["arm"] == ARM_REPLICA_ORDER  # intention to treat: still a replica-arm run
    # The rule order: the uncertainty order's ordered slots (coverage slots rotate with the run id).
    assert _ordered(decision) == ["p0", "p1", "p6", "p2", "p3", "p7"]
    assert WHY not in loop.seen_prompt
    assert any(r.getMessage().startswith(f"curiosity_replica_fallback run={run_id} reason=") for r in caplog.records)


def test_rule_arm_runs_never_ask_for_a_plan():
    loop, bus, conn, run_id, decision = _run(_ok(["p7", "p3", "p9"]), replica_arm_enabled=False)
    assert bus.requests == [] and conn.plans == {} and decision["arm"] == ARM_UNCERTAINTY_ORDER


def test_too_few_fresh_priors_means_no_request_and_a_recorded_fallback():
    from test_curiosity_investigation import _graph_loop

    bus = _PlanBus(_ok(["a", "b", "c"]))
    conn = _PlanConn()
    few = [_state("p0", 0.5, 0), _state("p1", 0.6, 0)]
    reader = _reader(few)
    loop = _graph_loop(bus, reader=reader, conn=conn, replica_arm_enabled=True, replica_arm_propensity=1.0)
    _moves_during_turn(loop, reader, lambda rid: few)
    assert asyncio.run(loop.tick()) is None
    [plan] = conn.plans.values()
    assert bus.requests == [] and plan["status"] == "invalid:no_candidates"
    assert plan["fallback_reason"].startswith("no_candidates: 2 fresh live priors")


def test_the_three_arm_layout_is_the_same_on_every_arm():
    """Same number of priors, same coverage slots, same stale bucket: only the top differs."""
    _, _, _, _, planned = _run(_ok(["p7", "p3", "p9"]))
    _, _, _, _, rule = _run(_ok([]), replica_arm_enabled=False)
    assert len(planned["offered"]) == len(rule["offered"])
    assert [o["slot"] for o in planned["offered"]] == [o["slot"] for o in rule["offered"]]
    assert planned["stale_offered"] == rule["stale_offered"]


def test_the_live_propensities_give_three_equal_arms():
    """HUB_CURIOSITY_VALUE_ORDER_PROPENSITY = HUB_CURIOSITY_REPLICA_ARM_PROPENSITY = 0.3333 (.env_example)."""
    from app.settings import Settings

    s = Settings()
    assert s.HUB_CURIOSITY_REPLICA_ARM_ENABLED is True
    assert s.HUB_CURIOSITY_VALUE_ORDER_PROPENSITY == s.HUB_CURIOSITY_REPLICA_ARM_PROPENSITY == 0.3333
    assert s.HUB_CURIOSITY_REPLICA_PLAN_DEADLINE_SEC == 90.0
    counts = {ARM_VALUE_ORDER: 0, ARM_REPLICA_ORDER: 0, ARM_UNCERTAINTY_ORDER: 0}
    for i in range(30000):
        d = draw_offer_arm(f"{i:012x}", value_enabled=True, value_propensity=0.3333, replica_enabled=True,
                           replica_propensity=0.3333)
        counts[d.arm] += 1
    assert d.propensities[ARM_UNCERTAINTY_ORDER] == pytest.approx(0.3334)
    for n in counts.values():
        assert abs(n / 30000 - 1 / 3) < 0.01


def test_a_pending_focus_rides_on_the_request_and_prime_is_recognised():
    from datetime import datetime, timezone

    from orion.curiosity.focus import PENDING_SQL
    from scripts.curiosity_replica_plan import _pending_focus

    class _Pool:
        def __init__(self, row):
            self.row = row

        def acquire(self):
            pool = self

            class _C:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    return False

                async def fetchrow(self, sql, *args):
                    assert sql == PENDING_SQL
                    return pool.row

            return _C()

    from collections import defaultdict

    def row(**kw):  # every column the focus reader reads; unset ones are NULL
        return defaultdict(lambda: None, {**base, **kw})

    base = {"focus_id": "f1", "question": "what heats the cabinet?", "why": "w", "origin": "self", "status": "pending",
            "created_at": datetime(2026, 11, 4, tzinfo=timezone.utc), "chat_correlation_id": "prime:abc"}
    got = asyncio.run(_pending_focus(_Pool(row()), enabled=True))
    assert got.from_prime and got.origin == "self"
    got = asyncio.run(_pending_focus(_Pool(row(origin="juniper", chat_correlation_id="c1")), enabled=True))
    assert got.origin == "juniper" and not got.from_prime
    assert asyncio.run(_pending_focus(_Pool(row()), enabled=False)) is None
