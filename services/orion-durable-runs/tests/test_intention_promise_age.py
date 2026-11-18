"""#3640: a take-on's carry run bids at the AGE OF THE PROMISE (Juniper 2026-11-18), not at its own bid age.

What must hold: the want opens at ``brief.promised_at`` (else ``requested_at``); with the live families.yaml the market's
hunger and rank come from the promise's age, so a promise made N hours before the bid ranks by N; a fresh promise
still ranks ~0; the hold end still counts from the run's own request; the 11-13 run (want_ledger
``intention-d8db9d7b1bba4cf7``, rank 0.00 live) replays to the rank its promise's age gives."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.run_bids import RunBidder, hold_end, want_for_run, want_opened_at
from orion.physiology.appetite import ActRecord, build_family_appetite, hunger, rank_value
from orion.physiology.families import load_families
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.intention import INTENTION_CARRY_WORKFLOW, INTENTION_FAMILY, intention_run_id

ROOT = Path(__file__).resolve().parents[3]
FAMS, _ = load_families(ROOT / "config/physiology/families.yaml")
CFG = FAMS.get(INTENTION_FAMILY)
CID = "d8db9d7b1bba4cf7"


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _req(*, requested_at: datetime, promised_at, horizon: datetime):
    brief = {"intention": "intention.carry", "concern_id": CID, "source": "turn_promise", "prime_act_id": "act_1",
             "prompt": f"prime_concern:{CID}\n\nI am carrying out something I told Juniper I would do.",
             "route": "chat_thread", "route_target": "9c91d646-cbd0-4977-8cdd-d2c82045c7f9",
             "quote_ref": "harness_turn_trace:a129b72f-e698-425a-a435-3388cd7cd759#promises_made[0]",
             "horizon_at": horizon.isoformat(), "retrieval_query": "a lazy browse", "timeout_sec": 1800.0}
    if promised_at is not None:
        brief["promised_at"] = promised_at.isoformat()
    return {"run_id": intention_run_id(CID), "correlation_id": "c-1", "workflow": INTENTION_CARRY_WORKFLOW,
            "requested_at": requested_at.isoformat(), "brief": brief,
            "admission": {"resource": "llm.route.agent", "preferred_lane": "agent", "priority": "background",
                          "deadline_at": horizon.isoformat()}}


def _market_rank(want, now, acts=()):
    """What Choice computes for the want's value (orion/physiology/choice.py: hunger, then rank_value)."""
    app = build_family_appetite(CFG, acts, now)
    h = hunger(app, now, opened_at=want.opened_at, seed_sec=3600.0 * 5, seed_lane="agent-gpu2")
    rank, basis, _ = rank_value(app, h.intensity, want.outcome_value)
    return h, rank, basis


def test_the_request_with_promised_at_validates():
    t = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
    req = DurableRunRequestV1.model_validate(_req(requested_at=t, promised_at=t - timedelta(hours=3),
                                                  horizon=t + timedelta(hours=60)))
    assert req.brief.promised_at == t - timedelta(hours=3)


def test_the_want_opens_at_the_promise_else_at_the_request():
    t = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
    p = t - timedelta(hours=7)
    assert want_for_run(_req(requested_at=t, promised_at=p, horizon=t + timedelta(hours=60)), 1).opened_at == p
    assert want_opened_at(_req(requested_at=t, promised_at=None, horizon=t + timedelta(hours=60))) == t
    assert want_opened_at({"requested_at": t.isoformat(), "brief": {"promised_at": "garbage"}}) == t


@pytest.mark.parametrize("hours", [0.5, 3, 12, 36])
def test_a_promise_made_n_hours_before_the_bid_ranks_by_n_not_by_bid_age(hours):
    bid = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
    requested = bid - timedelta(seconds=5)                              # the run was ordered 5 s ago
    # Another promise's carry closed 10 minutes ago: it does not reset this one.
    acts = [ActRecord(act_id="a0", family=INTENTION_FAMILY, opened_at=bid - timedelta(hours=2),
                      closed_at=bid - timedelta(minutes=10))]
    want = want_for_run(_req(requested_at=requested, promised_at=bid - timedelta(hours=hours),
                             horizon=bid + timedelta(hours=72 - hours)), CFG.tier)
    h, rank, basis = _market_rank(want, bid, acts)
    assert h.basis == "since_opened" and h.unfunded_sec == pytest.approx(hours * 3600)
    assert basis == "want_opened:ungraded" and rank == pytest.approx(1 - math.exp(-hours * 3600 / CFG.deferrable_window_s))


def test_a_fresh_promise_still_ranks_near_zero():
    bid = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
    want = want_for_run(_req(requested_at=bid - timedelta(seconds=5), promised_at=bid - timedelta(seconds=60),
                             horizon=bid + timedelta(hours=72)), CFG.tier)
    assert _market_rank(want, bid)[1] < 1e-3


def test_the_hold_end_still_counts_from_the_runs_own_request():
    t = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
    req = _req(requested_at=t, promised_at=t - timedelta(hours=30), horizon=t + timedelta(hours=42))
    want = want_for_run(req, 1)
    assert hold_end(want, CFG.deferrable_window_s, 1800.0, held_from=t) == t + timedelta(seconds=CFG.deferrable_window_s)

    class _M:
        families = FAMS

        def __init__(self):
            self.bids = []

        async def bid(self, w, now=None, *, reserve_at_open=False, object_text=None):
            self.bids.append(w)
            return SimpleNamespace(decision_id="d", outcome="deferred", reason="price", net=-0.1, charge=0.3,
                                   retry_after_sec=900.0, binding_constraint="lane:agent", lane="agent",
                                   cost_frac={}, why="why", state_tick_id="tick")

        def release(self, want_id):
            return 1

    async def go():
        m = _M()
        b = RunBidder(enabled=True, binding_enabled=True, market_getter=lambda: m)
        v = await b.bind(req, t + timedelta(hours=1))
        # Held (not released at promise + 24 h, which is already past): the hold runs from the request.
        assert not v.admit and m.bids[0].opened_at == t - timedelta(hours=30)
    asyncio.run(go())


def test_replay_of_the_11_13_carry_bid():
    """Live: want_ledger intention-d8db9d7b1bba4cf7 decided 2026-11-13 19:47:13.669909Z, rank 0.00 (hunger 5 s
    since_opened), net 0.000062. The request as stored (durable_admission_runs), plus the promise's time
    (harness_turn_trace a129b72f-... created_at 2026-11-12 07:08:40.440369Z, the brief's quote_ref)."""
    decided = _ts("2026-11-13T19:47:13.669909Z")
    req = _req(requested_at=_ts("2026-11-13T19:47:08.522865Z"), promised_at=_ts("2026-11-12T07:08:40.440369Z"),
               horizon=_ts("2026-11-15T07:08:40.440369Z"))
    old = build_family_appetite(CFG, (), decided)
    before = rank_value(old, hunger(old, decided, opened_at=_ts("2026-11-13T19:47:08.522865Z")).intensity, None)[0]
    assert before < 1e-3                                     # what the live decision saw (no lane seed then)
    h, rank, _ = _market_rank(want_for_run(req, CFG.tier), decided)
    assert h.unfunded_sec == pytest.approx(131913.2, abs=1.0)  # 36.6 h old at the bid
    assert rank == pytest.approx(0.783, abs=0.001)
    # net = weight x rank - charge: the live weight 1.03 and charge 0.000 now give +0.81, not +0.00006.
    assert 1.03 * rank > 0.8
