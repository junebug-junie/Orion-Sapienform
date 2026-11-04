"""Each finished read posts one event want (attention as the want competition, patch 2). No network."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from orion.physiology.event_wants import event_want_id
from scripts.reading_event_wants import FAMILY, READS_SQL, ReadingEventWants, plan_reads

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def _rows(recent_links=(2, 0)):
    # 40 older reads over the last 20 days, mostly no link; then the recent ones (inside the 3 h lookback).
    rows = [{"seed_id": f"old{k}", "url": f"https://x/{k}", "stage2_completed_at": NOW - timedelta(days=20) + timedelta(hours=10 * k),
             "links": 1 if k % 5 == 0 else 0} for k in range(40)]
    rows += [{"seed_id": f"new{i}", "url": f"https://n/{i}", "stage2_completed_at": NOW - timedelta(minutes=30 + i),
              "links": n} for i, n in enumerate(recent_links)]
    return rows


class _Conn:
    def __init__(self, state):
        self.state = state

    async def fetch(self, sql, *args):
        if sql == READS_SQL:
            return self.state["rows"]
        assert "want_ledger" in sql
        return [{"want_id": w} for w in self.state["ledger"] if w in args[0]]


class _Pool:
    def __init__(self, state):
        self.state = state

    def acquire(self):
        conn = _Conn(self.state)

        class _CM:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *a):
                return False

        return _CM()


def test_plan_ranks_each_recent_read_against_its_30_days():
    plan = plan_reads(_rows(), now=NOW, lookback=timedelta(hours=3))
    by = {p["seed_id"]: p for p in plan}
    assert set(by) == {"new0", "new1"}
    assert by["new1"]["salience"] == 0.0                       # no link: the rest point
    # new0 made 2 links; every read before it (the 40 older ones and new1, a minute earlier) made fewer.
    assert by["new0"]["salience"] == 1.0


def _loop(state, decision="funded"):
    bids = []

    async def bid(want):
        bids.append(want)
        return None if decision is None else SimpleNamespace(outcome=decision, reason="net_positive", value_rank=0.8)

    loop = ReadingEventWants(enabled=True, pool_provider=lambda: _Pool(state), bid=bid, clock=lambda: NOW)
    return loop, bids


def test_one_event_want_per_read_with_its_event_and_salience():
    state = {"rows": _rows(), "ledger": set()}
    loop, bids = _loop(state)
    summary = asyncio.run(loop.tick())
    assert summary["posted"] == 2
    w = next(b for b in bids if b.event_ref.endswith("new0"))
    assert w.family == FAMILY and w.verb == "look" and w.tier == 2 and w.salience == 1.0
    assert w.event_ref == "world_pulse_read_seed:new0" and w.want_id == event_want_id("reading", "new0")
    assert w.opened_at == NOW - timedelta(minutes=30) and w.object_ref == "https://n/0"
    asyncio.run(loop.tick())
    assert len(bids) == 2                                      # never posted twice


def test_a_read_already_in_the_ledger_is_not_reposted_after_a_restart():
    state = {"rows": _rows(), "ledger": {event_want_id("reading", "new0"), event_want_id("reading", "new1")}}
    loop, bids = _loop(state)
    assert asyncio.run(loop.tick())["already"] == 2 and bids == []


def test_no_answer_is_retried_next_tick():
    state = {"rows": _rows(), "ledger": set()}
    loop, bids = _loop(state, decision=None)
    asyncio.run(loop.tick())
    asyncio.run(loop.tick())
    assert len(bids) == 4


def test_too_little_history_posts_nothing():
    state = {"rows": [r for r in _rows() if r["seed_id"].startswith("new")], "ledger": set()}
    loop, bids = _loop(state)
    assert asyncio.run(loop.tick())["no_history"] == 2 and bids == []


def test_off_means_no_task():
    loop = ReadingEventWants(enabled=False, pool_provider=lambda: None)
    asyncio.run(loop.start())
    assert loop._task is None


def test_an_unpriced_decision_is_not_final_and_is_bid_again():
    """Review finding 2: physiology_unavailable is not a decision about the event; the ledger row it
    left does not count as posted either (POSTED_SQL skips it)."""
    from scripts.reading_event_wants import POSTED_SQL

    state = {"rows": _rows(), "ledger": set()}
    bids = []

    async def bid(want):
        bids.append(want)
        return SimpleNamespace(outcome="deferred", reason="physiology_unavailable", value_rank=None)

    loop = ReadingEventWants(enabled=True, pool_provider=lambda: _Pool(state), bid=bid, clock=lambda: NOW)
    asyncio.run(loop.tick())
    asyncio.run(loop.tick())
    assert len(bids) == 4 and "reason <> 'physiology_unavailable'" in POSTED_SQL
