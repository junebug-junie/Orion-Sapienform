"""Attention patch 3c: reading stage 1 binds (spec 2026-11-07 section 4; flag ORION_PHYSIOLOGY_BID_READING).

on: stage 1 bids at its first resource request and obeys; a price / sated deferral holds it at most
2,900 s after its request (under Hub's 3,500 s reclaim), then it asks the pool anyway; a capacity
deferral worth its price queues at the pool. Stage 2 never bids. shadow: patch 3b's value-only bid."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

from app.run_bids import (  # noqa: E402
    READING_BINDING_PRODUCER,
    READING_FAMILY,
    READING_HOLD_END_SEC,
    VALUE_PRODUCER,
    RunBidder,
    RunValue,
    reading_binds,
)
from orion.physiology.binding import BINDING_PRODUCERS  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402

FAMS, _ = load_families(ROOT / "config/physiology/families.yaml")
T0 = datetime(2026, 11, 11, 13, 0, tzinfo=timezone.utc)


class _Decision:
    def __init__(self, want, outcome, reason, net, retry=None):
        self.decision_id, self.want_id, self.family = f"{want.want_id}@x", want.want_id, want.family
        self.outcome, self.reason, self.net, self.retry_after_sec = outcome, reason, net, retry
        self.binding_constraint = "lane:agent" if reason == "capacity_reserved" else None
        self.consent = None


class _Market:
    def __init__(self, outcome="deferred", reason="price", net=-0.2, retry=600.0):
        self.families = FAMS
        self.answer = (outcome, reason, net, retry)
        self.bids, self.released, self.opened = [], [], []

    async def bid(self, want, now=None, *, reserve_at_open=False):
        self.bids.append(want)
        return _Decision(want, *self.answer)

    def release(self, want_id):
        self.released.append(want_id)
        return 1

    def mark_opened(self, want_id, at=None):
        self.opened.append(want_id)
        return 1


def _reading(stage, run_id, seed="seed-1", **brief):
    return {"run_id": run_id, "workflow": "reading.turn", "correlation_id": f"corr-{run_id}",
            "requested_at": T0.isoformat(), "brief": {"seed_id": seed, "stage": stage, **brief}, "admission": {}}


def _bidder(m, on=True):
    return RunBidder(enabled=True, market_getter=lambda: m, reading_binding_enabled=on)


def test_the_binding_producer_is_obeyed_by_the_market():
    assert READING_BINDING_PRODUCER in BINDING_PRODUCERS


def test_only_stage_1_binds_and_never_an_operators_forced_read():
    assert reading_binds(_reading(1, "r"), True)
    assert not reading_binds(_reading(1, "r"), False)
    assert not reading_binds(_reading(2, "r"), True)
    assert not reading_binds(_reading(1, "r", operator_forced=True), True)


def test_shadow_keeps_patch_3b_value_only_bid_and_bind_does_nothing():
    async def go():
        m = _Market(outcome="funded", reason="net_positive", net=0.8)
        b = _bidder(m, on=False)
        await b.on_submit(_reading(1, "s1"))
        assert [w.producer for w in m.bids] == [VALUE_PRODUCER]
        assert await b.bind(_reading(1, "s1"), T0) is None
    asyncio.run(go())


def test_on_a_price_deferral_holds_the_read_and_it_rebids():
    async def go():
        m = _Market(outcome="deferred", reason="price", net=-0.2, retry=600.0)
        b = _bidder(m)
        assert b.on_submit(_reading(1, "s1")) is None            # its binding bid is its one bid
        v = await b.bind(_reading(1, "s1"), T0 + timedelta(seconds=30))
        assert v.admit is False and v.why == "deferred:price"
        assert v.retry_at == T0 + timedelta(seconds=30 + 600)
        assert [w.producer for w in m.bids] == [READING_BINDING_PRODUCER] and m.bids[0].family == READING_FAMILY
        assert m.released == ["s1"]                              # a held want reserves nothing
    asyncio.run(go())


def test_a_held_read_asks_the_pool_anyway_at_2900_s_before_hubs_reclaim():
    async def go():
        m = _Market(outcome="deferred", reason="sated", net=0.0, retry=3600.0)
        b = _bidder(m)
        early = await b.bind(_reading(1, "s1"), T0 + timedelta(seconds=2500))
        assert early.admit is False and early.retry_at == T0 + timedelta(seconds=READING_HOLD_END_SEC)
        late = await b.bind(_reading(1, "s1"), T0 + timedelta(seconds=READING_HOLD_END_SEC))
        assert late.admit is True and late.why == "window_end"
        assert READING_HOLD_END_SEC < 3500
    asyncio.run(go())


def test_a_capacity_deferral_worth_its_price_queues_at_the_pool_and_carries_its_net():
    async def go():
        m = _Market(outcome="deferred", reason="capacity_reserved", net=0.74)
        b = _bidder(m)
        v = await b.bind(_reading(1, "s1"), T0)
        assert v.admit is True and v.why == "queued_at_pool"
        assert await b.value_for(_reading(1, "s1")) == RunValue("s1", 0.74, READING_FAMILY)
        assert await b.value_for(_reading(2, "s2")) == RunValue("s1", 0.74, READING_FAMILY)   # stage 2 rides on it
    asyncio.run(go())


def test_stage_2_never_bids_even_with_binding_on():
    async def go():
        m = _Market(outcome="funded", reason="net_positive", net=0.8)
        b = _bidder(m)
        assert b.on_submit(_reading(2, "s2")) is None
        assert await b.bind(_reading(2, "s2"), T0) is None
        assert m.bids == []
    asyncio.run(go())


def test_funded_read_keeps_its_reservation_until_the_terminal():
    async def go():
        m = _Market(outcome="funded", reason="net_positive", net=0.8)
        b = _bidder(m)
        v = await b.bind(_reading(1, "s1"), T0)
        assert v.admit is True and v.why == "funded" and m.released == []
        b.on_grant("s1", T0 + timedelta(minutes=1))
        b.on_terminal("s1")
        assert m.opened == ["s1"] and m.released == ["s1"]
    asyncio.run(go())
