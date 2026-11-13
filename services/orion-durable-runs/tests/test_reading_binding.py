"""Attention patch 3c: reading stage 1 binds (spec 2026-11-07 section 4; flag ORION_PHYSIOLOGY_BID_READING).

on: stage 1 bids at its first resource request and obeys; a price / sated deferral holds it at most
2,900 s after its request (a held read occupies one of Hub's 3 in-flight slots), then it asks the pool anyway; a capacity
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


def test_a_held_read_asks_the_pool_anyway_at_its_2900_s_hold_end():
    async def go():
        m = _Market(outcome="deferred", reason="sated", net=0.0, retry=3600.0)
        b = _bidder(m)
        early = await b.bind(_reading(1, "s1"), T0 + timedelta(seconds=2500))
        assert early.admit is False and early.retry_at == T0 + timedelta(seconds=READING_HOLD_END_SEC)
        late = await b.bind(_reading(1, "s1"), T0 + timedelta(seconds=READING_HOLD_END_SEC))
        assert late.admit is True and late.why == "window_end"
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


# --- the runtime path (review finding 3) -----------------------------------------------------------

def test_the_settings_flag_reaches_the_bidder():
    from app.admission_runtime import make_bidder
    from app.settings import Settings

    assert "physiology_bid_reading" in Settings.model_fields                  # a rename fails here
    assert Settings.model_fields["physiology_bid_reading"].default == "on"
    for mode, expect in (("on", True), ("shadow", False), ("typo", False)):
        s = Settings.model_construct(physiology_bid_reading=mode)
        assert make_bidder(s).reading_binding_enabled is expect


class _Store:
    def __init__(self, request):
        self.request, self.events = request, []

    async def get_run(self, run_id):
        return {"request": self.request}

    async def record_event(self, run_id, kind, payload, event_id=None):
        self.events.append(kind)


class _Holds:
    def __init__(self):
        self.calls = []

    def placement(self, admission):
        return "agent", "background", 0

    async def acquire(self, run_id, request_id, admission, *, correlation_id, **kw):
        from orion.schemas.gpu_pool import GpuLeaseReplyV1
        self.calls.append(kw)
        return GpuLeaseReplyV1(status="queued", lease_id="L1", position=1)


def test_a_held_read_rebids_on_wake_with_the_same_hold_end_then_asks_the_pool_with_its_net():
    from app.admission_runtime import AdmissionRuntime

    m = _Market(outcome="deferred", reason="price", net=-0.2, retry=600.0)
    rt = object.__new__(AdmissionRuntime)
    rt.store, rt.holds, rt.bids = _Store(_reading(1, "s1")), _Holds(), _bidder(m)
    t = [T0 + timedelta(seconds=60)]
    rt.now = lambda: t[0]
    state = {"run_id": "s1", "correlation_id": "corr-s1"}
    held = asyncio.run(rt.register(state))
    assert held["status"] == "waiting_resource" and held["hold_seq"] == 0 and held["hold"]["economy"] == "deferred:price"
    assert rt.holds.calls == [] and "run.economy_held" in rt.store.events          # no pool request while held
    # Woken at retry_at: it bids again; the market now funds it with a capacity deferral worth its price.
    t[0] = datetime.fromisoformat(held["hold"]["retry_at"])
    m.answer = ("deferred", "capacity_reserved", 0.7, None)
    out = asyncio.run(rt.register({**state, **held}))
    assert out["status"] == "waiting_resource" and len(m.bids) == 2
    assert rt.holds.calls == [{"value": RunValue("s1", 0.7, READING_FAMILY)}]
    # The hold end is fixed by the stored request (requested_at), not by when it re-bids.
    m.answer = ("deferred", "price", -0.2, 600.0)
    v = asyncio.run(rt.bids.bind(_reading(1, "s1"), T0 + timedelta(seconds=READING_HOLD_END_SEC)))
    assert v.admit is True and v.why == "window_end"
