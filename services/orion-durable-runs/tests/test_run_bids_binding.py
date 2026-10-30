"""Body economy Patch D: curiosity investigate's bid BINDS (app/run_bids.py bind_verdict / RunBidder.bind,
app/admission_runtime.py _economy_gate).

Pure lane: the verdict rule (funded / capacity worth its price -> ask the pool; price / sated / net<=0
-> hold and re-bid; no answer -> ask the pool; never held past the deferrable window), the bidder's
binding call (only curiosity investigate, only with the flag on; a broken market never holds a run),
and the flag shipping on.

Real Postgres + real in-process GPU pool (ORION_ADMISSION_TEST_DSN): a held run never asks the pool,
records run.economy_held with the market's reason, and runs once the market funds it.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

from app.run_bids import (  # noqa: E402
    BINDING_FAMILIES, BINDING_PRODUCER, HOLD_DEADLINE_MARGIN_SEC, HOLD_EVENT, RELEASE_EVENT, REBID_DEFAULT_SEC, REBID_MAX_SEC,
    REBID_MIN_SEC, RunBidder, bind_verdict, binding_family, hold_end, want_for_run,
)
from orion.physiology.families import load_families  # noqa: E402

FAMS, _ = load_families(ROOT / "config/physiology/families.yaml")
T0 = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
END = T0 + timedelta(hours=6)


def _dec(outcome="deferred", reason="price", net=-0.2, retry=900.0, binding="lane:agent", **kw):
    return SimpleNamespace(decision_id=f"d@{reason}", outcome=outcome, reason=reason, net=net, charge=0.3,
                           retry_after_sec=retry, binding_constraint=binding, lane="agent",
                           cost_frac={"lane:agent": 1.0}, why=f"why {reason}", state_tick_id="tick", **kw)


def _req(workflow="curiosity.investigate", line="investigate", run_id="run-1", requested_at=T0, **admission):
    brief = {"prompt": "p", "session_id": "orion_curiosity", "line": line}
    return {"run_id": run_id, "correlation_id": "6f1e2a3b-0000-4000-8000-000000000001", "workflow": workflow,
            "requested_at": requested_at.isoformat(), "brief": brief, "admission": {"deadline_at": None, **admission}}


# --- the verdict rule ------------------------------------------------------------------------------

def test_funded_and_a_capacity_deferral_worth_its_price_ask_the_pool():
    assert bind_verdict(_dec("funded", "net_positive", net=0.5), T0, END).why == "funded"
    v = bind_verdict(_dec(reason="capacity_reserved", net=0.8, retry=3600.0), T0, END)
    assert v.admit and v.why == "queued_at_pool"          # the pool's queue is the queue, never a refusal
    # The NOT-shipped policy (replay comparison): every capacity deferral holds.
    held = bind_verdict(_dec(reason="capacity_reserved", net=0.8, retry=3600.0), T0, END, capacity_queues_at_pool=False)
    assert not held.admit and held.retry_at == T0 + timedelta(seconds=3600)


@pytest.mark.parametrize("reason,net", [("price", -0.2), ("sated", 0.0), ("capacity_reserved", 0.0),
                                        ("capacity_reserved", -0.1), ("no_acceptable_lane", 0.4)])
def test_other_deferrals_hold_and_carry_the_markets_reason(reason, net):
    v = bind_verdict(_dec(reason=reason, net=net), T0, END)
    assert not v.admit and v.why == f"deferred:{reason}" and v.retry_at == T0 + timedelta(seconds=900)
    assert v.detail["reason"] == reason and v.detail["net"] == net and v.detail["charge"] == 0.3
    assert v.detail["binding_constraint"] == "lane:agent" and v.detail["why"] == f"why {reason}"


@pytest.mark.parametrize("retry,expect", [(None, REBID_DEFAULT_SEC), (5.0, REBID_MIN_SEC), (99999.0, REBID_MAX_SEC),
                                          (float("nan"), REBID_DEFAULT_SEC), (float("inf"), REBID_DEFAULT_SEC)])
def test_rebid_time_respects_retry_after_within_bounds(retry, expect):
    assert bind_verdict(_dec(retry=retry), T0, END).retry_at == T0 + timedelta(seconds=expect)


def test_never_held_past_the_window_and_never_dropped():
    near = END - timedelta(seconds=300)
    assert bind_verdict(_dec(retry=3000.0), near, END).retry_at == END     # last bid lands at the window's end
    v = bind_verdict(_dec(), END, END)
    assert v.admit and v.why == "window_end"                              # then it asks the pool anyway
    assert bind_verdict(_dec(), T0, None).why == "no_answer:unbounded_hold"   # no bound -> never hold


@pytest.mark.parametrize("reason", ["physiology_unavailable", "unknown_family", "family_never_bids"])
def test_no_market_answer_asks_the_pool_as_before(reason):
    v = bind_verdict(_dec(reason=reason, net=None), T0, END)
    assert v.admit and v.why == f"no_answer:{reason}"
    assert bind_verdict(None, T0, END).why == "no_answer:no_decision"


def test_hold_end_is_the_deferrable_window_or_before_the_runs_own_deadline():
    w = want_for_run(_req(), 2)
    assert hold_end(w, 21600.0) == T0 + timedelta(hours=6)
    deadline = T0 + timedelta(hours=2)
    w2 = want_for_run(_req(deadline_at=deadline.isoformat()), 2)
    assert hold_end(w2, 21600.0) == deadline - timedelta(seconds=HOLD_DEADLINE_MARGIN_SEC)
    assert hold_end(w.model_copy(update={"opened_at": None}), 21600.0) is None


def test_only_curiosity_investigate_binds_and_only_with_the_flag_on():
    assert BINDING_FAMILIES == frozenset({"durable.curiosity_investigate"})
    assert FAMS.get("durable.curiosity_investigate").deferrable_window_s > 0
    assert binding_family(_req(), True) == "durable.curiosity_investigate"
    assert binding_family(_req(), False) is None
    assert binding_family(_req(line="self_inquiry"), True) is None
    assert binding_family(_req("self_sense_eval", None), True) is None
    assert binding_family({**_req(), "brief": {"urgent": {"subject": "cabinet_ac"}}}, True) is None   # a reflex
    # Juniper's "Run now" (Hub tick(force=True)) is the operator's call: never held.
    forced = _req()
    forced["brief"] = {**forced["brief"], "operator_forced": True}
    assert binding_family(forced, True) is None


def test_unknown_flag_value_is_shadow_not_a_crash():
    from app.admission_runtime import _bind_mode

    assert _bind_mode("on") == "on" and _bind_mode(" ON ") == "on" and _bind_mode("shadow") == "shadow"
    assert _bind_mode("true") == "shadow" and _bind_mode(None) == "shadow"


# --- the bidder's binding call ----------------------------------------------------------------------

class _Market:
    def __init__(self, decision=None, *, fail=False, delay=0.0):
        self.families = FAMS
        self.decision, self.fail, self.delay = decision, fail, delay
        self.bids, self.released, self.opened = [], [], []

    async def bid(self, want, now=None, *, reserve_at_open=False):
        assert reserve_at_open is True
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("market down")
        self.bids.append((want, now))
        return self.decision

    def release(self, want_id):
        self.released.append(want_id)
        return 1

    def mark_opened(self, want_id, at=None):
        self.opened.append(want_id)
        return 1


def test_bind_holds_on_price_admits_on_funding_and_reserves_only_when_funded():
    async def go():
        price = _Market(_dec())
        v = await RunBidder(enabled=True, binding_enabled=True, market_getter=lambda: price).bind(_req(), T0)
        assert not v.admit and price.released == ["run-1"] and price.bids[0][1] == T0
        # The ledger tells an obeyed bid from a shadow one by producer (shadow is always true there).
        assert price.bids[0][0].producer == BINDING_PRODUCER and "bid_ms" in v.detail
        funded = _Market(_dec("funded", "net_positive", net=0.5))
        b = RunBidder(enabled=True, binding_enabled=True, market_getter=lambda: funded)
        v2 = await b.bind(_req(), T0)
        assert v2.admit and v2.why == "funded" and funded.released == []      # its pending reservation stays
        assert b.counts["bind_admitted"] == 1
        # ...until the run opens and ends (shadow flag off: binding alone still follows the run).
        b_off = RunBidder(enabled=False, binding_enabled=True, market_getter=lambda: funded)
        assert b_off.on_grant("run-1", T0) == 1 and b_off.on_terminal("run-1") == 1
    asyncio.run(go())


def test_a_broken_slow_or_absent_market_never_holds_a_run():
    async def go():
        for m, why in ((_Market(fail=True), "no_answer:no_decision"), (_Market(delay=1.0), "no_answer:no_decision")):
            b = RunBidder(enabled=True, binding_enabled=True, market_getter=lambda m=m: m, timeout_sec=0.05)
            v = await b.bind(_req(), T0)
            assert v.admit and v.why == why and m.released == ["run-1"]
        v = await RunBidder(enabled=True, binding_enabled=True, market_getter=lambda: None).bind(_req(), T0)
        assert v.admit and v.why == "no_answer:market_off"
        no_cfg = _Market()
        no_cfg.families = None
        assert (await RunBidder(enabled=True, binding_enabled=True, market_getter=lambda: no_cfg).bind(_req(), T0)).admit
    asyncio.run(go())


def test_bind_is_none_for_other_families_and_with_the_flag_off():
    async def go():
        m = _Market(_dec())
        on = RunBidder(enabled=True, binding_enabled=True, market_getter=lambda: m)
        assert await on.bind(_req(line="self_inquiry"), T0) is None
        assert await RunBidder(enabled=True, binding_enabled=False, market_getter=lambda: m).bind(_req(), T0) is None
        assert m.bids == []
    asyncio.run(go())


def test_binding_curiosity_does_not_also_shadow_bid_at_submit_and_shadow_is_unchanged_when_off():
    async def go():
        m = _Market(_dec("funded", "net_positive", net=0.5))
        on = RunBidder(enabled=True, binding_enabled=True, market_getter=lambda: m)
        assert on.on_submit(_req()) is None                              # curiosity: bound, no shadow bid
        t = on.on_submit(_req(line="self_inquiry", run_id="run-2"))      # siblings: still shadow
        await t
        off = RunBidder(enabled=True, binding_enabled=False, market_getter=lambda: m)
        await off.on_submit(_req(run_id="run-3"))                        # flag shadow: C3a, as before
        return m
    m = asyncio.run(go())
    assert [w.want_id for w, _ in m.bids] == ["run-2", "run-3"]


def test_flag_ships_on_in_settings_env_example_and_compose():
    from app.settings import Settings

    assert Settings.model_fields["physiology_bid_curiosity_investigate"].default == "on"
    svc = Path(__file__).resolve().parents[1]
    assert "\nORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE=on\n" in (svc / ".env_example").read_text()
    assert "ORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE=${ORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE:-on}" in (
        svc / "docker-compose.yml").read_text()


# --- real Postgres + real pool lane --------------------------------------------------------------

from test_admission_runtime_postgres import DSN, request, runtime, with_database  # noqa: E402
from pool_fixture import InProcessPool  # noqa: E402

pg = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


class _Switch(_Market):
    """Defers on price until ``fund`` is set."""
    fund = False

    async def bid(self, want, now=None, *, reserve_at_open=False):
        self.bids.append((want, now))
        return _dec("funded", "net_positive", net=0.5) if self.fund else _dec(retry=600.0)


@pg
def test_a_held_run_never_asks_the_pool_records_why_and_runs_once_funded():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        m = _Switch()
        rt = runtime(pool, saver, store, gpu=gpu, ORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE="on")
        assert rt.bids.binding_enabled
        rt.bids._market = lambda: m
        clock = [datetime.now(timezone.utc)]
        rt.now = lambda: clock[0]
        req = request("bind-held")
        receipt = await rt.submit(req)
        assert receipt["status"] == "waiting_resource" and rt.bids._tasks == set()   # no shadow bid at submit
        await rt._drive(await store.get_run(req.run_id))
        row = await store.get_run(req.run_id)
        assert row["terminal"] is None and len(m.bids) == 1
        hist = await store.history(req.run_id)
        held = [e for e in hist if e["event"] == HOLD_EVENT]
        assert len(held) == 1 and held[0]["detail"]["reason"] == "price" and held[0]["detail"]["net"] == -0.2
        assert held[0]["detail"]["binding_constraint"] == "lane:agent" and held[0]["detail"]["retry_at"]
        assert not any(e["event"] == "run.resource_granted" for e in hist)     # never asked the pool
        # Before retry_at: the driver leaves it alone (no second bid).
        await rt._drive(await store.get_run(req.run_id))
        assert len(m.bids) == 1
        # Still deferred after retry_at: a second held event, and no resumed / waiting rows for the cycle.
        clock[0] = clock[0] + timedelta(seconds=601)
        await rt._drive(await store.get_run(req.run_id))
        hist = await store.history(req.run_id)
        assert len(m.bids) == 2 and len([e for e in hist if e["event"] == HOLD_EVENT]) == 2
        assert not any(e["event"] in ("run.resumed", "run.waiting_resource") for e in hist)
        # After retry_at, funded: it asks the pool and runs.
        m.fund = True
        clock[0] = clock[0] + timedelta(seconds=601)
        for _ in range(3):
            await rt._drive(await store.get_run(req.run_id))
            if (await store.get_run(req.run_id))["terminal"]:
                break
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        hist = await store.history(req.run_id)
        assert [e["detail"]["verdict"] for e in hist if e["event"] == RELEASE_EVENT] == ["funded"]
        assert len(m.bids) == 3 and m.opened == [req.run_id] and m.released[-1] == req.run_id
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_flag_shadow_leaves_admission_exactly_as_before():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        m = _Switch()
        rt = runtime(pool, saver, store, gpu=gpu, ORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE="shadow")
        assert not rt.bids.binding_enabled
        rt.bids._market = lambda: m
        req = request("bind-shadow")
        await rt.submit(req)
        await asyncio.gather(*rt.bids._tasks)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"   # deferred in shadow, ran anyway
        assert not any(e["event"] in (HOLD_EVENT, RELEASE_EVENT) for e in await store.history(req.run_id))
        await rt.close()
    asyncio.run(with_database(scenario))
