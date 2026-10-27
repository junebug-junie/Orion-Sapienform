"""Body economy Patch C3a: durable-runs' own shadow bid site (app/run_bids.py).

Two lanes, as test_run_acts.py:
* pure: the want a run becomes (ids that join to grammar_acts), only want families bid, a duplicate
  submit bids once, a slow or broken market never reaches admission, open and release follow the run;
* real Postgres + the real in-process GPU pool (ORION_ADMISSION_TEST_DSN): a submitted run bids
  once, admission is unchanged, the reservation is re-anchored at the grant and released at the
  terminal -- also for a run refused before it ever ran.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

from app.run_bids import PRODUCER, RunBidder, want_for_run  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402

FAMS, _ = load_families(ROOT / "config/physiology/families.yaml")
T0 = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


def _req(workflow="curiosity.investigate", line="investigate", run_id="run-1", **admission):
    brief = {"prompt": "p", "session_id": "orion_curiosity", "line": line}
    return {"run_id": run_id, "correlation_id": "6f1e2a3b-0000-4000-8000-000000000001", "workflow": workflow,
            "requested_at": T0.isoformat(), "brief": brief, "admission": {"deadline_at": None, **admission}}


class _Decision:
    def __init__(self, want):
        self.decision_id = f"{want.want_id}@x"
        self.outcome, self.reason, self.binding_constraint = "funded", "net_positive", None


class _Market:
    def __init__(self, *, delay=0.0, fail=False):
        self.families = FAMS
        self.bids, self.opened, self.released = [], [], []
        self.delay, self.fail = delay, fail

    async def bid(self, want):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("market down")
        self.bids.append(want)
        return _Decision(want)

    def mark_opened(self, want_id, at=None):
        self.opened.append(want_id)
        return 1

    def release(self, want_id):
        self.released.append(want_id)
        return 1


def test_a_run_becomes_a_want_that_joins_its_act():
    deadline = T0 + timedelta(hours=2)
    w = want_for_run(_req(line="self_inquiry", deadline_at=deadline.isoformat()), 2)
    assert (w.want_id, w.family, w.verb, w.tier, w.producer) == ("run-1", "durable.self_inquiry", "ask", 2, PRODUCER)
    # The act's object_ref and correlation_id (app/run_acts.py): the join keys the shadow report uses.
    assert w.object_ref == "durable_run:run-1" and w.correlation_id == "6f1e2a3b-0000-4000-8000-000000000001"
    assert w.opened_at == T0 and w.window.latest == deadline
    assert want_for_run(_req(), 0) is None                                   # tier 0 never bids
    assert want_for_run(_req("reading.turn", None), 2) is None               # not a want family
    assert want_for_run({**_req(), "brief": {"urgent": {"subject": "cabinet_ac"}}}, 2) is None   # a reflex


@pytest.mark.parametrize("workflow,line,family", [
    ("curiosity.investigate", "investigate", "durable.curiosity_investigate"),
    ("curiosity.investigate", "self_inquiry", "durable.self_inquiry"),
    ("self_sense_eval", None, "durable.self_sense_eval"),
    ("journal.compose", None, "durable.journal_compose"),
])
def test_every_want_family_run_bids_once_at_its_families_yaml_tier(workflow, line, family):
    async def go():
        m = _Market()
        b = RunBidder(enabled=True, market_getter=lambda: m)
        task = b.on_submit(_req(workflow, line))
        assert b.on_submit(_req(workflow, line)) is None            # duplicate submit: one bid
        await task
        return m
    m = asyncio.run(go())
    assert [w.family for w in m.bids] == [family] and m.bids[0].tier == FAMS.get(family).tier


def test_non_want_runs_and_a_disabled_or_absent_market_bid_nothing():
    async def go():
        m = _Market()
        assert RunBidder(enabled=True, market_getter=lambda: m).on_submit(_req("reverie.visual", None)) is None
        assert RunBidder(enabled=False, market_getter=lambda: m).on_submit(_req()) is None
        assert RunBidder(enabled=True, market_getter=lambda: None).on_submit(_req()) is None
        off = RunBidder(enabled=False, market_getter=lambda: m)
        assert off.on_grant("run-1") == 0 and off.on_terminal("run-1") == 0
        return m
    m = asyncio.run(go())
    assert m.bids == [] and m.opened == [] and m.released == []


def test_submit_returns_before_a_slow_market_and_a_broken_one_never_raises():
    async def go():
        slow = _Market(delay=0.2)
        b = RunBidder(enabled=True, market_getter=lambda: slow)
        t = asyncio.get_running_loop().time()
        task = b.on_submit(_req())
        assert asyncio.get_running_loop().time() - t < 0.05          # scheduled, not awaited
        await task
        broken = _Market(fail=True)
        b2 = RunBidder(enabled=True, market_getter=lambda: broken)
        assert await b2.on_submit(_req()) is None and b2.counts["failed"] == 1
        timed = RunBidder(enabled=True, market_getter=lambda: _Market(delay=1.0), timeout_sec=0.05)
        assert await timed.on_submit(_req()) is None and timed.counts["failed"] == 1

        def boom():
            raise RuntimeError("getter")
        assert RunBidder(enabled=True, market_getter=boom).on_submit(_req()) is None
        return slow
    assert len(asyncio.run(go()).bids) == 1


def test_grant_and_terminal_follow_the_run():
    m = _Market()
    b = RunBidder(enabled=True, market_getter=lambda: m)
    assert b.on_grant("run-1", T0) == 1 and b.on_terminal("run-1") == 1
    assert m.opened == ["run-1"] and m.released == ["run-1"]


def test_flag_ships_on_in_settings_env_example_and_compose():
    from app.settings import Settings

    assert Settings.model_fields["physiology_shadow_bids_enabled"].default is True
    svc = Path(__file__).resolve().parents[1]
    assert "\nORION_PHYSIOLOGY_SHADOW_BIDS_ENABLED=true\n" in (svc / ".env_example").read_text()
    assert "ORION_PHYSIOLOGY_SHADOW_BIDS_ENABLED=${ORION_PHYSIOLOGY_SHADOW_BIDS_ENABLED:-true}" in (
        svc / "docker-compose.yml").read_text()


# --- real Postgres + real pool lane --------------------------------------------------------------

from test_admission_runtime_postgres import DSN, request, runtime, with_database  # noqa: E402
from pool_fixture import InProcessPool  # noqa: E402

pg = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


def _with_market(rt, m):
    rt.bids = RunBidder(enabled=True, market_getter=lambda: m)
    return rt


@pg
def test_admitted_run_bids_once_then_opens_and_releases_with_admission_unchanged():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        m = _Market()
        rt = _with_market(runtime(pool, saver, store, gpu=gpu), m)
        req = request("bid-done")
        receipt = await rt.submit(req)
        assert receipt["status"] == "waiting_resource"               # the receipt is today's
        await asyncio.gather(*rt.bids._tasks)
        [w] = m.bids
        assert (w.want_id, w.correlation_id, w.family) == (req.run_id, req.correlation_id, "durable.curiosity_investigate")
        await rt.submit(req)                                         # duplicate receipt: no second bid
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert m.opened == [req.run_id] and m.released == [req.run_id] and len(m.bids) == 1
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_a_run_refused_before_any_grant_still_releases_its_reservation():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        m = _Market()
        rt = _with_market(runtime(pool, saver, store, gpu=gpu), m)
        req = request("bid-refused", preferred_lane="no-such-route", resource="llm.route.no-such-route")
        await rt.submit(req)
        await asyncio.gather(*rt.bids._tasks)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "failed"
        assert m.opened == [] and m.released == [req.run_id]
        await rt.close()
    asyncio.run(with_database(scenario))
