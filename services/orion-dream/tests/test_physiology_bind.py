"""Body economy Patch D for the dream cycle: the sleep OBEYS its bid (app/physiology_bind.py).

The contract: a sleep that clears every gate of the dream's own asks the market just before its act opens.
Funded, or a full metacog lane worth its price -> it sleeps. Price (Prime's slow charge included), no hunger,
net <= 0 -> held, one want re-bid at later checks, never past the 6 h window or the overdue line; an overdue
sleep never bids. No answer -> it sleeps as before. Forced runs and arousal holds never bid. Flag shadow -> the
C3b shadow bid, exactly as before.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.schemas.physiology_want import WantDecisionV1, WantV1
from test_dream_cycle_v2 import _Fakes, _rows
from test_dream_reads_arousal import _deps, _state

REPO = Path(__file__).resolve().parents[3]


def _decision(want: WantV1, outcome: str, reason: str, *, net: float = 0.4, retry: float | None = None,
              why: str = "value 0.30 - lane charge 0.00 did not beat Prime's slow charge 0.50") -> WantDecisionV1:
    now = datetime.now(timezone.utc)
    return WantDecisionV1(
        decision_id=f"{want.want_id}@{now.isoformat()}", want_id=want.want_id, family=want.family,
        producer=want.producer, verb=want.verb, tier=want.tier, decided_at=now, outcome=outcome, reason=reason,
        lane="metacog", net=net, retry_after_sec=retry, binding_constraint="prime:slow" if reason == "price" else None,
        why=why, want=want,
    )


@pytest.fixture
def market(monkeypatch):
    """The market as the dream sees it: answers set per test; every bid and release recorded."""
    from app import physiology_bind as pb

    class _Market:
        answers: list = []
        wants: list = []
        released: list = []

    m = _Market()
    m.answers, m.wants, m.released = [], [], []

    async def fake_abid(want, timeout_sec):
        m.wants.append(want)
        a = m.answers.pop(0) if m.answers else ("funded", "net_positive", {})
        if a == "raises":
            raise RuntimeError("economy down")
        if a == "hangs":
            await asyncio.sleep(30)
        if a is None:
            return None
        outcome, reason, kw = a
        return _decision(want, outcome, reason, **kw)

    async def fake_release(want_id):
        m.released.append(want_id)
        return 1

    monkeypatch.setattr(pb, "_abid", fake_abid)
    monkeypatch.setattr(pb, "_arelease", fake_release)
    m.pb = pb
    return m


def _fakes(**kw):
    """Slept 8 h ago: past the 6 h refractory, well inside the 48 h overdue line (never slept reads overdue)."""
    ago = datetime.now(timezone.utc) - timedelta(hours=8)
    return _Fakes(_rows(), **{"last_start": ago, "last_end": ago, **kw})


def _bound(f, level="idle", **kw):
    from app.physiology_bind import DreamBinder

    deps, calls = _deps(f, _state(level))
    deps.economy = DreamBinder(**kw)
    events = []

    async def publish(event):
        events.append(event)

    deps.publish_act = publish
    return deps, calls, events


def _run(deps, **kw):
    from app.cycle import run_cycle_once

    async def go():
        out = await run_cycle_once(deps, **kw)
        for _ in range(5):
            await asyncio.sleep(0)
        return out

    return asyncio.run(go())


# --- the contract with config ---------------------------------------------------------------------------

def test_the_producer_is_one_the_market_charges_for_prime():
    from app.physiology_bind import BINDING_PRODUCER
    from orion.physiology.binding import BINDING_PRODUCERS

    assert BINDING_PRODUCER in BINDING_PRODUCERS


def test_the_window_matches_families_yaml_and_dream_declares_slow_only():
    from app.physiology_bind import DEFERRABLE_WINDOW_S, FAMILY
    from orion.physiology.families import load_families

    fams, err = load_families(REPO / "config/physiology/families.yaml")
    assert fams is not None, err
    cfg = fams.get(FAMILY)
    assert cfg.tier == 1 and cfg.deferrable_window_s == DEFERRABLE_WINDOW_S
    assert cfg.receptors == ("slow",)


def test_hold_end_is_the_earlier_of_the_window_and_the_overdue_line():
    from app.physiology_bind import hold_end

    t = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
    assert hold_end(t, t - timedelta(hours=10), lookback_hours=48) == t + timedelta(hours=6)
    assert hold_end(t, t - timedelta(hours=45), lookback_hours=48) == t + timedelta(hours=3)
    assert hold_end(t, None, lookback_hours=48) == t + timedelta(hours=6)


def test_the_flag_ships_on_and_anything_else_is_shadow(caplog):
    from app.physiology_bind import bind_mode
    from app.settings import settings

    assert settings.ORION_PHYSIOLOGY_BID_DREAM_CYCLE == "on" and bind_mode() == "on"
    assert bind_mode("shadow") == "shadow" and bind_mode(" ON ") == "on"
    with caplog.at_level(logging.WARNING, logger="orion-dream.physiology_bind"):
        assert bind_mode("yes please") == "shadow"
    assert "physiology_bind_unknown_mode" in caplog.text


# --- obeying the bid -----------------------------------------------------------------------------------

def test_a_funded_sleep_runs_bids_once_with_the_acts_id_and_frees_its_reservation(market, monkeypatch):
    from app import physiology_shadow as ps

    shadow_bids = []
    monkeypatch.setattr(ps, "abid", lambda w: shadow_bids.append(w))
    f = _fakes()
    deps, _, events = _bound(f)
    cycle = _run(deps)
    assert cycle is not None and cycle.status == "completed"
    (want,) = market.wants
    assert want.want_id == want.correlation_id == events[0].correlation_id
    assert (want.family, want.verb, want.tier, want.producer) == ("dream.cycle", "make", 1, "orion-dream.dream_cycle.binding")
    assert want.window.latest is not None and want.opened_at is not None
    assert shadow_bids == []                                  # one want, one bid: no shadow bid as well
    assert market.released == [want.want_id]                  # the act closed: the reservation is freed
    assert "body economy" not in (cycle.note or "")
    assert deps.economy.want is None                          # the next stretch opens a new want


def test_a_price_deferral_holds_the_sleep_records_why_and_the_same_want_sleeps_once_funded(market, caplog):
    caplog.set_level(logging.INFO, logger="orion-dream")
    market.answers = [("deferred", "price", {"net": -0.2, "retry": 60.0})]
    f = _fakes()
    deps, calls, events = _bound(f)
    assert _run(deps) is None
    assert deps.hold_reason == "economy_price" and f.persisted == [] and f.prompts == [] and events == []
    assert calls == [1]                                       # arousal was read first: the economy is after it
    held = deps.economy.want
    assert held.holds == 1 and held.retry_at is not None
    # The market said "wait": a re-bid before retry_at reads the same prices, so it holds without one.
    assert _run(deps) is None and len(market.wants) == 1 and deps.hold_reason == "economy_price"
    held.retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    cycle = _run(deps)
    assert cycle is not None and cycle.status == "completed"
    assert len(market.wants) == 2 and market.wants[0].want_id == market.wants[1].want_id == events[0].correlation_id
    assert cycle.cycle_id == held.cycle_id
    assert "body economy: held 1 time(s)" in cycle.note and "did not beat Prime's slow charge" in cycle.note
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("physiology_bind ")]
    assert any("verdict=deferred:price admit=False" in ln and "binding=prime:slow" in ln for ln in lines)
    assert any("verdict=waiting:price admit=False" in ln for ln in lines)
    assert any("verdict=funded admit=True" in ln for ln in lines)
    assert market.released[0] == market.wants[0].want_id     # the held bid reserved nothing


@pytest.mark.parametrize("answer,why", [
    (("deferred", "capacity_reserved", {"net": 0.3}), "queued_at_pool"),
    (None, "no_answer:no_decision"),
    ("raises", "no_answer:no_decision"),
    (("deferred", "physiology_unavailable", {"net": None}), "no_answer:physiology_unavailable"),
])
def test_a_full_lane_worth_its_price_or_no_answer_sleeps_as_before(market, caplog, answer, why):
    caplog.set_level(logging.INFO, logger="orion-dream.physiology_bind")
    market.answers = [answer]
    f = _fakes()
    deps, _, _ = _bound(f)
    cycle = _run(deps)
    assert cycle is not None and cycle.status == "completed"
    assert f"verdict={why} admit=True" in caplog.text


def test_a_hung_market_never_holds_the_sleep(market):
    market.answers = ["hangs"]
    f = _fakes()
    deps, _, _ = _bound(f, timeout_sec=0.05)
    cycle = _run(deps)
    assert cycle is not None and cycle.status == "completed"


@pytest.mark.parametrize("reason", ["sated", "price"])
def test_a_held_sleep_goes_at_the_end_of_its_window_without_a_bid(market, reason):
    market.answers = [("deferred", reason, {"net": 0.0})]
    f = _fakes()
    deps, _, _ = _bound(f)
    assert _run(deps) is None
    deps.economy.want.first_bid_at -= timedelta(hours=6, minutes=1)
    deps.economy.want.retry_at = None
    cycle = _run(deps)
    assert cycle is not None and cycle.status == "completed" and len(market.wants) == 1
    assert "body economy: window_end" in cycle.note


def test_an_overdue_sleep_never_bids_whatever_the_charge(market):
    market.answers = [("deferred", "price", {"net": -0.4})]
    f = _Fakes(_rows(), last_start=datetime.now(timezone.utc) - timedelta(hours=49),
               last_end=datetime.now(timezone.utc) - timedelta(hours=49))
    deps, _, _ = _bound(f)
    cycle = _run(deps)
    assert cycle is not None and cycle.status == "completed" and market.wants == []
    assert "body economy: floor:overdue" in cycle.note


def test_forced_runs_and_arousal_holds_never_bid(market):
    f = _fakes()
    deps, _, _ = _bound(f, level="strained")
    assert _run(deps) is None and deps.hold_reason == "arousal_strained" and market.wants == []
    f2 = _fakes()
    deps2, _, _ = _bound(f2)
    market.answers = [("deferred", "price", {"net": -0.4})]
    cycle = _run(deps2, force=True)
    assert cycle is not None and cycle.status == "completed" and market.wants == []


def test_a_sleep_attempt_closes_the_open_want(market):
    from app.physiology_bind import DreamBinder

    b = DreamBinder()
    t = datetime.now(timezone.utc)
    w = b.current(t, None)
    assert b.current(t + timedelta(minutes=10), None) is w            # the same want across checks
    assert b.current(t + timedelta(minutes=20), t + timedelta(minutes=15)) is not w   # a sleep ended since


def test_with_the_flag_shadow_the_dream_places_the_c3b_shadow_bid(monkeypatch):
    from app import physiology_shadow as ps

    bids = []

    async def fake(want):
        bids.append(want)
        return None

    monkeypatch.setattr(ps, "abid", fake)
    monkeypatch.setattr(ps, "_enabled", lambda: True)
    f = _Fakes(_rows())
    deps, _ = _deps(f, _state("idle"))
    assert deps.economy is None
    cycle = _run(deps)
    assert cycle is not None and cycle.status == "completed" and len(bids) == 1
    assert bids[0].producer == "orion-dream.dream_cycle"


# --- the real market rule with dream + Prime's charge -----------------------------------------------------

def test_the_real_choice_holds_a_short_gap_sleep_under_primes_slow_charge_and_a_hungry_one_outbids_it():
    from app.physiology_bind import FAMILY, hold_end
    from orion.physiology import prime_charge as pc
    from orion.physiology.appetite import ActRecord
    from orion.physiology.binding import bind_verdict
    from orion.physiology.choice import ReservationBook, build_context, decide
    from orion.physiology.families import load_families
    from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R, PhysiologyStateV1

    fams, _ = load_families(REPO / "config/physiology/families.yaml")
    now = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
    acts, t = [], now - timedelta(hours=8)
    for i, g in enumerate([9, 14, 7, 20, 11, 6, 16, 8, 13, 10]):
        acts.append(ActRecord(act_id=f"d{i}", family=FAMILY, opened_at=t - timedelta(seconds=10), closed_at=t,
                              outcome_value=1.0, outcome_source="producer",
                              cost={"seconds": 10.0, "gpu_slot_s_by_lane": {"metacog": 10.0}}))
        t -= timedelta(hours=g)
    price = {"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""}
    state = PhysiologyStateV1(tick_id="p", generated_at=now, constraints=[
        R(constraint="lane:metacog", unit="slot_s", level=0.0, budget=3600.0, budget_kind="capacity",
          detail={"slots": 1, "window_sec": 3600.0}, price=price)])
    ctx = build_context(fams, acts, [], state, now, weights={})
    sig = pc.PrimeRegulationV1(signal_id="s", act_id="a", family=FAMILY, direction="slow", arm="treated",
                               magnitude=0.5, why="dream slept four times yesterday", opened_at=now - timedelta(hours=1),
                               expires_at=now + timedelta(hours=3))
    want = WantV1(want_id="w", family=FAMILY, producer="test", verb="make", tier=1, opened_at=now)
    free = decide(want, state, ctx, ReservationBook(), now)
    held = decide(want, state, ctx, ReservationBook(), now, prime=sig)
    assert free.outcome == "funded"                              # 8 h after the last sleep, no charge: it sleeps
    assert held.outcome == "deferred" and held.reason == "price" and held.binding_constraint == "prime:slow"
    assert pc.effect_between("slow", held, free) == "held"
    v = bind_verdict(held, now, hold_end(now, now - timedelta(hours=8), lookback_hours=48))
    assert not v.admit and v.why == "deferred:price" and "did not beat Prime's slow charge" in v.detail["why"]
    later = now + timedelta(hours=8)                             # 16 h since the last sleep: hungry
    ctx2 = build_context(fams, acts, [], state.model_copy(update={"generated_at": later}), later, weights={})
    d2 = decide(want, state.model_copy(update={"generated_at": later}), ctx2, ReservationBook(), later, prime=sig)
    assert d2.outcome == "funded" and "beat Prime's slow charge" in d2.why


def test_arousal_holds_do_not_use_up_the_economys_window(market):
    """Live 10-29/30: arousal held most due checks for hours before the sleep. The window starts at the first bid."""
    f = _fakes()
    deps, _, _ = _bound(f, level="engaged")
    assert _run(deps) is None and deps.hold_reason == "arousal_engaged" and market.wants == []
    want = deps.economy.want
    assert want is not None and want.first_bid_at is None
    want.opened_at -= timedelta(hours=7)                     # the want arose 7 h ago; arousal held it since
    deps2, _, _ = _bound(f)
    deps2.economy = deps.economy
    market.answers = [("deferred", "price", {"net": -0.2})]
    assert _run(deps2) is None and deps2.hold_reason == "economy_price"   # priced, not waved through as window_end
    assert want.first_bid_at is not None and len(market.wants) == 1


def test_two_checks_racing_on_one_want_open_one_sleep(market):
    from app.cycle import run_cycle_once

    f = _fakes()
    deps, _, events = _bound(f)
    deps_b, _, _ = _bound(f)
    deps_b.economy = deps.economy
    gate = asyncio.Event()
    orig = market.pb._abid

    async def slow_abid(want, timeout_sec):
        await gate.wait()
        return await orig(want, timeout_sec)

    market.pb._abid = slow_abid

    async def go():
        a = asyncio.create_task(run_cycle_once(deps))
        b = asyncio.create_task(run_cycle_once(deps_b))
        await asyncio.sleep(0.05)
        gate.set()
        return await a, await b

    ra, rb = asyncio.run(go())
    done = [c for c in (ra, rb) if c is not None]
    assert len(done) == 1 and done[0].status == "completed"
    assert len({c.cycle_id for c in f.persisted}) == len(f.persisted) == 1
    assert "economy_duplicate" in (deps.hold_reason, deps_b.hold_reason)


def test_a_funded_sleep_that_raises_frees_its_reservation(market):
    f = _fakes()
    deps, _, _ = _bound(f)

    def boom(cycle):
        raise RuntimeError("db down")

    deps.persist_cycle = boom
    with pytest.raises(RuntimeError):
        _run(deps)
    (want,) = market.wants
    assert market.released == [want.want_id]
