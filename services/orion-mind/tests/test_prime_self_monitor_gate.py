"""Self-monitor E1 act gate on Prime's own acts (app/prime_loop.py; orion/self_monitor/gate.py).

Live 2026-11-03: ``act.prime.predict`` (Prime's concern bets) is held unreliable, works about 2-6%.
Driven through the real loop with the fake bus, model and market of test_prime_loop; the gate is the
real ``build_act_gate`` with only its Redis view read and its ledger read faked.
"""
from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from app import prime_loop
from orion.prime.state import close_concern
from orion.self_monitor.view import self_monitor_view
from test_prime_loop import MOVE, SETTINGS, T0, Bus, runtime

PREDICT = "act.prime.predict"
GATE_ON = {"SELF_MONITOR_ACT_GATE_ENABLED": True, "SELF_MONITOR_PROBE_INTERVAL_SEC": 21600.0,
           "SELF_MONITOR_LEDGER_PG_DSN": "postgresql://x/y", "ORION_BUS_URL": "redis://x:6379/0"}


def _state(status: str = "unreliable", *, faculty: str = PREDICT, age: timedelta = timedelta(0)) -> str:
    at = T0 - age
    return json.dumps({"kind": "self_monitor.state.v1", "computed_at": at.isoformat(), "window_days": 7.0,
                       "faculties": [{
                           "faculty": faculty, "gradeability": "graded",
                           "window_start": (at - timedelta(days=7)).isoformat(), "window_end": at.isoformat(),
                           "n_bets": 42, "n_graded": 41, "n_failed_to_execute": 0, "n_expired": 0,
                           "weight_graded": 41.0, "reliability": 0.05, "candidate_status": status,
                           "status": status, "status_since": (at - timedelta(days=1)).isoformat()}]})


class _Ledger:
    def __init__(self, rows=None, fail=False):
        self.rows, self.fail, self.calls = list(rows or []), fail, []

    def __call__(self, dsn, actor, verb, secs):
        self.calls.append((actor, verb))
        if self.fail:
            raise RuntimeError("db down")
        return [r for r in self.rows if r.get("verb", verb) == verb]


def _recent(verb: str = "predict", ref: str = "prime_concern:0a1b2c3d4e5f6a7b", ago=timedelta(hours=1)) -> dict:
    return {"verb": verb, "object_ref": ref, "test_before": None, "opened_at": T0 - ago}


def gated(monkeypatch, raw, ledger=None, *, enabled=True, **kw):
    """A runtime whose gate is the real build_act_gate, view and ledger faked."""
    ledger = ledger if ledger is not None else _Ledger()
    monkeypatch.setattr(prime_loop, "read_view_sync", lambda url, now: self_monitor_view(raw, now=now))
    monkeypatch.setattr(prime_loop, "_ledger_rows", ledger)
    rt = runtime(**kw)
    rt.gate = prime_loop.build_act_gate(SimpleNamespace(**{**vars(SETTINGS), **GATE_ON,
                                                           "SELF_MONITOR_ACT_GATE_ENABLED": enabled}))
    rt.ledger = ledger
    return rt


def _kinds(rt):
    return [(k, a["verb"]) for k, a in rt.bus.acts()]


# --- predict: a refused concern bet --------------------------------------------------------------

def test_failing_predict_places_no_bet_but_keeps_the_concern(monkeypatch):
    rt = gated(monkeypatch, _state(), _Ledger([_recent()]), answer=MOVE)
    asyncio.run(rt.tick())
    # The thought and its question are not failing; the concern's bet is refused.
    assert _kinds(rt) == [("act_opened", "infer"), ("act_opened", "ask"), ("act_closed", "infer")]
    [concern] = rt.state.concerns
    assert concern.text == "afternoon heat" and concern.act_opened_at is None
    # Prime is told once, as a frame line, never as a change that wakes it.
    told = [o for o in rt.state.recent_outcomes if "was not placed" in o]
    assert len(told) == 1 and "unreliable" in told[0] and "6 h" in told[0]
    assert rt.state.pending_changes == []
    # Next ticks: still no bet, not told again.
    for minutes in (1, 2, 5):
        rt.clock_box["now"] = T0 + timedelta(minutes=minutes)
        asyncio.run(rt._flush_concern_acts())
    assert [v for _, v in _kinds(rt)].count("predict") == 0
    assert len([o for o in rt.state.recent_outcomes if "was not placed" in o]) == 1


def test_refused_concern_is_rechecked_at_most_once_a_minute(monkeypatch):
    rt = gated(monkeypatch, _state(), _Ledger([_recent()]), answer=MOVE)
    asyncio.run(rt.tick())
    n = len(rt.ledger.calls)
    rt.clock_box["now"] = T0 + timedelta(seconds=15)
    asyncio.run(rt._flush_concern_acts())
    assert len(rt.ledger.calls) == n
    rt.clock_box["now"] = T0 + timedelta(seconds=61)
    asyncio.run(rt._flush_concern_acts())
    assert len(rt.ledger.calls) > n


def test_probe_places_one_bet_then_the_next_concern_waits(monkeypatch):
    rt = gated(monkeypatch, _state(), _Ledger([]), answer=MOVE)
    asyncio.run(rt.tick())
    assert ("act_opened", "predict") in _kinds(rt)
    assert rt.state.concerns[0].act_opened_at is not None
    # A second concern in the same window: the probe is spent (memo), so no second bet.
    from orion.prime.state import PrimeConcernV1

    extra = PrimeConcernV1(concern_id="ffeeddccbbaa9988", text="evening cooling", opened_at=T0, last_evidence_at=T0)
    rt.state = rt.state.model_copy(update={"concerns": [*rt.state.concerns, extra]})
    rt.clock_box["now"] = T0 + timedelta(minutes=2)
    asyncio.run(rt._flush_concern_acts())
    assert [v for _, v in _kinds(rt)].count("predict") == 1
    assert rt.state.concerns[1].act_opened_at is None


def test_a_started_bet_whose_publish_failed_is_retried_as_the_same_job(monkeypatch):
    class FlakyBus(Bus):
        failed = False

        async def publish(self, channel, envelope):
            act = (envelope.payload or {}).get("act") or {}
            if act.get("verb") == "predict" and not FlakyBus.failed:
                FlakyBus.failed = True
                raise RuntimeError("bus blip")
            await super().publish(channel, envelope)

    rt = gated(monkeypatch, _state(), _Ledger([]), answer=MOVE, bus=FlakyBus())
    asyncio.run(rt.tick())                       # probe claimed, open publish failed
    assert rt.state.concerns[0].act_opened_at is None
    rt.clock_box["now"] = T0 + timedelta(minutes=1)
    asyncio.run(rt._flush_concern_acts())        # same concern: same job -> admitted, no new probe
    assert ("act_opened", "predict") in _kinds(rt)
    assert rt.state.concerns[0].act_opened_at is not None


def test_a_refused_concern_ends_with_no_late_bet_and_the_loop_goes_on(monkeypatch):
    rt = gated(monkeypatch, _state(), _Ledger([_recent()]), answer=MOVE)
    asyncio.run(rt.tick())
    [concern] = rt.state.concerns
    # Prime drops it (or it fades/settles): the record is the act's outbox.
    rt.clock_box["now"] = T0 + timedelta(hours=2)
    rt.state = close_concern(rt.state, concern, end="dropped", now=rt.clock_box["now"], note="dropped unresolved")
    rt.state = rt.state.model_copy(update={"concerns": []})
    asyncio.run(rt._flush_concern_acts())
    [rec] = rt.state.concern_record
    assert rec.published is True and rec.end == "dropped"
    assert [v for _, v in _kinds(rt)].count("predict") == 0
    # Even if the faculty has healed by the time it ends: no late bet for a concern the gate refused.
    assert rt._bet_refused_at == {}
    # The loop keeps going: the next tick runs without error and saves.
    rt.clock_box["now"] = T0 + timedelta(hours=3)
    asyncio.run(rt.tick())


def test_late_bet_for_a_concern_never_refused_follows_the_gate(monkeypatch):
    """A concern whose open failed for another reason (bus down) and then ended: with the faculty
    failing, a late open+close is not placed (its outcome is known: it would test nothing)."""
    from orion.prime.state import PrimeConcernV1, PrimeStateV1

    rt = gated(monkeypatch, _state(), _Ledger([]), answer=MOVE)
    c = PrimeConcernV1(concern_id="0a1b2c3d4e5f6a7b", text="heat", opened_at=T0, last_evidence_at=T0)
    rt.state = close_concern(PrimeStateV1(), c, end="faded", now=T0, note="faded")
    asyncio.run(rt._flush_concern_acts())
    assert rt.bus.acts() == [] and rt.state.concern_record[0].published


# --- ask / regulate: refused before the holdback draw ---------------------------------------------

@pytest.mark.parametrize("holdback", [0.0, 1.0])
def test_failing_ask_is_refused_in_both_arms(monkeypatch, holdback):
    rt = gated(monkeypatch, _state(faculty="act.prime.ask"), _Ledger([_recent("ask", "curiosity_focus:9f8e7d")]),
               answer=MOVE, holdback=holdback)
    asyncio.run(rt.tick())
    assert ("act_opened", "ask") not in _kinds(rt)
    assert rt.bus.rpcs == [] and rt.state.open_nominations == []
    assert any(o.startswith("my move was not sent: my own record says my questions have been unreliable")
               for o in rt.state.recent_outcomes)
    # The thought itself closed produced; the refusal names itself in its note.
    closed = [a for k, a in rt.bus.acts() if k == "act_closed" and a["verb"] == "infer"][0]
    assert closed["outcome"]["value"] == 1.0 and "move refused" in closed["outcome"]["note"]


# --- infer: refused before the bid ----------------------------------------------------------------

def test_failing_infer_makes_no_bid_and_no_model_call(monkeypatch):
    rt = gated(monkeypatch, _state(faculty="act.prime.infer"), _Ledger([_recent("infer", "prime_thought:a1")]),
               answer=MOVE)
    asyncio.run(rt.tick())
    assert rt.bids == [] and rt.llm.calls == [] and rt.bus.acts() == []
    assert rt.state.pending_changes  # still has something to think about: the loop is intact
    assert len([o for o in rt.state.recent_outcomes if "did not start a thought" in o]) == 1


def test_infer_probe_is_spent_only_when_the_model_is_reached(monkeypatch):
    rt = gated(monkeypatch, _state(faculty="act.prime.infer"), _Ledger([]), answer=MOVE, outcome="deferred")
    asyncio.run(rt.tick())                       # probe offered, bid not funded: nothing spent
    assert rt.bids and rt.gate._memo.get("act.prime.infer") is None
    rt2 = gated(monkeypatch, _state(faculty="act.prime.infer"), _Ledger([]), answer=MOVE)
    asyncio.run(rt2.tick())                      # funded, model reached: the probe is spent
    assert rt2.gate._memo.get("act.prime.infer") is not None
    assert ("act_opened", "infer") in _kinds(rt2)


# --- must not loosen, must not tighten: unknown state == gate off ---------------------------------

def _observable(rt):
    return (_kinds(rt), [c.text for c in rt.state.concerns], [c.act_opened_at for c in rt.state.concerns],
            len(rt.bus.rpcs), [o for o in rt.state.recent_outcomes])


@pytest.mark.parametrize("raw", [None, "{bad", "STALE", "RELIABLE", "OTHER"])
def test_unknown_stale_absent_or_healthy_state_is_identical_to_gate_off(monkeypatch, raw):
    raw = {"STALE": _state(age=timedelta(hours=4)), "RELIABLE": _state("reliable"),
           "OTHER": _state(faculty="act.actions.journal.make.walkway_grade")}.get(raw, raw)
    on = gated(monkeypatch, raw, _Ledger([_recent()]), answer=MOVE)
    asyncio.run(on.tick())
    off = gated(monkeypatch, raw, _Ledger([_recent()]), answer=MOVE, enabled=False)
    asyncio.run(off.tick())
    assert _observable(on) == _observable(off)
    assert ("act_opened", "predict") in _kinds(on)
    # The common case reads no ledger at all.
    assert on.ledger.calls == []


def test_gate_fault_changes_nothing(monkeypatch):
    rt = gated(monkeypatch, _state(), answer=MOVE)

    def boom(*a, **k):
        raise RuntimeError("gate bug")

    rt.gate.decide = boom
    asyncio.run(rt.tick())
    assert ("act_opened", "predict") in _kinds(rt)


def test_unreadable_ledger_falls_back_to_the_memo(monkeypatch):
    rt = gated(monkeypatch, _state(), _Ledger(fail=True), answer=MOVE)
    asyncio.run(rt.tick())
    # One probe (never a permanent refusal because the ledger is down).
    assert [v for _, v in _kinds(rt)].count("predict") == 1


def test_gate_is_off_without_the_setting():
    gate = prime_loop.build_act_gate(SETTINGS)
    assert gate.enabled is False


def test_settings_ship_the_gate_on():
    from app.settings import Settings

    s = Settings()
    assert s.SELF_MONITOR_ACT_GATE_ENABLED is True and s.SELF_MONITOR_PROBE_INTERVAL_SEC == 21600.0
