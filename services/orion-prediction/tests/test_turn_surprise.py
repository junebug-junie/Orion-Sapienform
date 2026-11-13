"""Chat-turn surprise (app/turn_surprise.py): the math, the held-out calibration, the rest point,
and the settle pass that builds prediction.turn_surprise.v1 events."""
from __future__ import annotations

import math
import random
import statistics
import typing
from datetime import datetime, timedelta, timezone

import pytest

import app.turn_surprise as ts
from app.turn_self_bets import TurnSelfBets
from orion.schemas.prediction import PredictionTurnSurpriseBetV1, PredictionTurnSurpriseV1

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
ASK, MORE = "turn.self.asks_question", "turn.self.more_steps_than_usual"


@pytest.mark.parametrize("q", [0.02, 0.1, 0.3, 0.7, 0.95])
def test_excess_has_mean_zero_and_the_stated_variance_under_its_own_forecast(q):
    hit, miss = ts.bet_terms(ASK, q, q, True), ts.bet_terms(ASK, q, q, False)
    mean = q * hit.excess + (1 - q) * miss.excess
    var = q * hit.excess ** 2 + (1 - q) * miss.excess ** 2
    assert mean == pytest.approx(0.0, abs=1e-12)
    assert var == pytest.approx(hit.calm_var, rel=1e-9)
    # The unlikely outcome is the surprising one.
    assert (hit.excess > 0) == (q < 0.5)


def test_a_coin_flip_bet_carries_no_surprise_and_a_turn_of_only_those_has_no_reading():
    b = ts.bet_terms(ASK, 0.5, 0.5, True)
    assert b.excess == pytest.approx(0.0, abs=1e-12) and b.calm_var == pytest.approx(0.0, abs=1e-12)
    assert ts.turn_surprise("c", NOW, [b]) is None
    assert ts.turn_surprise("c", NOW, []) is None


def test_a_calm_turn_rests_at_exactly_zero_and_a_two_percent_event_saturates():
    calm = ts.turn_surprise("c", NOW, [ts.bet_terms(ASK, 0.1, 0.1, False), ts.bet_terms(MORE, 0.2, 0.2, False)])
    assert calm.z < 0 and ts.node_value(calm.z) == 0.0
    shock = ts.turn_surprise("c", NOW, [ts.bet_terms(ASK, 0.02, 0.02, True)])
    assert shock.z == pytest.approx(math.sqrt(0.98 / 0.02))   # standardized Bernoulli residual, ~7
    assert ts.node_value(shock.z) == 1.0


def test_calibrated_turns_read_standard_normal_and_mostly_rest():
    rng = random.Random(7)
    zs = []
    for _ in range(20000):
        bets = []
        for fac in ts.USED_FACULTIES:
            q = rng.choice([0.05, 0.1, 0.15, 0.25, 0.4, 0.7])
            bets.append(ts.bet_terms(fac, q, q, rng.random() < q))
        zs.append(ts.turn_surprise("c", NOW, bets).z)
    assert statistics.fmean(zs) == pytest.approx(0.0, abs=0.03)
    assert statistics.pstdev(zs) == pytest.approx(1.0, abs=0.03)
    assert sum(ts.node_value(z) == 0.0 for z in zs) / len(zs) > 0.5


def test_calibration_map_is_identity_with_no_history_and_learns_a_shift():
    assert ts.fit_calibration([]) == (0.0, 1.0)
    rng = random.Random(3)
    # Stated 0.1 but it happens 30 % of the time: the map moves 0.1 toward 0.3.
    hist = [(0.1, rng.random() < 0.3) for _ in range(400)]
    q = ts.calibrated(0.1, ts.fit_calibration(hist))
    assert 0.25 < q < 0.35
    # Well calibrated stated numbers stay roughly where they are.
    good = [(p, rng.random() < p) for p in (0.05, 0.2, 0.5, 0.8) for _ in range(300)]
    ab = ts.fit_calibration(good)
    assert abs(ab[0]) < 0.25 and abs(ab[1] - 1) < 0.25


def _row(cid, fac, placed, p, hit, caller="chat", status=None, settled=None):
    return {"correlation_id": cid, "faculty": fac, "caller": caller, "placed_at": placed,
            "settled_at": settled or placed + timedelta(minutes=2), "stated_p": p, "hit": hit,
            "status": status or ("happened" if hit else "not_happened")}


def _history(n=40, start=NOW - timedelta(days=2), p=0.1, rate=0.1, fac=ASK):
    rng = random.Random(11)
    return [_row(f"h{fac}{i}", fac, start + timedelta(minutes=10 * i), p, rng.random() < rate) for i in range(n)]


def test_score_turn_uses_only_bets_placed_strictly_before_and_already_settled():
    turn = [_row("t", ASK, NOW, 0.1, True), _row("t", MORE, NOW, 0.2, False)]
    hist = ts.CalibrationHistory(_history() + _history(fac=MORE, p=0.2, rate=0.2))
    base, why = ts.score_turn(turn, hist)
    assert why == "scored" and len(base.bets) == 2
    # Bets placed later, or placed earlier but settled after this turn settled, change nothing.
    later = _history(start=NOW + timedelta(minutes=1), rate=1.0)
    late_settled = [dict(r, settled_at=NOW + timedelta(hours=1)) for r in _history(start=NOW - timedelta(hours=3), rate=1.0)]
    hist2 = ts.CalibrationHistory(_history() + _history(fac=MORE, p=0.2, rate=0.2) + later + late_settled)
    again, _ = ts.score_turn(turn, hist2)
    assert again.z == pytest.approx(base.z)


def test_score_turn_refusals():
    hist = ts.CalibrationHistory(_history())
    assert ts.score_turn([], hist) == (None, "no_rows")
    assert ts.score_turn([_row("t", ASK, NOW, 0.1, True, caller="orion_curiosity")], hist)[1] == "not_chat"
    assert ts.score_turn([_row("t", ASK, NOW, 0.1, True)], ts.CalibrationHistory(_history(n=ts.MIN_CAL - 1)))[1] \
        == "too_little_calibration_history"
    # Unused kinds and void rows never count.
    other = [_row("t", "turn.self.fetches_outside", NOW, 0.02, True),
             _row("t", ASK, NOW, None, None, status="void")]
    assert ts.score_turn(other, hist) == (None, "no_graded_used_bet")


def test_heldout_skill_credits_informative_bets_and_not_a_constant():
    rng = random.Random(5)
    rows = []
    for i in range(600):
        t = NOW - timedelta(days=10) + timedelta(minutes=20 * i)
        p = rng.choice([0.05, 0.6])
        rows.append(_row(f"a{i}", ASK, t, p, rng.random() < p))                 # knows when
        rows.append(_row(f"b{i}", MORE, t, 0.25, rng.random() < 0.25))         # says only the rate
    good, flat = ts.heldout_skill(rows, ASK), ts.heldout_skill(rows, MORE)
    assert good["gap"] > 3 * good["se"] and good["prob_better"] > 0.99
    assert flat["gap"] < good["gap"] and flat["prob_better"] < 0.9


def test_used_faculties_match_the_event_schema():
    allowed = typing.get_args(PredictionTurnSurpriseBetV1.model_fields["faculty"].annotation)
    assert set(allowed) == set(ts.USED_FACULTIES)


# --- the settle pass ------------------------------------------------------------------------------
class FakeStore:
    def __init__(self, pending, history=()):
        self.pending = pending
        self.rows: dict[str, dict] = {}
        self.history = list(history)
        self.fail_history = False

    def pending_turn_self_bets(self, since, limit):
        done = {r["correlation_id"] for r in self.rows.values()}
        return [b for b in self.pending if b["correlation_id"] not in done and b["created_at"] >= since][:limit]

    def insert_turn_self_bets(self, rows):
        new = [r for r in rows if r["bet_id"] not in self.rows]
        self.rows.update({r["bet_id"]: r for r in new})
        return len(new)

    def turn_size_history(self, since):
        return []

    def turn_surprise_calibration_rows(self, caller, faculties, since):
        if self.fail_history:
            raise RuntimeError("db down")
        live = [r for r in self.rows.values() if r["caller"] == caller and r["faculty"] in faculties
                and r["status"] in ("happened", "not_happened")]
        return [r for r in self.history + live if r["placed_at"] >= since]


PRED = {"schema_version": "turn.self_prediction.v2", "asks_question_p": 0.1, "fetches_outside_p": 0.02,
        "more_steps_than_usual_p": 0.2, "longer_reply_than_usual_p": 0.3}
CHAT_SESSION = "0b9c2f7e-6f0a-4c37-9d3e-2c1a7f5e9b10"


def _pending(cid, session=CHAT_SESSION, age=timedelta(minutes=3), text="Here you go."):
    return {"correlation_id": cid, "session_id": session, "created_at": NOW - age, "self_prediction": PRED,
            "run_artifact": {"final_text": text, "step_count": 3, "source_fetches": [], "exit_code": 0}}


def test_a_settled_chat_turn_becomes_one_valid_event_and_others_do_not():
    store = FakeStore([_pending("chat1", text="Want me to keep going?"), _pending("cur1", session="orion_curiosity:x"),
                       _pending("old1", age=timedelta(hours=2))],
                      history=_history() + _history(fac=MORE, p=0.2, rate=0.2))
    svc = TurnSelfBets(store, clock=lambda: NOW, surprise_events_enabled=True)
    svc.settle_pending()
    events = svc.take_pending_surprise()
    assert [e.correlation_id for e in events] == ["chat1"]
    ev = PredictionTurnSurpriseV1.model_validate(events[0].model_dump(mode="json"))
    # more_steps has no reference in this fake (void), so only the question bet scores; it happened at ~0.1.
    assert [b.faculty for b in ev.bets] == [ASK] and ev.bets[0].happened and ev.z > 2
    assert svc.surprise_counts["stale_turn"] == 1
    assert svc.take_pending_surprise() == []


def test_surprise_is_off_unless_enabled_and_a_failure_never_costs_the_settle_pass():
    store = FakeStore([_pending("chat1")], history=_history())
    off = TurnSelfBets(store, clock=lambda: NOW)
    assert off.settle_pending()["rows_written"] == 6 and off.pending_surprise == []
    store2 = FakeStore([_pending("chat2")], history=_history())
    store2.fail_history = True
    svc = TurnSelfBets(store2, clock=lambda: NOW, surprise_events_enabled=True)
    assert svc.settle_pending()["rows_written"] == 6
    assert svc.pending_surprise == [] and "db down" in svc.last_surprise_error
