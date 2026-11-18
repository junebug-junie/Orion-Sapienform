"""The power-bill bet in the organ: ~30 daily guesses at one bill are ONE answer, never 30 samples."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.self_monitor import SettledBetV1
from orion.self_monitor import adapters
from orion.self_monitor.collect import SOURCES, psycopg_sql
from orion.self_monitor.record import BILL_WINDOW_DAYS, MIN_GRADED, kish, reduce_bets, wilson

NOW = datetime(2027, 9, 1, tzinfo=timezone.utc)


def _guesses(bill: int, n: int, inside: int, *, settled: datetime, key: bool = True, source="prediction_bill_forecast_kwh"):
    return [SettledBetV1(
        bet_id=f"b{bill}:{i}", faculty="forecast.energy.bill_kwh", source=source,
        made_at=settled - timedelta(days=35 - i), settled_at=settled, claim_kind="interval", stated=0.8,
        answer_key=f"energy_bill:UP1:{bill}" if key else None,
        outcome="upheld" if i < inside else "refuted", settled_by="world", evidence_ref=f"b{bill}:{i}",
    ) for i in range(n)]


def _one(bets):
    (rec,) = [r for r in reduce_bets(bets, now=NOW) if r.faculty == "forecast.energy.bill_kwh"]
    return rec


def test_thirty_guesses_on_one_bill_count_as_one_answer():
    bets = _guesses(0, 30, 30, settled=NOW - timedelta(days=1))
    rec = _one(bets)
    assert (rec.n_bets, rec.n_graded) == (30, 1)            # n_graded counts answers
    assert rec.gradeability == "ungradeable:too_few" and rec.status == "unknown"
    assert "1 so far" in rec.unlock and "30 bets on 1 answers" in " ".join(rec.reasons)
    # The interval is one answer's, not thirty's.
    assert (rec.reliability_lo, rec.reliability_hi) == pytest.approx(wilson(1.0, 1.0))
    # Without the shared key the same thirty WOULD read as thirty samples: the key is what stops it.
    unkeyed = _one(_guesses(0, 30, 30, settled=NOW - timedelta(days=1), key=False))
    assert unkeyed.gradeability == "graded"


def test_ten_bills_grade_with_n_eff_about_ten_not_three_hundred():
    bets = []
    for b in range(MIN_GRADED):
        bets += _guesses(b, 30, 30 if b % 5 else 12, settled=NOW - timedelta(days=30 * b + 1))
    rec = _one(bets)
    assert rec.gradeability == "graded" and (rec.n_bets, rec.n_graded) == (300, MIN_GRADED)
    weights = [0.5 ** ((30 * b + 1) / 180.0) for b in range(MIN_GRADED)]
    shares = [1.0 if b % 5 else 0.4 for b in range(MIN_GRADED)]
    rel = sum(w * s for w, s in zip(weights, shares)) / sum(weights)
    assert rec.reliability == pytest.approx(rel)
    assert (rec.reliability_lo, rec.reliability_hi) == pytest.approx(wilson(rel, kish(weights)))
    assert kish(weights) < MIN_GRADED                     # the effective n is bills, decayed: under 10
    assert rec.candidate_status == "reliable"           # stated 0.80 sits inside the 10-bill interval


def test_nine_bills_still_unknown():
    bets = []
    for b in range(MIN_GRADED - 1):
        bets += _guesses(b, 30, 30, settled=NOW - timedelta(days=30 * b + 1))
    rec = _one(bets)
    assert rec.gradeability == "ungradeable:too_few" and rec.candidate_status == "unknown"
    assert f"{MIN_GRADED - 1} so far" in rec.unlock


def test_bill_window_holds_a_year_while_other_sources_keep_seven_days():
    old = NOW - timedelta(days=300)
    rec = _one(_guesses(0, 3, 3, settled=old))
    assert rec.n_bets == 3 and rec.window_end - rec.window_start == timedelta(days=BILL_WINDOW_DAYS)
    other = _guesses(0, 3, 3, settled=old, source="prediction_forecast", key=False)
    (r,) = [x for x in reduce_bets(other, now=NOW) if x.faculty == "forecast.energy.bill_kwh"]
    assert r.n_bets == 0                                   # a fast source's 300-day-old bet is out of window


def test_cue_ranks_a_bill_faculty_by_its_answers_not_its_guesses():
    from orion.self_monitor.cue import _evidence_tier

    bets = []
    for b in range(MIN_GRADED):
        bets += _guesses(b, 30, 30, settled=NOW - timedelta(days=30 * b + 1))
    assert _evidence_tier(_one(bets)) == 2                  # 10 bills: the tens tier, not hundreds


def test_bill_sources_read_live_rows_over_their_own_window():
    by = {name: sql for name, sql, _ in SOURCES}
    for name in ("prediction_bill_forecast_kwh", "prediction_bill_forecast_usd"):
        sql = by[name]
        assert f"interval '{int(BILL_WINDOW_DAYS)} days'" in sql and "origin = 'live'" in sql
        assert "CAST(%(since)s AS timestamptz)" in psycopg_sql(sql)


def _row(**over):
    row = {"bet_id": "bill:UP1:2026-11-14", "usage_point_id": "UP1", "origin": "live",
           "as_of": NOW - timedelta(days=8), "settled_at": NOW, "coverage": 0.8, "kwh_in_80": True,
           "usd_in_80": None, "bill_period_start": "2026-10-19"}
    row.update(over)
    return row


def test_adapter_maps_one_row_to_one_bet_per_quantity_and_keys_it_by_the_bill():
    bet = adapters.bill_kwh(_row())
    assert bet.faculty == "forecast.energy.bill_kwh" and bet.claim_kind == "interval" and bet.stated == 0.8
    assert bet.answer_key == "energy_bill:UP1:2026-10-19" and bet.outcome == "upheld"
    assert adapters.bill_usd(_row()) is None                # a bill with no tax line settles no $ claim
    assert adapters.bill_usd(_row(usd_in_80=False)).outcome == "refuted"
    assert adapters.bill_kwh(_row(origin="replay")) is None
    assert adapters.bill_kwh(_row(settled_at=None)) is None
    assert adapters.bill_kwh(_row(coverage=None, kwh_in_80=None)) is None   # no range stated: no bet
