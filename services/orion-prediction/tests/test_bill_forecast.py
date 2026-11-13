from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app import bill_forecast as B
from orion.energy.reconcile import project_period
from orion.energy.testing import flat_test_tariff
from orion.schemas.energy import EnergyBillActualV1, EnergyUsageIntervalV1
from orion.self_monitor import adapters
from orion.self_monitor.collect import SOURCES, build_state

UTC = ZoneInfo("UTC")
REPO = Path(__file__).resolve().parents[3]


def _kwh(t: datetime) -> float:
    """Deterministic, uneven daily usage: ~1 kWh/h with a day-to-day swing (never flat)."""
    return 1.0 + 0.4 * math.sin(t.timetuple().tm_yday * 1.7) + 0.1 * (t.hour % 3)


def _rows(start: datetime, end: datetime, *, lag: timedelta = timedelta(hours=1), skip=()) -> list[dict]:
    out, t = [], start
    while t < end:
        if t not in skip:
            out.append({"source": "file_drop", "usage_point_id": "UP1", "interval_start": t,
                        "interval_end": t + timedelta(hours=1), "energy_kwh": _kwh(t), "quality": None,
                        "retrieved_at": t + timedelta(hours=1) + lag, "source_file": None})
        t += timedelta(hours=1)
    return out


def _ivs(rows) -> list[EnergyUsageIntervalV1]:
    return [EnergyUsageIntervalV1(**r) for r in rows]


def _bill(start: date, end: date, kwh: float, *, taxes=5.0, total=100.0, got: datetime | None = None) -> dict:
    return {"source": "file_drop", "usage_point_id": None, "billing_period_start": start, "billing_period_end": end,
            "kwh_billed": kwh, "energy_charge": None, "customer_charge": None, "adjustments": None, "fees": None,
            "taxes": taxes, "credits": None, "current_charges": total, "amount_due": None, "due_date": None,
            "statement_artifact_id": None,
            "retrieved_at": got or datetime.combine(end, datetime.min.time(), tzinfo=UTC) + timedelta(days=3),
            "source_file": None}


SEP1 = datetime(2026, 9, 1, tzinfo=UTC)
OCT1 = datetime(2026, 10, 1, tzinfo=UTC)
NOV1 = datetime(2026, 11, 1, tzinfo=UTC)


# --- the range -------------------------------------------------------------------------------------


def test_spread_needs_seven_complete_days_and_clamps_persistence():
    days = [(date(2026, 9, 1) + timedelta(days=i), 10.0 + (i % 2)) for i in range(6)]
    assert B.spread(days) is None
    sp = B.spread(days + [(date(2026, 9, 7), 10.0)])
    assert sp is not None and sp.n_days == 7 and sp.rho == 0.0      # alternating days: negative rho read as 0
    trend = [(date(2026, 9, 1) + timedelta(days=i), float(i)) for i in range(28)]
    assert B.spread(trend).rho == B.RHO_MAX                          # a ramp is pure persistence: capped


def test_kwh_range_is_the_sum_of_days_interval_and_shrinks_as_the_cycle_fills():
    sp = B.Spread(s_kwh=10.0, rho=0.0, n_days=28, n_pairs=27)
    lo, hi, sd = B.kwh_range(900.0, 300.0, remaining_days=20.0, observed_days=10.0, sp=sp)
    assert sd == pytest.approx(10.0 * math.sqrt(20.0 + 400.0 / 10.0))
    assert (lo, hi) == pytest.approx((900.0 - B.Z80 * sd, 900.0 + B.Z80 * sd))
    late = B.kwh_range(900.0, 870.0, remaining_days=1.0, observed_days=29.0, sp=sp)
    assert late[2] < sd and late[0] >= 870.0                         # never below the kWh already used
    assert B.kwh_range(900.0, 900.0, remaining_days=0.0, observed_days=30.0, sp=sp) == (900.0, 900.0, 0.0)


def test_daily_totals_drop_days_with_a_hole_and_days_not_yet_over():
    rows = _rows(SEP1, SEP1 + timedelta(days=3), skip={SEP1 + timedelta(days=1, hours=5)})
    got = B.daily_totals(_ivs(rows), UTC, until=SEP1 + timedelta(days=2, hours=12))
    assert [d for d, _ in got] == [date(2026, 9, 1)]                 # 9-2 has a hole, 9-3 is not over


def test_daily_totals_count_a_25_hour_dst_day_as_complete():
    den = ZoneInfo("America/Denver")
    start = datetime(2026, 11, 1, tzinfo=den).astimezone(timezone.utc)
    rows = _rows(start, start + timedelta(hours=25))
    assert [d for d, _ in B.daily_totals(_ivs(rows), den, until=start + timedelta(days=2))] == [date(2026, 11, 1)]


# --- the bet ---------------------------------------------------------------------------------------


def _bet(rows, as_of, bills=(), origin="live"):
    return B.make_bet(tariff=flat_test_tariff(), tz=UTC, cycle_start_day=1, usage_point_id="UP1",
                      intervals=_ivs(rows), bills=[EnergyBillActualV1(**b) for b in bills], as_of=as_of, origin=origin)


def test_bet_point_is_project_period_and_dollars_are_the_kwh_range_itemized():
    rows = _rows(SEP1, OCT1 + timedelta(days=12))
    as_of = OCT1 + timedelta(days=12, hours=1)
    row = _bet(rows, as_of)
    ledger = B.build_ledger(flat_test_tariff(), UTC, 1, _ivs(rows), [])
    proj, _, _ = project_period(ledger, "UP1", OCT1, NOV1)
    assert row["projected_kwh"] == pytest.approx(proj.projected_kwh)
    assert row["projected_usd"] == pytest.approx(proj.projected_total_usd)
    assert row["period_start"] == date(2026, 10, 1) and row["period_end"] == date(2026, 11, 1)
    assert row["days_into_cycle"] == pytest.approx(12.0) and row["remaining_days"] == pytest.approx(19.0)
    assert row["coverage"] == 0.8 and row["kwh_lo80"] < row["projected_kwh"] < row["kwh_hi80"]
    t = flat_test_tariff()
    assert row["usd_lo80"] == pytest.approx(t.itemize(row["kwh_lo80"], start=date(2026, 10, 1), end=date(2026, 11, 1)).pre_tax_usd)
    assert row["usd_hi80"] == pytest.approx(t.itemize(row["kwh_hi80"], start=date(2026, 10, 1), end=date(2026, 11, 1)).pre_tax_usd)
    assert row["range_inputs"]["n_days"] == B.TRAILING_DAYS


def test_bet_without_seven_meter_days_is_stored_without_a_range():
    rows = _rows(OCT1, OCT1 + timedelta(days=3))
    row = _bet(rows, OCT1 + timedelta(days=3, hours=1))
    assert row["projected_kwh"] > 0 and row["coverage"] is None and row["kwh_lo80"] is None
    assert "fewer than 7" in row["range_reason"]


def test_less_than_a_day_into_the_cycle_is_a_gap_not_a_bet():
    rows = _rows(SEP1, OCT1 + timedelta(hours=10))
    assert _bet(rows, OCT1 + timedelta(hours=11)) == {"gap": "usage_incomplete", "period_start": date(2026, 10, 1)}


def test_replay_knows_only_what_the_feed_had_delivered():
    rows = _rows(SEP1, NOV1)
    as_of = OCT1 + timedelta(days=10)
    lag = timedelta(days=3)
    known = B.known_at(_ivs(rows), as_of, lag)
    assert max(iv.interval_end for iv in known) == as_of - lag
    cut = [r for r in rows if r["interval_end"] <= as_of - lag]
    assert B.make_bet(tariff=flat_test_tariff(), tz=UTC, cycle_start_day=1, usage_point_id="UP1", intervals=known,
                      bills=[], as_of=as_of, origin="replay") == _bet(cut, as_of, origin="replay")


def test_replay_lag_is_the_median_delivery_delay():
    rows = _rows(SEP1, SEP1 + timedelta(days=5), lag=timedelta(hours=70))
    assert B.replay_lag(_ivs(rows)) == timedelta(hours=70)
    assert B.replay_lag([]) == B.DEFAULT_REPLAY_LAG


# --- settlement ------------------------------------------------------------------------------------


def test_match_bill_uses_the_ledgers_read_snap_and_prefers_the_latest_reissue():
    a = EnergyBillActualV1(**_bill(date(2026, 10, 19), date(2026, 11, 18), 3000.0))
    reissued = EnergyBillActualV1(**{**_bill(date(2026, 10, 19), date(2026, 11, 18), 3010.0),
                                     "retrieved_at": a.retrieved_at + timedelta(days=1)})
    assert B.match_bill(date(2026, 10, 18), [a, reissued]) is reissued
    assert B.match_bill(date(2026, 9, 18), [a]) is None


def test_settlement_uses_the_pre_tax_basis_and_a_bill_without_tax_cannot_settle_dollars():
    row = {"kwh_lo80": 2900.0, "kwh_hi80": 3100.0, "usd_lo80": 90.0, "usd_hi80": 96.0}
    now = datetime(2026, 11, 22, tzinfo=UTC)
    s = B.settlement(row, EnergyBillActualV1(**_bill(date(2026, 10, 19), date(2026, 11, 18), 3000.0)), now=now)
    assert (s["billed_usd"], s["billed_basis"], s["kwh_in_80"], s["usd_in_80"]) == (95.0, "pre_tax", True, True)
    s = B.settlement(row, EnergyBillActualV1(**_bill(date(2026, 10, 19), date(2026, 11, 18), 3200.0, taxes=None)),
                     now=now)
    assert (s["billed_basis"], s["kwh_in_80"], s["usd_in_80"]) == ("tax_unknown", False, None)


# --- the lifecycle: bet saved -> bill arrives -> settled -> graded --------------------------------------


def test_lifecycle_daily_bets_settle_on_the_bill_and_grade_as_one_answer():
    feed = _rows(SEP1, NOV1 + timedelta(days=3))
    store = B.MemoryBillStore()
    bf = B.BillForecast(store, flat_test_tariff(), tz=UTC, cycle_start_day=1)
    bf.replayed = True                    # live only: every bet below is made on its own day
    for d in range(2, 31):
        now = OCT1 + timedelta(days=d, minutes=30)
        store.intervals = [r for r in feed if r["retrieved_at"] <= now]
        out = bf.tick(now)
        assert out["issued"] == f"bill:UP1:{now.date().isoformat()}"
        assert bf.tick(now + timedelta(hours=1))["issued"] is None          # once a day
    live = store.bill_bet_rows(SEP1)
    assert len(live) == 29 and {r["origin"] for r in live} == {"live"} and all(r["settled_at"] is None for r in live)

    truth = sum(r["energy_kwh"] for r in feed if OCT1 <= r["interval_start"] < NOV1)
    store.bills = [_bill(date(2026, 10, 1), date(2026, 11, 1), truth, taxes=4.0,
                         total=flat_test_tariff().itemize(truth, start=date(2026, 10, 1), end=date(2026, 11, 1)).pre_tax_usd + 4.0)]
    settled_at = NOV1 + timedelta(days=3, hours=2)
    assert bf.tick(settled_at)["settled"] == 29
    rows = store.bill_bet_rows(SEP1)
    assert all(r["billed_kwh"] == truth and r["kwh_in_80"] is not None for r in rows)
    late = rows[-1]
    assert abs(late["projected_kwh"] - truth) < abs(rows[0]["projected_kwh"] - truth)  # more data, closer

    state = build_state({"prediction_bill_forecast_kwh": rows, "prediction_bill_forecast_usd": rows,
                         **{n: [] for n, _, _ in SOURCES if not n.startswith("prediction_bill_forecast")}},
                        now=settled_at + timedelta(hours=1), window_days=7.0)
    rec = {f.faculty: f for f in state.faculties}
    kwh = rec["forecast.energy.bill_kwh"]
    assert (kwh.n_bets, kwh.n_graded) == (29, 1)               # 29 guesses, ONE answer
    assert kwh.status == "unknown" and kwh.gradeability == "ungradeable:too_few"
    assert "1 so far" in kwh.unlock and "29 bets on 1 answers" in " ".join(kwh.reasons)
    assert kwh.window_end - kwh.window_start == timedelta(days=400)
    assert rec["forecast.energy.bill_usd"].n_graded == 1


def test_a_second_meter_never_leaks_into_the_spread_or_the_replay():
    one = _rows(SEP1, OCT1 + timedelta(days=12))
    other = [{**r, "usage_point_id": "UP2", "energy_kwh": r["energy_kwh"] * (5 + (r["interval_start"].day % 4))}
             for r in one]
    now = OCT1 + timedelta(days=12, hours=2)
    a = B.MemoryBillStore(one)
    b = B.MemoryBillStore(one + other)
    for store in (a, b):
        B.BillForecast(store, flat_test_tariff(), tz=UTC, cycle_start_day=1, usage_point_id="UP1").tick(now)
    assert a.bill_bet_rows(SEP1) == b.bill_bet_rows(SEP1)


def test_a_bad_row_is_skipped_and_counted_not_fatal():
    rows = _rows(SEP1, OCT1 + timedelta(days=12))
    rows.append({**rows[0], "interval_start": OCT1 + timedelta(days=20), "interval_end": OCT1 + timedelta(days=20),
                 "energy_kwh": -1.0})
    bf = B.BillForecast(B.MemoryBillStore(rows, [{"bogus": 1}]), flat_test_tariff(), tz=UTC, cycle_start_day=1)
    out = bf.tick(OCT1 + timedelta(days=12, hours=2))
    assert out["skipped_rows"] == {"energy_usage_interval": 1, "energy_bill_actual": 1}
    assert out["issued"] is not None


def test_identical_days_give_no_range_rather_than_a_zero_width_one():
    rows = [{**r, "energy_kwh": 1.0} for r in _rows(SEP1, OCT1 + timedelta(days=12))]
    row = _bet(rows, OCT1 + timedelta(days=12, hours=1))
    assert row["coverage"] is None and row["range_reason"].startswith("zero daily spread")


def test_days_into_cycle_are_real_hours_across_the_fall_back_change():
    den = ZoneInfo("America/Denver")
    start = datetime(2026, 10, 18, tzinfo=den).astimezone(timezone.utc)
    rows = _rows(start - timedelta(days=10), start + timedelta(days=16))
    row = B.make_bet(tariff=flat_test_tariff(), tz=den, cycle_start_day=18, usage_point_id="UP1",
                     intervals=_ivs(rows), bills=[], as_of=start + timedelta(days=16, hours=1), origin="live")
    # 10-18 -> 11-18 Denver is 31 days + 1 h (the 11-01 fall-back): 16 days seen, 15 days + 1 h left.
    assert row["days_into_cycle"] == pytest.approx(16.0)
    assert row["remaining_days"] == pytest.approx(15.0 + 1 / 24)


def test_one_missing_meter_hour_does_not_freeze_the_bet_at_the_clock_change():
    """#3284 live shape: RMP's feed has no 2026-11-02 08:00 UTC hour; the 11-13 bet stopped at 14.1 days."""
    den = ZoneInfo("America/Denver")
    start = datetime(2026, 10, 18, tzinfo=den).astimezone(timezone.utc)
    hole = datetime(2026, 11, 2, 8, tzinfo=timezone.utc)
    metered_to = datetime(2026, 11, 11, 9, tzinfo=timezone.utc)
    rows = _rows(start - timedelta(days=10), metered_to, skip={hole})
    row = B.make_bet(tariff=flat_test_tariff(), tz=den, cycle_start_day=18, usage_point_id="UP1",
                     intervals=_ivs(rows), bills=[], as_of=datetime(2026, 11, 13, 7, tzinfo=timezone.utc),
                     origin="live")
    assert row["covered_through"] == metered_to
    assert row["days_into_cycle"] == pytest.approx((metered_to - start).total_seconds() / 86400.0)
    assert row["range_inputs"]["bridged_gap_hours"] == pytest.approx(1.0)
    # The hole is filled at the run rate, so the projection is not short by an hour.
    metered = sum(r["energy_kwh"] for r in rows if r["interval_start"] >= start)
    assert row["observed_kwh"] == pytest.approx(metered)
    assert row["projected_kwh"] > metered * (31 * 24 + 1) / ((metered_to - start).total_seconds() / 3600.0)


def test_replayed_bets_are_backfilled_once_and_never_graded():
    feed = _rows(SEP1, NOV1 + timedelta(days=3), lag=timedelta(days=3))
    store = B.MemoryBillStore(feed, [_bill(date(2026, 9, 1), date(2026, 10, 1), 700.0)])
    bf = B.BillForecast(store, flat_test_tariff(), tz=UTC, cycle_start_day=1)
    now = NOV1 + timedelta(days=3)
    out = bf.tick(now)
    assert out["replay"]["inserted"] > 20 and out["replay"]["lag_hours"] == pytest.approx(72.0)
    rows = store.bill_bet_rows(SEP1)
    assert {r["origin"] for r in rows if r["bet_date"] < now.date()} == {"replay"}
    sep = [r for r in rows if r["period_start"] == date(2026, 9, 1)]
    assert sep and all(r["settled_at"] is not None for r in sep)
    assert all(adapters.bill_kwh(r) is None for r in sep)          # replays are reported, not graded
    assert bf.tick(now + timedelta(hours=1))["replay"] is None      # once per process


# --- config parity with orion-energy ---------------------------------------------------------------


def _env(path: Path) -> dict:
    return dict(re.findall(r"^([A-Z_]+)=(.*)$", path.read_text(), flags=re.M))


def test_energy_keys_match_orion_energy():
    mine = _env(REPO / "services/orion-prediction/.env_example")
    energy = _env(REPO / "services/orion-energy/.env_example")
    for key in ("ENERGY_TARIFF_PATH", "ENERGY_TIMEZONE", "ENERGY_BILLING_CYCLE_START_DAY", "ENERGY_USAGE_POINT_ID"):
        assert mine[key] == energy[key], key
    from app.settings import Settings
    s = Settings()
    assert s.energy_tariff_path == mine["ENERGY_TARIFF_PATH"] and s.energy_timezone == mine["ENERGY_TIMEZONE"]
    assert str(s.energy_billing_cycle_start_day) == mine["ENERGY_BILLING_CYCLE_START_DAY"]
    assert (REPO / "config/energy" / Path(s.energy_tariff_path).name).exists()
