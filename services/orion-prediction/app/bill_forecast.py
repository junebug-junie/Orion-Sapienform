"""Power-bill bet: once a day, Orion's own projection of this bill's kWh and pre-tax dollars, with an
80 % range, settled when Rocky Mountain Power's bill for the period lands.

The point estimate is ``orion.energy.reconcile.project_period`` (the run rate the stakes snapshot
already uses, priced by ``Tariff.itemize``); nothing is re-priced here.

The range. The projection is ``observed + R x (observed run rate)``: R days still to come, priced at
the mean of the n days seen so far. If the house's daily kWh is a stationary series with day-to-day
spread s and lag-1 autocorrelation rho, the error of that projected total is

    var = s^2 x VIF x (R + R^2 / n),     VIF = (1 + rho) / (1 - rho)

-- the variance of a sum of R future days (R s^2) plus the variance of R times a mean estimated from n
days (R^2 s^2 / n), both inflated for day-to-day persistence (the long-run variance of an AR(1)). This
is the textbook prediction interval for a future total from an estimated mean. s and rho come from the
last ``TRAILING_DAYS`` complete local days the house's meter had delivered at the bet's time (any
period). The 80 % kWh range is +-1.2816 sd (normal), floored at the kWh already used; the dollar range
is the same two kWh numbers priced by ``Tariff.itemize`` (monotone in kWh). It does NOT cover tariff
error (Orion vs RMP on the 2026-10-19 bill: -$3.55, -0.9 %) or the meter-read date moving.

Why not the run-rate error by days-into-cycle measured on past bills: only one completed period
(2026-09-18 -> 10-19) has meter data at all (the feed has a hole 08-07 -> 09-17), and one period cannot
give a range. The range above is a stated model; the bills grade it.

Two kinds of row: ``live`` (made by this loop on that day) and ``replay`` (backfilled from stored
meter history as of a past day, knowing only the intervals that would have arrived by then: the meter
feed lags ~3 days, measured from ``retrieved_at``). Replays are reported, never graded by the
self-monitor: this method was chosen while looking at that history.
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from orion.energy.ledger import READ_SNAP_DAYS, UsageLedger
from orion.energy.reconcile import _utility_pretax, project_period
from orion.energy.tariff import Tariff
from orion.schemas.energy import EnergyBillActualV1, EnergyUsageIntervalV1

logger = logging.getLogger("orion-prediction.bill_forecast")

MODEL = "bill_run_rate_v1"
COVERAGE = 0.8
Z80 = 1.2815515655446004        # normal 90th percentile: a central 80 % range
TRAILING_DAYS = 28              # complete local days used for the daily spread
MIN_SPREAD_DAYS = 7             # fewer than this: no range (the bet is stored without one)
MIN_RHO_PAIRS = 5               # consecutive-day pairs needed to estimate persistence (else rho = 0)
RHO_MAX = 0.8                   # VIF cap (9x); a negative rho is read as 0 (never narrows the range)
DEFAULT_REPLAY_LAG = timedelta(days=3)
HISTORY_DAYS = 400              # meter history read for a bet / replay


@dataclass(frozen=True)
class Spread:
    s_kwh: float
    rho: float
    n_days: int
    n_pairs: int

    @property
    def vif(self) -> float:
        return (1.0 + self.rho) / (1.0 - self.rho)


def _day_bounds(day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """Local midnight to midnight as UTC instants: two aware datetimes sharing one tzinfo subtract as
    wall clock (24 h even on a 25 h DST day), so the length must be taken in UTC."""
    a = datetime.combine(day, time(0), tzinfo=tz).astimezone(timezone.utc)
    b = datetime.combine(day + timedelta(days=1), time(0), tzinfo=tz).astimezone(timezone.utc)
    return a, b


def daily_totals(intervals: Iterable[EnergyUsageIntervalV1], tz: ZoneInfo, *, until: datetime) -> list[tuple[date, float]]:
    """kWh per complete local day that ended at or before ``until``. A day missing any hour is left out
    (a hole is unknown usage, never zero). DST days count 23 / 25 h."""
    secs: dict[date, float] = {}
    kwh: dict[date, float] = {}
    for iv in intervals:
        day = iv.interval_start.astimezone(tz).date()
        secs[day] = secs.get(day, 0.0) + (iv.interval_end - iv.interval_start).total_seconds()
        kwh[day] = kwh.get(day, 0.0) + iv.energy_kwh
    out = []
    for day in sorted(kwh):
        a, b = _day_bounds(day, tz)
        if b <= until and abs(secs[day] - (b - a).total_seconds()) < 1.0:
            out.append((day, kwh[day]))
    return out


def spread(days: Sequence[tuple[date, float]]) -> Optional[Spread]:
    """Daily spread and lag-1 persistence of the last ``TRAILING_DAYS`` complete days, or None."""
    days = list(days)[-TRAILING_DAYS:]
    if len(days) < MIN_SPREAD_DAYS:
        return None
    vals = [v for _, v in days]
    s = statistics.stdev(vals)
    mean = statistics.fmean(vals)
    var = statistics.pvariance(vals)
    by_day = dict(days)
    pairs = [(by_day[d], by_day[d + timedelta(days=1)]) for d, _ in days if d + timedelta(days=1) in by_day]
    rho = 0.0
    if len(pairs) >= MIN_RHO_PAIRS and var > 0:
        rho = sum((a - mean) * (b - mean) for a, b in pairs) / len(pairs) / var
    return Spread(s_kwh=s, rho=min(RHO_MAX, max(0.0, rho)), n_days=len(days), n_pairs=len(pairs))


def kwh_range(projected: float, observed: float, *, remaining_days: float, observed_days: float,
              sp: Spread) -> tuple[float, float, float]:
    """(lo, hi, sd) of the period-end kWh at 80 %. ``lo`` never under the kWh already used."""
    if remaining_days <= 0:
        return projected, projected, 0.0
    var = sp.s_kwh ** 2 * sp.vif * (remaining_days + remaining_days ** 2 / observed_days)
    sd = math.sqrt(max(0.0, var))
    return max(observed, projected - Z80 * sd), projected + Z80 * sd, sd


def build_ledger(tariff: Tariff, tz: ZoneInfo, cycle_start_day: int, intervals: Iterable[EnergyUsageIntervalV1],
                 read_dates: Iterable[date]) -> UsageLedger:
    ledger = UsageLedger(tariff, tz=tz, cycle_start_day=cycle_start_day)
    ledger.set_read_dates(read_dates)
    for iv in intervals:
        ledger.upsert(iv)
    return ledger


def read_dates(bills: Iterable[EnergyBillActualV1]) -> list[date]:
    out: set[date] = set()
    for b in bills:
        out.update((b.billing_period_start, b.billing_period_end))
    return sorted(out)


def make_bet(*, tariff: Tariff, tz: ZoneInfo, cycle_start_day: int, usage_point_id: str,
             intervals: Sequence[EnergyUsageIntervalV1], bills: Sequence[EnergyBillActualV1],
             as_of: datetime, origin: str) -> dict:
    """One bet as of ``as_of`` from what Orion knew then: ``intervals`` and ``bills`` must already be cut
    to that knowledge. Returns the row, or ``{"gap": reason}`` when there is nothing to bet on."""
    ledger = build_ledger(tariff, tz, cycle_start_day, intervals, read_dates(bills))
    start, end = ledger.cycle_bounds(as_of)
    projection, gap, covered = project_period(ledger, usage_point_id, start, end)
    if projection is None:
        return {"gap": gap or "no_projection", "period_start": start.astimezone(tz).date()}
    observed_days = (projection.covered_through - start).total_seconds() / 86400.0
    remaining_days = max(0.0, (end - projection.covered_through).total_seconds() / 86400.0)
    sp = spread(daily_totals(intervals, tz, until=projection.covered_through))
    p_start, p_end = start.astimezone(tz).date(), end.astimezone(tz).date()
    row = {
        "bet_id": f"bill:{usage_point_id}:{as_of.astimezone(tz).date().isoformat()}",
        "usage_point_id": usage_point_id,
        "bet_date": as_of.astimezone(tz).date(),
        "origin": origin,
        "as_of": as_of,
        "model": MODEL,
        "period_start": p_start,
        "period_end": p_end,
        "covered_through": projection.covered_through,
        "days_into_cycle": observed_days,
        "remaining_days": remaining_days,
        "observed_kwh": projection.observed_kwh,
        "projected_kwh": projection.projected_kwh,
        "projected_usd": projection.projected_total_usd,
        "coverage": None, "kwh_lo80": None, "kwh_hi80": None, "usd_lo80": None, "usd_hi80": None,
        "range_reason": None,
        "range_inputs": {},
        "tariff_version": tariff.version,
    }
    if sp is None:
        row["range_reason"] = f"fewer than {MIN_SPREAD_DAYS} complete meter days before the bet"
        return row
    lo, hi, sd = kwh_range(projection.projected_kwh, projection.observed_kwh, remaining_days=remaining_days,
                           observed_days=observed_days, sp=sp)
    row.update(
        coverage=COVERAGE, kwh_lo80=lo, kwh_hi80=hi,
        usd_lo80=tariff.itemize(lo, start=p_start, end=p_end).pre_tax_usd,
        usd_hi80=tariff.itemize(hi, start=p_start, end=p_end).pre_tax_usd,
        range_inputs={"s_kwh_per_day": sp.s_kwh, "rho": sp.rho, "vif": sp.vif, "n_days": sp.n_days,
                      "n_pairs": sp.n_pairs, "sd_kwh": sd, "z": Z80},
    )
    return row


def match_bill(period_start: date, bills: Iterable[EnergyBillActualV1]) -> Optional[EnergyBillActualV1]:
    """The bill for a bet's period: the one whose start is nearest the bet's assumed start, within the
    ledger's own read-snap rule (a scheduled start of 10-18 is the 10-19 meter read). Latest retrieval
    wins a re-issued bill."""
    near = [b for b in bills if abs((b.billing_period_start - period_start).days) <= READ_SNAP_DAYS]
    if not near:
        return None
    return min(near, key=lambda b: (abs((b.billing_period_start - period_start).days), -b.retrieved_at.timestamp()))


def settlement(row: Mapping, bill: EnergyBillActualV1, *, now: datetime) -> dict:
    """The settled fields for one bet. Dollars on the pre-tax basis of ``reconcile._utility_pretax``; a
    bill without a tax line cannot settle the dollar claim (``usd_in_80`` stays None)."""
    pretax, basis = _utility_pretax(bill)
    kwh_in = usd_in = None
    if row.get("kwh_lo80") is not None:
        kwh_in = row["kwh_lo80"] <= bill.kwh_billed <= row["kwh_hi80"]
    if row.get("usd_lo80") is not None and basis == "pre_tax":
        usd_in = row["usd_lo80"] <= pretax <= row["usd_hi80"]
    return {
        "settled_at": now,
        "bill_period_start": bill.billing_period_start,
        "bill_period_end": bill.billing_period_end,
        "billed_kwh": bill.kwh_billed,
        "billed_usd": pretax,
        "billed_basis": basis,
        "kwh_in_80": kwh_in,
        "usd_in_80": usd_in,
    }


def replay_lag(intervals: Iterable[EnergyUsageIntervalV1]) -> timedelta:
    """Median delay from an hour's end to the batch that delivered it (each ``retrieved_at`` batch's
    newest hour). Live 2026-11-13: ~3 days."""
    newest: dict[datetime, datetime] = {}
    for iv in intervals:
        got = iv.retrieved_at
        if got not in newest or iv.interval_end > newest[got]:
            newest[got] = iv.interval_end
    lags = [(got - end).total_seconds() for got, end in newest.items() if got > end]
    return timedelta(seconds=statistics.median(lags)) if lags else DEFAULT_REPLAY_LAG


def known_at(intervals: Iterable[EnergyUsageIntervalV1], as_of: datetime, lag: timedelta) -> list[EnergyUsageIntervalV1]:
    """What the meter feed would have delivered by ``as_of`` at the feed's usual lag. The first import
    (2026-09-28) delivered two months at once, so ``retrieved_at`` cannot be used for older days."""
    cut = as_of.astimezone(timezone.utc) - lag    # UTC: local-time arithmetic would slip an hour over DST
    return [iv for iv in intervals if iv.interval_end <= cut]


def pick_usage_point(intervals: Iterable[EnergyUsageIntervalV1], configured: str) -> tuple[Optional[str], Optional[str]]:
    points = sorted({iv.usage_point_id for iv in intervals})
    if configured:
        return (configured, None) if configured in points else (None, f"usage point {configured!r} has no meter data")
    if len(points) == 1:
        return points[0], None
    return None, ("no meter data" if not points else f"several usage points {points}; set ENERGY_USAGE_POINT_ID")


def bills_for_point(bills: Iterable[EnergyBillActualV1], point: str) -> list[EnergyBillActualV1]:
    return [b for b in bills if b.usage_point_id in (None, point)]


class BillForecast:
    """Daily live bet, one-time replay backfill per process, settlement. Store: app/store.py (bill section)."""

    def __init__(self, store, tariff: Tariff, *, tz: ZoneInfo, cycle_start_day: int, usage_point_id: str = ""):
        self.store = store
        self.tariff = tariff
        self.tz = tz
        self.cycle_start_day = cycle_start_day
        self.usage_point_id = usage_point_id
        self.replayed = False
        self.last_pass: dict = {}

    def _bet(self, intervals, bills, as_of: datetime, origin: str, point: str) -> dict:
        return make_bet(tariff=self.tariff, tz=self.tz, cycle_start_day=self.cycle_start_day, usage_point_id=point,
                        intervals=intervals, bills=bills, as_of=as_of, origin=origin)

    def replay(self, intervals, bills, point: str, now: datetime) -> dict:
        """Backfill every past local day without a row, as of 00:05 that day. Returns counts."""
        if not intervals:
            return {"inserted": 0, "gaps": 0}
        lag = replay_lag(intervals)
        have = self.store.bill_bet_dates(point)
        first = min(iv.interval_start for iv in intervals).astimezone(self.tz).date() + timedelta(days=1)
        today = now.astimezone(self.tz).date()
        inserted = gaps = 0
        day = first
        while day < today:
            if day not in have:
                as_of = datetime.combine(day, time(0, 5), tzinfo=self.tz)
                known = known_at(intervals, as_of, lag)
                known_bills = [b for b in bills if b.retrieved_at <= as_of]
                row = self._bet(known, known_bills, as_of, "replay", point)
                if "gap" in row:
                    gaps += 1
                elif self.store.insert_bill_bet(row):
                    inserted += 1
            day += timedelta(days=1)
        return {"inserted": inserted, "gaps": gaps, "lag_hours": lag.total_seconds() / 3600.0}

    def settle(self, bills: Sequence[EnergyBillActualV1], now: datetime) -> int:
        n = 0
        for row in self.store.pending_bill_bets():
            point_bills = bills_for_point(bills, row["usage_point_id"])
            bill = match_bill(row["period_start"], point_bills)
            if bill is None:
                continue
            n += self.store.settle_bill_bet(row["bet_id"], settlement(row, bill, now=now))
        return n

    def tick(self, now: Optional[datetime] = None) -> dict:
        now = now or datetime.now(timezone.utc)
        intervals = [EnergyUsageIntervalV1(**r) for r in self.store.energy_intervals(now - timedelta(days=HISTORY_DAYS))]
        bills = [EnergyBillActualV1(**r) for r in self.store.energy_bills()]
        point, why = pick_usage_point(intervals, self.usage_point_id)
        out: dict = {"at": now, "issued": None, "gap": None, "replay": None, "settled": 0}
        if point is None:
            out["gap"] = why
            self.last_pass = out
            return out
        bills = bills_for_point(bills, point)
        if not self.replayed:
            out["replay"] = self.replay(intervals, bills, point, now)
            self.replayed = True
        today = now.astimezone(self.tz).date()
        if today not in self.store.bill_bet_dates(point):
            row = self._bet(intervals, bills, now, "live", point)
            if "gap" in row:
                out["gap"] = row["gap"]
            else:
                out["issued"] = self.store.insert_bill_bet(row)
                logger.info("bill_bet_placed bet_id=%s period=%s projected_kwh=%.0f range=%s..%s usd=%.2f",
                            row["bet_id"], row["period_start"], row["projected_kwh"], row["kwh_lo80"],
                            row["kwh_hi80"], row["projected_usd"])
        out["settled"] = self.settle(bills, now)
        if out["settled"]:
            logger.info("bill_bets_settled n=%d", out["settled"])
        self.last_pass = out
        return out

    def live(self, now: Optional[datetime] = None) -> dict:
        now = now or datetime.now(timezone.utc)
        rows = self.store.bill_bet_rows(now - timedelta(days=HISTORY_DAYS))
        return {"model": MODEL, "coverage": COVERAGE, "current": rows[-1] if rows else None,
                "by_period": grade_by_period(rows)}


def grade_by_period(rows: Iterable[Mapping]) -> list[dict]:
    """Per bill period and origin: how many bets, how many settled, the share of 80 % ranges that held
    (kWh, $), and the mean absolute kWh / $ miss. One bill is ONE answer: these shares describe that
    bill, they are not n independent trials."""
    groups: dict[tuple, list[Mapping]] = {}
    for r in rows:
        groups.setdefault((r["period_start"], r["origin"]), []).append(r)
    out = []
    for (start, origin), rs in sorted(groups.items()):
        settled = [r for r in rs if r.get("settled_at") is not None]
        kin = [r["kwh_in_80"] for r in settled if r.get("kwh_in_80") is not None]
        uin = [r["usd_in_80"] for r in settled if r.get("usd_in_80") is not None]
        out.append({
            "period_start": start, "origin": origin, "n_bets": len(rs), "n_settled": len(settled),
            "billed_kwh": settled[0]["billed_kwh"] if settled else None,
            "billed_usd": settled[0]["billed_usd"] if settled else None,
            "kwh_in_80_share": (sum(kin) / len(kin)) if kin else None,
            "usd_in_80_share": (sum(uin) / len(uin)) if uin else None,
            "mean_abs_kwh_miss": (statistics.fmean(abs(r["projected_kwh"] - r["billed_kwh"]) for r in settled)
                                  if settled else None),
            "mean_abs_usd_miss": (statistics.fmean(abs(r["projected_usd"] - r["billed_usd"]) for r in settled)
                                  if settled else None),
        })
    return out


class MemoryBillStore:
    """The bill section of ``PostgresStore`` in memory: the dry-run replay eval and the tests. The energy
    rows are given; bets live in a dict."""

    def __init__(self, intervals: Sequence[dict] = (), bills: Sequence[dict] = ()):
        self.intervals = list(intervals)
        self.bills = list(bills)
        self.bets: dict[str, dict] = {}

    def energy_intervals(self, since: datetime) -> list[dict]:
        return [r for r in self.intervals if r["interval_start"] >= since]

    def energy_bills(self) -> list[dict]:
        return list(self.bills)

    def bill_bet_dates(self, usage_point_id: str) -> set:
        return {r["bet_date"] for r in self.bets.values() if r["usage_point_id"] == usage_point_id}

    def insert_bill_bet(self, row: dict) -> Optional[str]:
        if row["bet_id"] in self.bets or row["bet_date"] in self.bill_bet_dates(row["usage_point_id"]):
            return None
        self.bets[row["bet_id"]] = {**row, "settled_at": None}
        return row["bet_id"]

    def pending_bill_bets(self) -> list[dict]:
        return sorted((dict(r) for r in self.bets.values() if r["settled_at"] is None), key=lambda r: r["as_of"])

    def settle_bill_bet(self, bet_id: str, fields: dict) -> int:
        row = self.bets.get(bet_id)
        if row is None or row["settled_at"] is not None:
            return 0
        row.update(fields)
        return 1

    def bill_bet_rows(self, since: datetime) -> list[dict]:
        return sorted((dict(r) for r in self.bets.values() if r["as_of"] >= since), key=lambda r: r["as_of"])
