"""#3284: RMP's 2026-11-01 fall-back file stamped two hours alike, leaving 2026-11-02 08:00 UTC empty,
and that one empty hour froze every cost and bill number from then on."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from orion.energy.espi import parse_espi
from orion.energy.importer_status import DEFAULT_STALE_AFTER_HOURS
from orion.energy.ledger import MAX_BRIDGED_GAP, UsageLedger, bridged_seconds
from orion.energy.reconcile import project_period
from orion.energy.stakes import build_stakes_snapshot
from orion.energy.testing import flat_test_tariff, hourly, make_test_ledger
from orion.schemas.energy import EnergyBillForecastV1, EnergyImporterStatusV1

DENVER = ZoneInfo("America/Denver")
RETRIEVED = datetime(2026, 11, 4, 5, 46, tzinfo=timezone.utc)
H = 3600
# The real blocks (processed/*2026-11-0[12].xml): block start, block seconds, reading stamps.
NOV1_BLOCK = int(datetime(2026, 11, 1, 7, tzinfo=timezone.utc).timestamp())
NOV2_BLOCK = int(datetime(2026, 11, 2, 8, tzinfo=timezone.utc).timestamp())
_FIRST = datetime(2026, 11, 1, 8, tzinfo=timezone.utc)
_HOLE = datetime(2026, 11, 2, 8, tzinfo=timezone.utc)


def _rmp_nov1_stamps() -> list[int]:
    # 25 readings, hour 10:00 UTC stamped twice, last at 11-02 07:00 (as delivered).
    hours = [0, 1, 2, 2] + list(range(3, 24))
    return [NOV1_BLOCK + H + h * H for h in hours]


def _feed(blocks: list[tuple[int, int, list[tuple[int, int]]]]) -> bytes:
    entries = []
    for i, (b_start, b_dur, readings) in enumerate(blocks):
        rs = "".join(
            f"<espi:IntervalReading><espi:timePeriod><espi:duration>{H}</espi:duration>"
            f"<espi:start>{s}</espi:start></espi:timePeriod><espi:value>{v}</espi:value></espi:IntervalReading>"
            for s, v in readings
        )
        entries.append(
            f'<entry><link rel="self" href="/User/u/UsagePoint/01/MeterReading/01/IntervalBlock/{i}"/>'
            f"<content><espi:IntervalBlock><espi:interval><espi:duration>{b_dur}</espi:duration>"
            f"<espi:start>{b_start}</espi:start></espi:interval>{rs}</espi:IntervalBlock></content></entry>"
        )
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:espi="http://naesb.org/espi">'
        '<entry><link rel="self" href="/User/u/UsagePoint/01/MeterReading/01"/>'
        '<link rel="related" href="/ReadingType/1"/><content><espi:MeterReading/></content></entry>'
        '<entry><link rel="self" href="/ReadingType/1"/><content><espi:ReadingType>'
        "<espi:flowDirection>1</espi:flowDirection><espi:powerOfTenMultiplier>3</espi:powerOfTenMultiplier>"
        "<espi:uom>72</espi:uom></espi:ReadingType></content></entry>" + "".join(entries) + "</feed>"
    ).encode()


def _nov1_nov2_feed() -> bytes:
    nov1 = [(s, 3 if i != 3 else 7) for i, s in enumerate(_rmp_nov1_stamps())]
    nov2 = [(NOV2_BLOCK + H + h * H, 2) for h in range(24)]
    return _feed([(NOV1_BLOCK, 25 * H, nov1), (NOV2_BLOCK, 24 * H, nov2)])


# --- ingest ----------------------------------------------------------------------------------------


def test_fall_back_block_with_a_doubled_stamp_is_laid_end_to_end() -> None:
    rows = parse_espi(_nov1_nov2_feed(), retrieved_at=RETRIEVED, source="rockymountain_power")
    starts = [r.interval_start for r in rows]
    assert len(rows) == 49 and len(set(starts)) == 49
    assert starts == [_FIRST + timedelta(hours=h) for h in range(49)]   # no hole, no collision
    assert _HOLE in starts
    # Every reading kept, in order: the doubled hour's second reading (7 kWh) is now 11:00.
    assert sum(r.energy_kwh for r in rows) == pytest.approx(24 * 3 + 7 + 24 * 2)
    assert next(r for r in rows if r.interval_start == _FIRST + timedelta(hours=3)).energy_kwh == pytest.approx(7)


def test_doubled_stamp_that_the_block_length_does_not_explain_is_left_as_delivered() -> None:
    stamps = _rmp_nov1_stamps()
    # Block claims 24 h but holds 25 readings: no basis to re-stamp anything.
    xml = _feed([(NOV1_BLOCK, 24 * H, [(s, 3) for s in stamps])])
    rows = parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")
    assert [int(r.interval_start.timestamp()) for r in rows] == stamps


def test_doubled_stamp_plus_a_hole_elsewhere_is_left_as_delivered() -> None:
    # Same reading count as the block, but 12:00 is skipped: which reading belongs there is unknowable.
    stamps = _rmp_nov1_stamps()
    stamps = stamps[:6] + [s + H for s in stamps[6:]]
    xml = _feed([(NOV1_BLOCK, 25 * H, [(s, 3) for s in stamps])])
    rows = parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")
    assert [int(r.interval_start.timestamp()) for r in rows] == stamps


def test_normal_block_is_untouched() -> None:
    readings = [(NOV2_BLOCK + H + h * H, 2) for h in range(24)]
    rows = parse_espi(_feed([(NOV2_BLOCK, 24 * H, readings)]), retrieved_at=RETRIEVED, source="file_drop")
    assert [int(r.interval_start.timestamp()) for r in rows] == [s for s, _ in readings]


# --- ledger ----------------------------------------------------------------------------------------


def _denver_ledger_with_hole() -> tuple[UsageLedger, datetime]:
    """10-18 -> 11-18 Denver cycle, metered through 11-11 08:00 UTC, 11-02 08:00 UTC missing (live shape)."""
    led = UsageLedger(flat_test_tariff(), tz=DENVER, cycle_start_day=18)
    start = datetime(2026, 10, 18, tzinfo=DENVER)
    end_metered = datetime(2026, 11, 11, 9, tzinfo=timezone.utc)
    hours = int((end_metered - start.astimezone(timezone.utc)).total_seconds() // H)
    skip = int((_HOLE - start.astimezone(timezone.utc)).total_seconds() // H)
    for iv in hourly(start, hours, kwh=1.0, skip=frozenset({skip})):
        led.upsert(iv)
    return led, start


def test_one_missing_hour_no_longer_freezes_coverage() -> None:
    led, start = _denver_ledger_with_hole()
    _, end = led.cycle_bounds(start)
    prefix, covered = led.window_prefix("UP1", start, end)
    assert covered == datetime(2026, 11, 11, 9, tzinfo=timezone.utc)
    assert bridged_seconds(prefix, start) == H
    # Strict mode (billed-total reconcile) still sees the hole.
    _, strict = led.window_prefix("UP1", start, end, bridge=False)
    assert strict == _HOLE


def test_projection_imputes_the_hole_at_the_run_rate_never_as_zero() -> None:
    led, start = _denver_ledger_with_hole()
    _, end = led.cycle_bounds(start)
    proj, gap, covered = project_period(led, "UP1", start, end)
    assert gap is None and proj is not None
    metered_h = (covered - start.astimezone(timezone.utc)).total_seconds() / H - 1
    assert proj.observed_kwh == pytest.approx(metered_h)                 # 1 kWh per metered hour
    assert proj.bridged_hours == pytest.approx(1.0)
    period_h = (end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds() / H  # 745: fall-back
    assert proj.projected_kwh == pytest.approx(period_h)                 # the hole is filled at 1 kWh/h


def test_holes_past_the_per_hole_or_total_limit_still_stop_coverage() -> None:
    s = datetime(2026, 9, 1, tzinfo=timezone.utc)
    long_hole = int(MAX_BRIDGED_GAP.total_seconds() // H) + 1
    led = make_test_ledger()
    for iv in hourly(s, 20, skip=frozenset(range(5, 5 + long_hole))):
        led.upsert(iv)
    assert led.window_prefix("UP1", s, s + timedelta(hours=20))[1] == s + timedelta(hours=5)
    # Three 3 h holes: the first two fit the 6 h budget, the third does not.
    led = make_test_ledger()
    for iv in hourly(s, 40, skip=frozenset({*range(5, 8), *range(15, 18), *range(25, 28)})):
        led.upsert(iv)
    assert led.window_prefix("UP1", s, s + timedelta(hours=40))[1] == s + timedelta(hours=25)


def test_a_short_hole_at_the_cycle_start_is_bridged_on_purpose() -> None:
    s = datetime(2026, 9, 1, tzinfo=timezone.utc)
    led = make_test_ledger()
    for iv in hourly(s, 48, skip=frozenset({0})):
        led.upsert(iv)
    prefix, covered = led.window_prefix("UP1", s, s + timedelta(hours=48))
    assert covered == s + timedelta(hours=48) and bridged_seconds(prefix, s) == H
    proj, gap, _ = project_period(led, "UP1", s, s + timedelta(hours=48))
    assert gap is None and proj.projected_kwh == pytest.approx(48.0)      # 47 metered + 1 imputed


# --- lifecycle: coverage walks past the hole as the feed delivers ------------------------------------


def test_stakes_coverage_moves_past_the_hole_day_by_day() -> None:
    led = UsageLedger(flat_test_tariff(), tz=DENVER, cycle_start_day=18)
    start = datetime(2026, 10, 18, tzinfo=DENVER)
    start_utc = start.astimezone(timezone.utc)
    hole_h = int((_HOLE - start_utc).total_seconds() // H)
    forecast = EnergyBillForecastV1(
        source="file_drop", billing_period_start=date(2026, 10, 18), billing_period_end=date(2026, 11, 18),
        as_of=start_utc, projected_total_usd=200.0, retrieved_at=start_utc,
    )
    delivered = 0
    for day in range(15, 25):                     # 11-02 .. 11-11: the days that froze live
        upto = day * 24
        for iv in hourly(start, upto, kwh=1.0, skip=frozenset({hole_h})):
            led.upsert(iv)
        delivered = upto
        now = start_utc + timedelta(hours=delivered + 24)
        importer = EnergyImporterStatusV1(state="healthy", reason="usage_fresh", source="file_drop", as_of=now)
        snap = build_stakes_snapshot(
            ledger=led, usage_point_id="UP1", forecast=forecast, importer=importer, now=now,
            near_ratio=1.0, over_ratio=1.10, stale_after_hours=DEFAULT_STALE_AFTER_HOURS,
        )
        assert snap.covered_through == start_utc + timedelta(hours=delivered)
        assert not snap.pressure_reason.startswith("coverage_lag")
        assert snap.pressure != "unknown"
