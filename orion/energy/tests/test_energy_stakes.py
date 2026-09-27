from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from orion.energy.stakes import build_stakes_snapshot, current_forecast
from orion.energy.testing import UTC, hourly, make_test_ledger
from orion.schemas.energy import EnergyBillForecastV1, EnergyImporterStatusV1

S = datetime(2026, 9, 1, tzinfo=timezone.utc)
NOW = S + timedelta(hours=72)


def _importer(state="healthy") -> EnergyImporterStatusV1:
    return EnergyImporterStatusV1(state=state, reason="x", source="file_drop", as_of=NOW)


def _forecast(total=80.0, **over) -> EnergyBillForecastV1:
    base = dict(
        source="file_drop", billing_period_start=date(2026, 9, 1), billing_period_end=date(2026, 10, 1),
        as_of=NOW, projected_total_usd=total, retrieved_at=NOW,
    )
    base.update(over)
    return EnergyBillForecastV1(**base)


def _snap(forecast, importer="healthy", point="UP1"):
    led = make_test_ledger()
    for iv in hourly(S, 72):
        led.upsert(iv)
    return build_stakes_snapshot(
        ledger=led, usage_point_id=point, forecast=forecast, importer=_importer(importer),
        now=NOW, near_ratio=1.0, over_ratio=1.10,
    )


def test_over_forecast_hand_oracle() -> None:
    # Projected 88.40 (see reconcile oracle) / 80.00 = 1.105 >= 1.10.
    s = _snap(_forecast(80.0))
    assert s.pressure == "over_forecast"
    assert s.projected_to_forecast_ratio == pytest.approx(1.105)
    assert s.orion_projected_total_usd == pytest.approx(88.4)
    assert s.cycle_accumulated_kwh == pytest.approx(72.0)
    assert s.cycle_to_date_total_usd == pytest.approx(17.2)
    assert s.marginal_usd_per_kwh == pytest.approx(0.10)
    assert (s.cycle_start, s.cycle_end) == (date(2026, 9, 1), date(2026, 10, 1))
    assert s.covered_through == NOW


@pytest.mark.parametrize("total,pressure", [(85.0, "near_forecast"), (90.0, "normal")])
def test_near_and_normal(total, pressure) -> None:
    assert _snap(_forecast(total)).pressure == pressure


def test_unhealthy_importer_is_unknown_but_keeps_numbers() -> None:
    s = _snap(_forecast(80.0), importer="stale")
    assert (s.pressure, s.pressure_reason) == ("unknown", "importer_stale")
    assert s.projected_to_forecast_ratio is None
    assert s.cycle_to_date_total_usd == pytest.approx(17.2)


def test_no_forecast_is_unknown() -> None:
    s = _snap(None)
    assert (s.pressure, s.pressure_reason) == ("unknown", "no_forecast_total")
    assert s.orion_projected_total_usd == pytest.approx(88.4)
    assert (s.cycle_start, s.cycle_end) == (date(2026, 9, 1), date(2026, 10, 1))


def test_kwh_only_forecast_is_unknown() -> None:
    s = _snap(_forecast(None, projected_kwh=700.0))
    assert s.pressure_reason == "no_forecast_total"


def test_no_usage_point_is_unknown() -> None:
    s = _snap(_forecast(80.0), point=None)
    assert (s.pressure, s.pressure_reason) == ("unknown", "no_usage_point")


def test_current_forecast_picks_latest_live_period() -> None:
    old = _forecast(70.0, billing_period_start=date(2026, 8, 1), billing_period_end=date(2026, 9, 1))
    early = _forecast(75.0, as_of=NOW - timedelta(days=1), retrieved_at=NOW - timedelta(days=1))
    late = _forecast(80.0)
    assert current_forecast([old, early, late], now=NOW, tz=UTC) is late
    assert current_forecast([old], now=NOW, tz=UTC) is None
