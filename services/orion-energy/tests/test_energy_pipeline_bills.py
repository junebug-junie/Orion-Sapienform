from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from app.pipeline import EnergyChannels, EnergyPipeline, StakesConfig
from orion.energy.importer_status import PortalStatus
from orion.energy.testing import hourly, make_test_ledger
from orion.schemas.energy import (
    ENERGY_BILL_ACTUAL_KIND,
    ENERGY_BILL_FORECAST_KIND,
    ENERGY_IMPORTER_STATUS_KIND,
    ENERGY_RECONCILE_KIND,
    ENERGY_STAKES_KIND,
    ENERGY_USAGE_KIND,
    EnergyBillActualV1,
    EnergyBillForecastV1,
)

S = datetime(2026, 9, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def _pipe(**kw) -> EnergyPipeline:
    return EnergyPipeline(ledger=make_test_ledger(), channels=EnergyChannels("u", "a", "r"), usage_point_id="UP1", **kw)


def _bill(retrieved=NOW) -> EnergyBillActualV1:
    return EnergyBillActualV1(
        source="file_drop", billing_period_start=date(2026, 9, 1), billing_period_end=date(2026, 9, 3),
        kwh_billed=470.0, taxes=3.0, current_charges=64.6, retrieved_at=retrieved,
    )


def test_bill_after_usage_publishes_bill_and_priced_reconcile() -> None:
    p = _pipe()
    p.ingest_intervals(hourly(S, 48, kwh=10.0), now=NOW)
    out = p.ingest_bills([_bill()], now=NOW)
    assert [o.kind for o in out] == [ENERGY_BILL_ACTUAL_KIND, ENERGY_RECONCILE_KIND]
    assert out[0].channel == "orion:energy:bill:actual"
    assert out[1].channel == "orion:energy:reconcile"
    assert out[1].payload.orion_total_usd == pytest.approx(59.6)


def test_bill_before_usage_reconciles_again_when_usage_lands() -> None:
    p = _pipe()
    first = p.ingest_bills([_bill()], now=NOW)
    assert first[1].payload.reconcile_gap == "no_usage"
    out = p.ingest_intervals(hourly(S, 48, kwh=10.0), now=NOW)
    recs = [o.payload for o in out if o.kind == ENERGY_RECONCILE_KIND]
    assert len(recs) == 1 and recs[0].orion_total_usd == pytest.approx(59.6)


def test_stale_bill_redelivery_is_ignored() -> None:
    p = _pipe()
    p.ingest_bills([_bill()], now=NOW)
    assert p.ingest_bills([_bill(retrieved=NOW - timedelta(days=1))], now=NOW) == []


def test_status_tick_with_no_data_is_stale_and_emits_no_usage() -> None:
    p = EnergyPipeline(ledger=make_test_ledger(), channels=EnergyChannels("u", "a", "r"))
    out = p.status_tick(now=NOW, portal=None, last_file_at=None)
    assert [o.kind for o in out] == [ENERGY_IMPORTER_STATUS_KIND, ENERGY_STAKES_KIND]
    assert ENERGY_USAGE_KIND not in {o.kind for o in out}
    assert out[0].payload.state == "stale" and out[0].payload.reason == "no_usage_yet"
    assert out[1].payload.pressure == "unknown"


def test_status_tick_over_forecast() -> None:
    p = _pipe()
    p.ingest_intervals(hourly(S, 72), now=NOW)
    p.replay_bills([
        EnergyBillForecastV1(
            source="file_drop", billing_period_start=date(2026, 9, 1), billing_period_end=date(2026, 10, 1),
            as_of=NOW, projected_total_usd=80.0, retrieved_at=NOW,
        )
    ])
    out = p.status_tick(now=NOW, portal=None, last_file_at=NOW)
    assert out[0].payload.state == "healthy"
    assert out[1].payload.pressure == "over_forecast"
    assert out[1].channel == "orion:energy:stakes:snapshot"


def test_status_tick_portal_reauth() -> None:
    p = _pipe(stakes=StakesConfig(portal_enabled=True))
    p.ingest_intervals(hourly(S, 72), now=NOW)
    portal = PortalStatus(state="reauth_required", reason="session_expired", last_attempt_at=NOW, last_success_at=None)
    out = p.status_tick(now=NOW, portal=portal, last_file_at=None)
    assert out[0].payload.state == "reauth_required"
    assert out[1].payload.pressure_reason == "importer_reauth_required"


def _forecast(retrieved=NOW, total=80.0) -> EnergyBillForecastV1:
    return EnergyBillForecastV1(
        source="rockymountain_power", billing_period_start=date(2026, 9, 1), as_of=NOW,
        projected_total_usd=total, retrieved_at=retrieved,
    )


def test_stale_forecast_redelivery_is_ignored() -> None:
    p = _pipe()
    first = p.ingest_bills([_forecast()], now=NOW)
    assert [o.kind for o in first] == [ENERGY_BILL_FORECAST_KIND, ENERGY_RECONCILE_KIND]
    # Same (billing_period_start, as_of) key, older retrieval: dropped, nothing republished.
    assert p.ingest_bills([_forecast(retrieved=NOW - timedelta(days=1), total=1.0)], now=NOW) == []
    # Newer retrieval of the same key wins and republishes.
    newer = p.ingest_bills([_forecast(retrieved=NOW + timedelta(hours=1), total=90.0)], now=NOW)
    assert newer[0].payload.projected_total_usd == 90.0


def test_actual_bill_moves_the_cycle_to_its_read_dates_and_reaccrues() -> None:
    """Usage the scheduled day put in a new cycle is re-accrued into the bill's period."""
    p = EnergyPipeline(
        ledger=make_test_ledger(cycle_start_day=18), channels=EnergyChannels("u", "a", "r"), usage_point_id="UP1",
    )
    p.ingest_intervals(hourly(datetime(2026, 9, 18, tzinfo=timezone.utc), 24 * 33), now=NOW)  # Sep 18 - Oct 20
    bill = EnergyBillActualV1(
        source="file_drop", billing_period_start=date(2026, 9, 18), billing_period_end=date(2026, 10, 19),
        kwh_billed=3106, current_charges=427.39, retrieved_at=NOW,
    )
    out = p.ingest_bills([bill], now=NOW)
    accrued = [o.payload for o in out if o.channel == "a"]
    by_hour = {a.interval_start: a.cycle_start for a in accrued}
    assert by_hour[datetime(2026, 10, 18, 12, tzinfo=timezone.utc)] == date(2026, 9, 18)
    assert by_hour[datetime(2026, 10, 19, 12, tzinfo=timezone.utc)] == date(2026, 10, 19)
    assert p.ingest_bills([bill.model_copy(update={"retrieved_at": NOW + timedelta(hours=1)})], now=NOW)[-1].kind \
        == ENERGY_RECONCILE_KIND  # same reads again: nothing re-accrues


def test_reissued_bill_replaces_the_older_read() -> None:
    p = EnergyPipeline(
        ledger=make_test_ledger(cycle_start_day=18), channels=EnergyChannels("u", "a", "r"), usage_point_id="UP1",
    )
    estimated = EnergyBillActualV1(
        source="file_drop", billing_period_start=date(2026, 9, 18), billing_period_end=date(2026, 10, 18),
        kwh_billed=3000, current_charges=400.0, retrieved_at=NOW,
    )
    corrected = estimated.model_copy(update={
        "billing_period_end": date(2026, 10, 19), "retrieved_at": NOW + timedelta(hours=1),
    })
    p.ingest_bills([estimated], now=NOW)
    p.ingest_bills([corrected], now=NOW)
    assert p._ledger.cycle_bounds(datetime(2026, 10, 18, 12, tzinfo=timezone.utc))[1] == datetime(
        2026, 10, 19, tzinfo=timezone.utc
    )
