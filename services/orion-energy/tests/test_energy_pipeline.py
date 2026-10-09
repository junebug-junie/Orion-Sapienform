from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.pipeline import EnergyChannels, EnergyPipeline
from orion.energy.ledger import UsageLedger
from orion.energy.tariff import load_tariff
from orion.schemas.energy import (
    ENERGY_ACCRUED_KIND,
    ENERGY_RUN_COST_KIND,
    ENERGY_USAGE_KIND,
    EnergyUsageIntervalV1,
)
from orion.schemas.power import PowerIntentSettledV1

REPO = Path(__file__).resolve().parents[3]
TARIFF = load_tariff(REPO / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml")
CH = EnergyChannels(usage="u", accrued="a", run_cost="r")
T = datetime(2026, 9, 10, 18, tzinfo=timezone.utc)
NOW = datetime(2026, 9, 10, 19, 5, tzinfo=timezone.utc)
_TZ = ZoneInfo("America/Denver")


def _pipeline(**kw) -> EnergyPipeline:
    led = UsageLedger(TARIFF, tz=_TZ, cycle_start_day=1)
    return EnergyPipeline(ledger=led, channels=CH, **kw)


def _iv(start: datetime, kwh: float, retrieved: datetime = NOW, point: str = "UP123") -> EnergyUsageIntervalV1:
    return EnergyUsageIntervalV1(
        source="file_drop",
        usage_point_id=point,
        interval_start=start,
        interval_end=start + timedelta(hours=1),
        energy_kwh=kwh,
        retrieved_at=retrieved,
    )


def _prepare(*intervals: EnergyUsageIntervalV1) -> list[EnergyUsageIntervalV1]:
    """Contiguous cycle prefix: zero-kWh span from cycle start, plus T-1h when targeting T."""
    if not intervals:
        return []
    led = UsageLedger(TARIFF, tz=_TZ, cycle_start_day=1)
    by_point: dict[str, list[EnergyUsageIntervalV1]] = {}
    for iv in intervals:
        by_point.setdefault(iv.usage_point_id, []).append(iv)

    prepared: list[EnergyUsageIntervalV1] = []
    for point, ivs in by_point.items():
        ivs_sorted = sorted(ivs, key=lambda iv: iv.interval_start)
        cycle_start, _ = led.cycle_bounds(ivs_sorted[0].interval_start)
        first = ivs_sorted[0].interval_start
        if first.astimezone(timezone.utc) > cycle_start.astimezone(timezone.utc):
            prepared.append(
                EnergyUsageIntervalV1(
                    source="file_drop",
                    usage_point_id=point,
                    interval_start=cycle_start,
                    interval_end=first,
                    energy_kwh=0.0,
                    retrieved_at=NOW,
                )
            )

        starts = {iv.interval_start.astimezone(timezone.utc) for iv in ivs_sorted}
        for iv in ivs_sorted:
            t_minus_1 = (T - timedelta(hours=1)).astimezone(timezone.utc)
            if iv.interval_start.astimezone(timezone.utc) == T.astimezone(timezone.utc) and t_minus_1 not in starts:
                prepared.append(_iv(T - timedelta(hours=1), 0.0, retrieved=iv.retrieved_at, point=point))
                starts.add(t_minus_1)
            prepared.append(iv)
    return prepared


def _settled() -> PowerIntentSettledV1:
    return PowerIntentSettledV1(
        intent_id="i-1",
        workload_kind="reverie.diffusion",
        node="circe",
        gpu_index=2,
        outcome="settled",
        window_start=T,
        window_end=T + timedelta(hours=1),
        sample_count=3600,
        actual_mean_watts=250.0,
        actual_peak_watts=300.0,
        energy_joules=250.0 * 3600,
        baseline_watts=50.0,
    )


def test_ingest_publishes_usage_then_accrual() -> None:
    p = _pipeline()
    prepared = _prepare(_iv(T - timedelta(hours=2), 100.0))
    p.replay(prepared[:-1])
    out = p.ingest_intervals([prepared[-1]], now=NOW)
    usage = [o for o in out if o.kind == ENERGY_USAGE_KIND]
    accrued = [o for o in out if o.kind == ENERGY_ACCRUED_KIND]
    assert len(usage) == 1 and usage[0].payload.energy_kwh == 100.0
    assert any(a.payload.energy_kwh == 100.0 for a in accrued)


def test_stale_redelivery_publishes_nothing() -> None:
    p = _pipeline()
    prepared = _prepare(_iv(T, 1.0))
    p.replay(prepared[:-1])
    p.ingest_intervals([prepared[-1]], now=NOW)
    assert p.ingest_intervals([_iv(T, 5.0, retrieved=NOW - timedelta(days=1))], now=NOW) == []


def test_settlement_before_house_data_is_repriced_when_it_arrives() -> None:
    p = _pipeline()
    first_batch = _prepare(_iv(T - timedelta(hours=2), 100.0))
    p.replay(first_batch[:-1])
    p.ingest_intervals([first_batch[-1]], now=NOW)
    first = p.on_settlement(_settled(), now=NOW)
    assert first[0].kind == ENERGY_RUN_COST_KIND
    assert first[0].payload.house_share_gap == "house_interval_missing"
    assert p.pending_count() == 1

    later = NOW + timedelta(days=1)
    out = p.ingest_intervals(
        [
            _iv(T - timedelta(hours=1), 0.0, retrieved=later),
            _iv(T, 1.0, retrieved=later),
        ],
        now=later,
    )
    run_costs = [o.payload for o in out if o.kind == ENERGY_RUN_COST_KIND]
    assert len(run_costs) == 1
    assert run_costs[0].house_share_cost_usd == pytest.approx(0.2 * run_costs[0].marginal_usd_per_kwh)
    assert p.pending_count() == 0


def test_blind_settlement_is_not_kept_pending() -> None:
    p = _pipeline()
    blind = _settled().model_copy(
        update={
            "outcome": "no_samples",
            "energy_joules": None,
            "actual_mean_watts": None,
            "actual_peak_watts": None,
        }
    )
    p.on_settlement(blind, now=NOW)
    assert p.pending_count() == 0


def test_pending_expires_after_window() -> None:
    p = _pipeline(pending_hours=24)
    p.on_settlement(_settled(), now=NOW)
    assert p.pending_count() == 1
    much_later = NOW + timedelta(days=3)
    p.ingest_intervals([_iv(T + timedelta(days=2), 1.0, retrieved=much_later)], now=much_later)
    assert p.pending_count() == 0


def test_usage_point_resolution() -> None:
    p = _pipeline()
    assert p.usage_point() is None
    prepared = _prepare(_iv(T, 1.0))
    p.replay(prepared[:-1])
    p.ingest_intervals([prepared[-1]], now=NOW)
    assert p.usage_point() == "UP123"
    p.ingest_intervals([_iv(T, 1.0, point="UP9")], now=NOW)
    assert p.usage_point() is None
    assert _pipeline(usage_point_id="UP9").usage_point() == "UP9"


def test_replay_warms_ledger_without_output() -> None:
    p = _pipeline()
    p.replay(_prepare(_iv(T - timedelta(hours=2), 100.0)))
    est = p.on_settlement(_settled(), now=NOW)[0].payload
    assert est.estimated_run_cost_usd is not None


def test_mid_cycle_only_ingest_logs_incomplete_and_blocks_run_cost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    caplog.set_level(logging.WARNING, logger="orion-energy.pipeline")
    p = _pipeline()
    mid_cycle = datetime(2026, 9, 10, 18, tzinfo=timezone.utc)
    iv = _iv(mid_cycle, 1.0)
    out = p.ingest_intervals([iv], now=NOW)
    assert len([o for o in out if o.kind == ENERGY_USAGE_KIND]) == 1
    assert not [o for o in out if o.kind == ENERGY_ACCRUED_KIND]
    assert any("energy_cycle_incomplete" in r.getMessage() for r in caplog.records)
    est = p.on_settlement(_settled(), now=NOW)[0].payload
    assert est.run_cost_gap == "no_cycle_usage"
