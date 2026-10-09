"""Energy kinds -> tables, with newest-wins upserts on natural keys."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.dialects import postgresql

from app.energy_persist import ENERGY_UPSERTS
from app.models.energy import (
    EnergyBillActualSQL,
    EnergyBillForecastSQL,
    EnergyCostAccruedSQL,
    EnergyImporterStatusSQL,
    EnergyReconcileSQL,
    EnergyRunCostSQL,
    EnergyStakesSnapshotSQL,
    EnergyUsageIntervalSQL,
)
from app.settings import Settings
from app.worker import MODEL_MAP

from orion.schemas.energy import (
    EnergyBillActualV1,
    EnergyBillForecastV1,
    EnergyCostAccruedV1,
    EnergyImporterStatusV1,
    EnergyReconcileV1,
    EnergyRunCostEstimatedV1,
    EnergyStakesSnapshotV1,
    EnergyUsageIntervalV1,
)

T0 = datetime(2026, 9, 10, 18, tzinfo=timezone.utc)
T1 = datetime(2026, 9, 10, 19, tzinfo=timezone.utc)

ROUTES = [
    ("energy.usage.observed.v1", "EnergyUsageIntervalSQL", EnergyUsageIntervalSQL, EnergyUsageIntervalV1, "orion:energy:usage:observed"),
    ("energy.cost.accrued.v1", "EnergyCostAccruedSQL", EnergyCostAccruedSQL, EnergyCostAccruedV1, "orion:energy:cost:accrued"),
    ("energy.run_cost.estimated.v1", "EnergyRunCostSQL", EnergyRunCostSQL, EnergyRunCostEstimatedV1, "orion:energy:run_cost:estimated"),
    ("energy.bill.actual.v1", "EnergyBillActualSQL", EnergyBillActualSQL, EnergyBillActualV1, "orion:energy:bill:actual"),
    ("energy.bill.forecast.v1", "EnergyBillForecastSQL", EnergyBillForecastSQL, EnergyBillForecastV1, "orion:energy:bill:forecast"),
    ("energy.reconcile.v1", "EnergyReconcileSQL", EnergyReconcileSQL, EnergyReconcileV1, "orion:energy:reconcile"),
    ("energy.stakes.snapshot.v1", "EnergyStakesSnapshotSQL", EnergyStakesSnapshotSQL, EnergyStakesSnapshotV1, "orion:energy:stakes:snapshot"),
    ("energy.importer.status.v1", "EnergyImporterStatusSQL", EnergyImporterStatusSQL, EnergyImporterStatusV1, "orion:energy:importer:status"),
]


@pytest.mark.parametrize("kind,name,model,schema,channel", ROUTES)
def test_every_schema_field_has_a_column(kind, name, model, schema, channel) -> None:
    cols = {c.key for c in sa_inspect(model).columns}
    missing = [f for f in schema.model_fields if f not in cols]
    assert not missing, f"{model.__tablename__} missing {missing}"


@pytest.mark.parametrize("kind,name,model,schema,channel", ROUTES)
def test_kind_routes_to_table(kind, name, model, schema, channel) -> None:
    model_cls, schema_cls = MODEL_MAP[name]
    assert model_cls is model and schema_cls is schema
    assert Settings().route_map.get(kind) == name


@pytest.mark.parametrize("kind,name,model,schema,channel", ROUTES)
def test_channel_subscribed_even_with_stale_env(kind, name, model, schema, channel) -> None:
    example = Path(__file__).resolve().parents[1] / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert channel in json.loads(raw)
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert channel in stale.effective_subscribe_channels


class _CapturingSession:
    def __init__(self) -> None:
        self.statements = []
        self.committed = False

    def execute(self, stmt):
        self.statements.append(stmt)

    def commit(self) -> None:
        self.committed = True


def _sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_usage_upsert_newer_retrieval_wins() -> None:
    sess = _CapturingSession()
    row = EnergyUsageIntervalV1(
        source="file_drop", usage_point_id="UP123", interval_start=T0, interval_end=T1,
        energy_kwh=1.2, retrieved_at=T1,
    ).model_dump()
    assert ENERGY_UPSERTS[EnergyUsageIntervalSQL](sess, row) is True
    sql = _sql(sess.statements[0])
    assert "ON CONFLICT ON CONSTRAINT uq_energy_usage_interval_point_start DO UPDATE" in sql
    assert "energy_usage_interval.retrieved_at <= excluded.retrieved_at" in sql
    assert sess.committed


def test_accrued_upsert_newest_computation_wins_and_coerces_date() -> None:
    sess = _CapturingSession()
    row = EnergyCostAccruedV1(
        usage_point_id="UP123", interval_start=T0, interval_end=T1, energy_kwh=1.0,
        interval_cost_usd=0.11, marginal_usd_per_kwh=0.11, cycle_start=date(2026, 9, 1),
        cycle_accumulated_kwh=101.0, cycle_energy_cost_usd=11.2, cycle_to_date_total_usd=23.36,
        tariff_version="rmp-ut-sch1-2026-08-10", computed_at=T1,
    ).model_dump(mode="json")
    ENERGY_UPSERTS[EnergyCostAccruedSQL](sess, row)
    stmt = sess.statements[0]
    sql = _sql(stmt)
    assert "uq_energy_cost_accrued_point_start_tariff" in sql
    assert "energy_cost_accrued.computed_at <= excluded.computed_at" in sql
    assert stmt.compile(dialect=postgresql.dialect()).params["cycle_start"] == date(2026, 9, 1)


def test_run_cost_upsert_keyed_on_intent() -> None:
    sess = _CapturingSession()
    row = EnergyRunCostEstimatedV1(
        intent_id="i-1", workload_kind="reverie.diffusion", node="circe", window_start=T0,
        window_end=T1, settlement_outcome="settled", energy_kwh=0.2,
        energy_basis="incremental_over_baseline", estimated_run_cost_usd=0.022,
        house_share_gap="house_interval_missing", computed_at=T1,
    ).model_dump()
    ENERGY_UPSERTS[EnergyRunCostSQL](sess, row)
    sql = _sql(sess.statements[0])
    assert "ON CONFLICT (intent_id) DO UPDATE" in sql
    assert "energy_run_cost.computed_at <= excluded.computed_at" in sql


def test_bill_actual_upsert_by_period_and_dates_stay_dates() -> None:
    sess = _CapturingSession()
    row = EnergyBillActualV1(
        source="file_drop", billing_period_start=date(2026, 8, 12), billing_period_end=date(2026, 9, 11),
        kwh_billed=712.0, current_charges=101.23, due_date=date(2026, 10, 2), retrieved_at=T1,
    ).model_dump(mode="json")
    ENERGY_UPSERTS[EnergyBillActualSQL](sess, row)
    stmt = sess.statements[0]
    sql = _sql(stmt)
    assert "uq_energy_bill_actual_period" in sql
    assert "energy_bill_actual.retrieved_at <= excluded.retrieved_at" in sql
    params = stmt.compile(dialect=postgresql.dialect()).params
    # Regression: the old suffix rule parsed any *_start/*_end string as a datetime.
    assert params["billing_period_start"] == date(2026, 8, 12)
    assert params["billing_period_end"] == date(2026, 9, 11)
    assert params["due_date"] == date(2026, 10, 2)
    assert params["retrieved_at"] == T1


def test_forecast_upsert_by_period_and_as_of() -> None:
    sess = _CapturingSession()
    row = EnergyBillForecastV1(
        source="rockymountain_power", billing_period_start=date(2026, 9, 11), as_of=T0,
        projected_total_usd=96.0, retrieved_at=T1,
    ).model_dump(mode="json")
    ENERGY_UPSERTS[EnergyBillForecastSQL](sess, row)
    sql = _sql(sess.statements[0])
    assert "uq_energy_bill_forecast_period_as_of" in sql
    assert "energy_bill_forecast.retrieved_at <= excluded.retrieved_at" in sql


def test_reconcile_upsert_keeps_bucket_deltas() -> None:
    sess = _CapturingSession()
    row = EnergyReconcileV1(
        reconcile_kind="actual", usage_point_id="UP1", billing_period_start=date(2026, 8, 12),
        billing_period_end=date(2026, 9, 11), utility_as_of=T0, utility_kwh=712.0,
        utility_total_usd=97.13, utility_basis="pre_tax", orion_method="metered_period",
        orion_kwh=710.0, orion_energy_usd=85.0, orion_fixed_usd=12.16, orion_total_usd=97.16,
        delta_kwh=-2.0, delta_usd=0.03, delta_pct=0.0003, bucket_deltas={"customer_charge": 0.16},
        tariff_version="t1", computed_at=T1,
    ).model_dump(mode="json")
    ENERGY_UPSERTS[EnergyReconcileSQL](sess, row)
    stmt = sess.statements[0]
    sql = _sql(stmt)
    assert "uq_energy_reconcile_key" in sql
    assert "energy_reconcile.computed_at <= excluded.computed_at" in sql
    assert stmt.compile(dialect=postgresql.dialect()).params["bucket_deltas"] == {"customer_charge": 0.16}


@pytest.mark.parametrize(
    "model,payload,constraint",
    [
        (
            EnergyStakesSnapshotSQL,
            EnergyStakesSnapshotV1(as_of=T1, importer_state="stale", pressure="unknown", pressure_reason="importer_stale"),
            "uq_energy_stakes_snapshot_as_of",
        ),
        (
            EnergyImporterStatusSQL,
            EnergyImporterStatusV1(state="stale", reason="no_usage_yet", source="file_drop", as_of=T1),
            "uq_energy_importer_status_as_of",
        ),
    ],
)
def test_snapshots_insert_once(model, payload, constraint) -> None:
    sess = _CapturingSession()
    ENERGY_UPSERTS[model](sess, payload.model_dump(mode="json"))
    sql = _sql(sess.statements[0])
    assert f"ON CONFLICT ON CONSTRAINT {constraint} DO NOTHING" in sql
    assert sess.committed
