from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base


class EnergyUsageIntervalSQL(Base):
    """Whole-house metered kWh per interval (``orion:energy:usage:observed``).

    One row per (usage point, interval start). Late utility corrections upsert in
    place; the newer ``retrieved_at`` wins (see ``app/energy_persist.py``).
    """

    __tablename__ = "energy_usage_interval"
    __table_args__ = (
        UniqueConstraint("usage_point_id", "interval_start", name="uq_energy_usage_interval_point_start"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source = Column(String, nullable=False)
    usage_point_id = Column(String, nullable=False)
    interval_start = Column(DateTime(timezone=True), nullable=False, index=True)
    interval_end = Column(DateTime(timezone=True), nullable=False)
    energy_kwh = Column(Float, nullable=False)
    quality = Column(String, nullable=True)
    retrieved_at = Column(DateTime(timezone=True), nullable=False)
    source_file = Column(String, nullable=True)


class EnergyCostAccruedSQL(Base):
    """Tariff-priced usage interval (``orion:energy:cost:accrued``), one row per tariff version."""

    __tablename__ = "energy_cost_accrued"
    __table_args__ = (
        UniqueConstraint(
            "usage_point_id", "interval_start", "tariff_version",
            name="uq_energy_cost_accrued_point_start_tariff",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    usage_point_id = Column(String, nullable=False)
    interval_start = Column(DateTime(timezone=True), nullable=False, index=True)
    interval_end = Column(DateTime(timezone=True), nullable=False)
    energy_kwh = Column(Float, nullable=False)
    interval_cost_usd = Column(Float, nullable=False)
    marginal_usd_per_kwh = Column(Float, nullable=False)
    cycle_start = Column(Date, nullable=False, index=True)
    cycle_accumulated_kwh = Column(Float, nullable=False)
    cycle_energy_cost_usd = Column(Float, nullable=False)
    cycle_to_date_total_usd = Column(Float, nullable=False)
    tariff_version = Column(String, nullable=False)
    cost_basis = Column(String, nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False)


class EnergyRunCostSQL(Base):
    """Dollar cost of a settled power intent (``orion:energy:run_cost:estimated``).

    NULL cost columns mean UNKNOWN and always come with a ``*_gap`` reason; they
    must never be read as free.
    """

    __tablename__ = "energy_run_cost"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    intent_id = Column(String, nullable=False, unique=True)
    workload_kind = Column(String, nullable=False)
    node = Column(String, nullable=False)
    gpu_index = Column(Integer, nullable=True)
    window_start = Column(DateTime(timezone=True), nullable=False, index=True)
    window_end = Column(DateTime(timezone=True), nullable=False)
    settlement_outcome = Column(String, nullable=False)
    energy_kwh = Column(Float, nullable=True)
    energy_basis = Column(String, nullable=True)
    estimated_run_cost_usd = Column(Float, nullable=True)
    marginal_usd_per_kwh = Column(Float, nullable=True)
    cycle_kwh_basis = Column(Float, nullable=True)
    cycle_kwh_basis_as_of = Column(DateTime(timezone=True), nullable=True)
    run_cost_gap = Column(String, nullable=True)
    house_share_cost_usd = Column(Float, nullable=True)
    house_kwh_overlap = Column(Float, nullable=True)
    house_share_gap = Column(String, nullable=True)
    tariff_version = Column(String, nullable=True)
    cost_basis = Column(String, nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False)


class EnergyBillActualSQL(Base):
    """Closed RMP billing period (``orion:energy:bill:actual``). NULL money = not on the bill."""

    __tablename__ = "energy_bill_actual"
    __table_args__ = (
        UniqueConstraint("billing_period_start", "billing_period_end", name="uq_energy_bill_actual_period"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source = Column(String, nullable=False)
    usage_point_id = Column(String, nullable=True)
    billing_period_start = Column(Date, nullable=False, index=True)
    billing_period_end = Column(Date, nullable=False)
    kwh_billed = Column(Float, nullable=False)
    energy_charge = Column(Float, nullable=True)
    customer_charge = Column(Float, nullable=True)
    adjustments = Column(Float, nullable=True)
    fees = Column(Float, nullable=True)
    taxes = Column(Float, nullable=True)
    credits = Column(Float, nullable=True)
    current_charges = Column(Float, nullable=False)
    amount_due = Column(Float, nullable=True)
    due_date = Column(Date, nullable=True)
    statement_artifact_id = Column(String, nullable=True)
    retrieved_at = Column(DateTime(timezone=True), nullable=False)
    source_file = Column(String, nullable=True)


class EnergyBillForecastSQL(Base):
    """RMP in-cycle projection (``orion:energy:bill:forecast``), one row per (period, as_of)."""

    __tablename__ = "energy_bill_forecast"
    __table_args__ = (
        UniqueConstraint("billing_period_start", "as_of", name="uq_energy_bill_forecast_period_as_of"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source = Column(String, nullable=False)
    usage_point_id = Column(String, nullable=True)
    billing_period_start = Column(Date, nullable=False, index=True)
    billing_period_end = Column(Date, nullable=True)
    as_of = Column(DateTime(timezone=True), nullable=False)
    days_into_cycle = Column(Integer, nullable=True)
    projected_kwh = Column(Float, nullable=True)
    projected_total_usd = Column(Float, nullable=True)
    retrieved_at = Column(DateTime(timezone=True), nullable=False)
    source_file = Column(String, nullable=True)


class EnergyReconcileSQL(Base):
    """Orion vs RMP (``orion:energy:reconcile``). NULL Orion total always has a ``reconcile_gap``."""

    __tablename__ = "energy_reconcile"
    __table_args__ = (
        UniqueConstraint(
            "reconcile_kind", "billing_period_start", "utility_as_of", "tariff_version",
            name="uq_energy_reconcile_key",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    reconcile_kind = Column(String, nullable=False)
    usage_point_id = Column(String, nullable=False)
    billing_period_start = Column(Date, nullable=False, index=True)
    billing_period_end = Column(Date, nullable=True)
    utility_as_of = Column(DateTime(timezone=True), nullable=False)
    utility_kwh = Column(Float, nullable=True)
    utility_total_usd = Column(Float, nullable=True)
    utility_basis = Column(String, nullable=False)
    orion_method = Column(String, nullable=False)
    orion_covered_through = Column(DateTime(timezone=True), nullable=True)
    orion_kwh = Column(Float, nullable=True)
    orion_energy_usd = Column(Float, nullable=True)
    orion_fixed_usd = Column(Float, nullable=True)
    orion_total_usd = Column(Float, nullable=True)
    reconcile_gap = Column(String, nullable=True)
    delta_kwh = Column(Float, nullable=True)
    delta_usd = Column(Float, nullable=True)
    delta_pct = Column(Float, nullable=True)
    bucket_deltas = Column(JSONB, nullable=False, default=dict)
    tariff_version = Column(String, nullable=False)
    cost_basis = Column(String, nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False)


class EnergyStakesSnapshotSQL(Base):
    """Stakes snapshot (``orion:energy:stakes:snapshot``); Hub + curiosity read the latest row."""

    __tablename__ = "energy_stakes_snapshot"
    __table_args__ = (UniqueConstraint("as_of", name="uq_energy_stakes_snapshot_as_of"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    as_of = Column(DateTime(timezone=True), nullable=False, index=True)
    usage_point_id = Column(String, nullable=True)
    cycle_start = Column(Date, nullable=True)
    cycle_end = Column(Date, nullable=True)
    covered_through = Column(DateTime(timezone=True), nullable=True)
    cycle_accumulated_kwh = Column(Float, nullable=True)
    cycle_to_date_total_usd = Column(Float, nullable=True)
    marginal_usd_per_kwh = Column(Float, nullable=True)
    orion_projected_total_usd = Column(Float, nullable=True)
    forecast_total_usd = Column(Float, nullable=True)
    forecast_as_of = Column(DateTime(timezone=True), nullable=True)
    projected_to_forecast_ratio = Column(Float, nullable=True)
    importer_state = Column(String, nullable=False)
    pressure = Column(String, nullable=False)
    pressure_reason = Column(String, nullable=False)
    tariff_version = Column(String, nullable=True)


class EnergyImporterStatusSQL(Base):
    """Importer health (``orion:energy:importer:status``)."""

    __tablename__ = "energy_importer_status"
    __table_args__ = (UniqueConstraint("as_of", name="uq_energy_importer_status_as_of"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    state = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    source = Column(String, nullable=False)
    last_success_at = Column(DateTime(timezone=True), nullable=True)
    last_attempt_at = Column(DateTime(timezone=True), nullable=True)
    latest_interval_end = Column(DateTime(timezone=True), nullable=True)
    usage_lag_hours = Column(Float, nullable=True)
    as_of = Column(DateTime(timezone=True), nullable=False, index=True)
