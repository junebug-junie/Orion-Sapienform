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
