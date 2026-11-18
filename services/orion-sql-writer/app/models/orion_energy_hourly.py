from sqlalchemy import Column, DateTime, Float, Index, Integer, String
from sqlalchemy.sql import func

from app.db import Base


class OrionEnergyHourlySQL(Base):
    """Orion's own electricity, one row per (UTC hour, source). Single writer:
    app/orion_energy_rollup.py, which integrates the raw watts already in Postgres.

    ``source`` is a node name from ``orion_biometrics_cluster.measurements_by_node``
    (athena, circe, hecate, ...; each node's ``chassis_watts``, which is BMC or PDU wall
    draw) or ``cooling`` (the cabinet AC's Shelly plug, ``home_cooling_sample``).

    Why a table and not a view: raw cluster watts are pruned after
    ORION_BIOMETRICS_CLUSTER_RETENTION_DAYS (30), and the Hub tab needs months and years.
    This table has no retention (~5 rows/hour).

    ``kwh`` is what was measured, not extrapolated: each sample's watts are held until the
    next sample, capped so a gap is not filled in. ``covered_sec`` says how much of the hour
    that was. An hour with no samples has no row: unknown, never 0 kWh.
    """

    __tablename__ = "orion_energy_hourly"

    hour_start = Column(DateTime(timezone=True), primary_key=True)
    source = Column(String, primary_key=True)
    kwh = Column(Float, nullable=False)
    covered_sec = Column(Float, nullable=False)
    sample_count = Column(Integer, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (Index("idx_orion_energy_hourly_hour", "hour_start"),)
