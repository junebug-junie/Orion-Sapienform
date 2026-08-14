from sqlalchemy import Column, Float, String
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base
from uuid import uuid4


class BiometricsSummarySQL(Base):
    __tablename__ = "orion_biometrics_summary"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    timestamp = Column(String, nullable=False)
    node = Column(String, nullable=True)
    service_name = Column(String, nullable=True)
    service_version = Column(String, nullable=True)
    pressures = Column(JSONB, nullable=True)
    headroom = Column(JSONB, nullable=True)
    composites = Column(JSONB, nullable=True)
    constraint = Column(String, nullable=True)
    telemetry_error_rate = Column(Float, nullable=True)
    # Raw physical units (chassis_watts, temp_c_max, ...) alongside the normalised pressures.
    # A key is absent when unmeasured on that node -- never 0.0 -- so a fleet total summed
    # from this column is honest about what it could not see. See BiometricsSummaryV1.
    # NB: `Base.metadata.create_all` creates missing TABLES, not missing COLUMNS, so an
    # existing deployment needs the ALTER in scripts/sql/2026-08-14_biometrics_measurements.sql.
    measurements = Column(JSONB, nullable=True)
