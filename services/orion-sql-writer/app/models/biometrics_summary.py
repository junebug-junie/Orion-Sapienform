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
