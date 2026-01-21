from sqlalchemy import Column, Float, String
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base
from uuid import uuid4


class BiometricsInductionSQL(Base):
    __tablename__ = "orion_biometrics_induction"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    timestamp = Column(String, nullable=False)
    node = Column(String, nullable=True)
    service_name = Column(String, nullable=True)
    service_version = Column(String, nullable=True)
    window_sec = Column(Float, nullable=True)
    metrics = Column(JSONB, nullable=True)
