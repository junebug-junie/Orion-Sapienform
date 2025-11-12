from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base
from uuid import uuid4

class BiometricsTelemetry(Base):
    __tablename__ = "orion_biometrics"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    timestamp = Column(String, nullable=False)
    node = Column(String, nullable=True)
    service_name = Column(String, nullable=True)
    service_version = Column(String, nullable=True)
    gpu = Column(JSONB, nullable=True)
    cpu = Column(JSONB, nullable=True)
