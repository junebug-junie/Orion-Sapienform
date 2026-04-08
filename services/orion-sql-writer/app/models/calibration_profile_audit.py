from sqlalchemy import Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class CalibrationProfileAuditSQL(Base):
    __tablename__ = "calibration_profile_audit"

    audit_id = Column(String, primary_key=True)
    profile_id = Column(String, index=True, nullable=True)
    previous_profile_id = Column(String, index=True, nullable=True)
    event_type = Column(String, index=True, nullable=False)
    operator_id = Column(String, index=True, nullable=False)
    rationale = Column(String, nullable=False)
    details = Column(JSONB, nullable=True)
    payload = Column(JSONB, nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
