from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class EndogenousRuntimeAuditSQL(Base):
    __tablename__ = "endogenous_runtime_audit"

    id = Column(Integer, primary_key=True, autoincrement=True)
    runtime_record_id = Column(String, index=True, nullable=True)
    invocation_surface = Column(String, index=True, nullable=True)
    status = Column(String, index=True, nullable=False)
    workflow_type = Column(String, index=True, nullable=True)
    decision_outcome = Column(String, index=True, nullable=True)
    mentor_invoked = Column(Boolean, nullable=False, default=False)
    cooldown_applied = Column(Boolean, nullable=False, default=False)
    debounce_applied = Column(Boolean, nullable=False, default=False)
    error = Column(String, nullable=True)
    calibration_profile_id = Column(String, index=True, nullable=True)
    payload = Column(JSONB, nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
