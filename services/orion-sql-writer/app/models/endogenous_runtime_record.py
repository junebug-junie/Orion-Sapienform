from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class EndogenousRuntimeRecordSQL(Base):
    __tablename__ = "endogenous_runtime_records"

    runtime_record_id = Column(String, primary_key=True)
    correlation_id = Column(String, index=True, nullable=True)
    request_id = Column(String, index=True, nullable=True)
    invocation_surface = Column(String, index=True, nullable=False)
    trigger_outcome = Column(String, index=True, nullable=False)
    workflow_type = Column(String, index=True, nullable=False)
    subject_ref = Column(String, index=True, nullable=True)
    anchor_scope = Column(String, nullable=True)
    mentor_invoked = Column(Boolean, nullable=False, default=False)
    execution_success = Column(Boolean, nullable=False, default=True)
    calibration_profile_id = Column(String, index=True, nullable=True)
    materialized_artifact_ids = Column(JSONB, nullable=True)
    payload = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
