from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class CalibrationProfileStateSQL(Base):
    __tablename__ = "calibration_profiles"

    profile_id = Column(String, primary_key=True)
    profile_version = Column(Integer, nullable=False, default=1)
    source_recommendation_ids = Column(JSONB, nullable=False)
    source_evaluation_request_id = Column(String, nullable=True)
    overrides = Column(JSONB, nullable=False)
    scope = Column(JSONB, nullable=False)
    state = Column(String, index=True, nullable=False)
    previous_profile_id = Column(String, index=True, nullable=True)
    created_by = Column(String, nullable=False)
    rationale = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    activated_at = Column(DateTime(timezone=True), nullable=True)
    rolled_back_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
