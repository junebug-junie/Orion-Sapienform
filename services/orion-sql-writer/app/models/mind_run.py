from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class MindRunSQL(Base):
    __tablename__ = "mind_runs"

    mind_run_id = Column(String, primary_key=True)
    correlation_id = Column(String, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=True)
    trigger = Column(String, nullable=False)
    ok = Column(Boolean, nullable=False)
    error_code = Column(String, nullable=True)
    snapshot_hash = Column(String, nullable=False, default="")
    router_profile_id = Column(String, nullable=False, default="")
    result_jsonb = Column(JSONB, nullable=False)
    request_summary_jsonb = Column(JSONB, nullable=False)
    redaction_profile_id = Column(String, nullable=True)
    created_at_utc = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
