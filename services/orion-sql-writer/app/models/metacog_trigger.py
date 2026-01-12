from uuid import uuid4
from sqlalchemy import Column, String, Float, Integer, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base
from datetime import datetime

class MetacogTriggerSQL(Base):
    __tablename__ = "metacog_trigger"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    correlation_id = Column(String, index=True, nullable=True)

    trigger_kind = Column(String, nullable=False)  # baseline, dense, manual, pulse
    reason = Column(Text, nullable=True)
    zen_state = Column(String, nullable=True)      # zen, not_zen, unknown
    pressure = Column(Float, default=0.0)
    window_sec = Column(Integer, default=15)

    frame_refs = Column(JSONB, default=list)       # List[str]
    signal_refs = Column(JSONB, default=list)      # List[str]
    upstream = Column(JSONB, default=dict)         # Upstream trigger summary

    timestamp = Column(DateTime, default=datetime.utcnow)
