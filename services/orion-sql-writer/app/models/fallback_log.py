from sqlalchemy import Column, String, Integer, Text, JSON, DateTime
from sqlalchemy.sql import func
from app.db import Base

class BusFallbackLog(Base):
    __tablename__ = "bus_fallback_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    kind = Column(String, index=True)
    correlation_id = Column(String, index=True, nullable=True)
    payload = Column(JSON)  # Store the raw payload here
    created_at = Column(String, default=func.now())
    created_at_ts = Column(DateTime(timezone=True), index=True, nullable=True)
    error = Column(Text, nullable=True)
