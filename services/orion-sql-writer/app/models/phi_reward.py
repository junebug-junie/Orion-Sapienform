from sqlalchemy import Column, DateTime, JSON, String
from sqlalchemy.sql import func

from app.db import Base


class PhiRewardSQL(Base):
    """Durable store for PhiIntrinsicRewardV1 events from orion:self:phi_reward."""

    __tablename__ = "phi_rewards"

    correlation_id = Column(String, primary_key=True)
    generated_at = Column(DateTime(timezone=True), nullable=False, index=True)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
