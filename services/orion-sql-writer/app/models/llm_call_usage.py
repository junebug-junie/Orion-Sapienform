"""One row per LLM call: the per-call token ledger (2026-10-29).

Writer: sql-writer off ``orion:llm:call:usage`` (``LlmCallUsageV1``, contract and join rules in
``orion/schemas/llm_call_usage.py``). Producer: orion-llm-gateway (``recorder='gateway'``, every
call it places, harness turns included via the FCC adapter's header forward). ``harness_relay`` is
retired. PK ``call_id``, so a redelivered event is one row.

Totals over time: ``WHERE recorder = 'gateway'``. One turn or act: ``WHERE correlation_id = ...``.

Table and indexes: ``services/orion-sql-db/manual_migration_llm_call_usage_v1.sql``.
"""

from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.sql import func

from app.db import Base


class LlmCallUsageSQL(Base):
    __tablename__ = "llm_call_usage"

    call_id = Column(String, primary_key=True)
    recorded_at = Column(DateTime(timezone=True), nullable=False)
    recorder = Column(String, nullable=False)
    path = Column(String, nullable=False)
    correlation_id = Column(Text, nullable=True)
    route = Column(String, nullable=True)
    model = Column(Text, nullable=True)
    served_by = Column(String, nullable=True)
    pool_role = Column(String, nullable=True)
    holder = Column(String, nullable=True)
    verb = Column(String, nullable=True)
    node = Column(String, nullable=True)
    outcome = Column(String, nullable=False)
    stream = Column(Boolean, nullable=False)
    prompt_tokens = Column(Integer, nullable=True)
    cached_prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    token_source = Column(String, nullable=False)
    duration_ms = Column(Integer, nullable=True)
    wait_ms = Column(Integer, nullable=True)
    decode_tps = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_llm_call_usage_recorded", "recorded_at"),
        Index("idx_llm_call_usage_corr_recorded", "correlation_id", "recorded_at"),
        Index("idx_llm_call_usage_route_recorded", "route", "recorded_at"),
    )
