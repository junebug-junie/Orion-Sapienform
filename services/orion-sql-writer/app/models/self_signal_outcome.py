from sqlalchemy import JSON, Column, DateTime, Float, Index, Integer, String
from sqlalchemy.sql import func

from app.db import Base


class SelfSignalHourSQL(Base):
    """One row per completed UTC hour (SelfSignalHourV1, orion/schemas/self_signal_outcome.py):
    each internal body signal's condition that hour (high / normal / unknown) and that hour's
    outcome counts. Single writer: sql-writer off ``orion:equilibrium:self_signal:hour``
    (orion-equilibrium-service app/self_signal_outcome_rollup.py). Upserted on hour_id: a retried
    publish rewrites the same hour. Outlives substrate_node_prediction_error_history's 7-day prune
    so the association can accumulate. 24 rows/day; no retention policy (tiny)."""

    __tablename__ = "self_signal_hour"

    hour_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    hour_start = Column(DateTime(timezone=True), nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False)
    schema_version = Column(String, nullable=False, default="self_signal.hour.v1")
    signals = Column(JSON, nullable=False)
    outcomes = Column(JSON, nullable=False)
    history_hours = Column(Float, nullable=False, default=0.0)

    __table_args__ = (Index("idx_self_signal_hour_hour", "hour_start"),)


class SelfSignalOutcomeAssociationSQL(Base):
    """One row per (signal, outcome) per hourly refresh (SelfSignalOutcomeAssociationV1): next-hour
    outcome rate when the signal was high vs normal, rate ratio and 95% interval, verdict, and the
    one clause a prompt may show. Append-only; readers take the newest row per (signal_id,
    outcome) and cite association_id. ~21 rows/hour; bounded retention
    SELF_SIGNAL_OUTCOME_ASSOCIATION_RETENTION_DAYS (30, grammar_retention_loop)."""

    __tablename__ = "self_signal_outcome_association"

    association_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    computed_at = Column(DateTime(timezone=True), nullable=False)
    schema_version = Column(String, nullable=False, default="self_signal.outcome_association.v1")
    signal_id = Column(String, nullable=False)
    outcome = Column(String, nullable=False)
    window_start = Column(DateTime(timezone=True), nullable=True)
    window_end = Column(DateTime(timezone=True), nullable=True)
    n_high = Column(Integer, nullable=False, default=0)
    events_high = Column(Integer, nullable=False, default=0)
    n_normal = Column(Integer, nullable=False, default=0)
    events_normal = Column(Integer, nullable=False, default=0)
    rate_high = Column(Float, nullable=True)
    rate_normal = Column(Float, nullable=True)
    ratio = Column(Float, nullable=True)
    ci_low = Column(Float, nullable=True)
    ci_high = Column(Float, nullable=True)
    ci_method = Column(String, nullable=False)
    ci_z = Column(Float, nullable=False)
    se_inflation = Column(Float, nullable=False)
    n_comparisons = Column(Integer, nullable=False)
    n_strata = Column(Integer, nullable=False, default=0)
    min_hours = Column(Integer, nullable=False, default=0)
    min_history_hours = Column(Float, nullable=False, default=0.0)
    verdict = Column(String, nullable=False)
    excluded_reason = Column(String, nullable=False, default="")
    clause = Column(String, nullable=False, default="")

    __table_args__ = (
        Index("idx_self_signal_assoc_pair_time", "signal_id", "outcome", "computed_at"),
    )
