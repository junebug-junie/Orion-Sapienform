"""One row per shadow bid decision (body economy Patch C1, Organ 4 Choice).

Writer: sql-writer off ``orion:physiology:want:decision`` (``WantDecisionV1``; producer:
orion-durable-runs ``POST /physiology/bid``). Each row is a want and what Choice decided about it:
funded or deferred, the lane, the prices it saw, the net, the binding constraint and what it
displaced. ``shadow`` is true for every Patch C row: nobody obeyed these decisions. PK
``decision_id`` (``<want_id>@<decided_at>``), so a redelivered event is one row.

Table and indexes: ``services/orion-sql-db/manual_migration_want_ledger_v1.sql`` (jsonb there;
generic JSON here so the sqlite write-path test runs). ``decision`` keeps the full decision verbatim
(the want included); the flat columns are for the shadow-vs-real disagreement queries.
"""

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.sql import func

from app.db import Base


class WantLedgerSQL(Base):
    __tablename__ = "want_ledger"

    decision_id = Column(String, primary_key=True)
    want_id = Column(String, nullable=False)
    family = Column(String, nullable=False)
    producer = Column(String, nullable=False)
    verb = Column(String, nullable=False)
    tier = Column(Integer, nullable=False)
    decided_at = Column(DateTime(timezone=True), nullable=False)
    outcome = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    lane = Column(String, nullable=True)
    net = Column(Float, nullable=True)
    binding_constraint = Column(String, nullable=True)
    shadow = Column(Boolean, nullable=False)
    correlation_id = Column(Text, nullable=True)
    decision = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_want_ledger_decided", "decided_at"),
        Index("idx_want_ledger_family_decided", "family", "decided_at"),
        Index("idx_want_ledger_want", "want_id"),
    )


def want_ledger_row(decision) -> dict:
    """``WantDecisionV1`` -> row. The whole decision is kept as JSON-safe data."""
    as_json = decision.model_dump(mode="json")
    return dict(decision_id=decision.decision_id, want_id=decision.want_id, family=decision.family,
                producer=decision.producer, verb=decision.verb, tier=decision.tier, decided_at=decision.decided_at,
                outcome=decision.outcome, reason=decision.reason, lane=decision.lane, net=decision.net,
                binding_constraint=decision.binding_constraint, shadow=decision.shadow,
                correlation_id=decision.want.correlation_id, decision=as_json)
