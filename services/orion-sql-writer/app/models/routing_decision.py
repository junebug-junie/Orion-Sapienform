from uuid import uuid4
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from app.db import Base
from datetime import datetime


class RoutingDecisionSQL(Base):
    """Durable record of one chat routing decision and the confidence gate.

    Consumes RoutingDecisionRecordV1 on orion:routing:decision. This is the
    evidence a routing mutation should be judged on: `chat_reflective_lane_threshold`
    is the one knob Orion can turn about its own behaviour, and until this table
    existed the gate it drives left no trace at all -- its inputs went into an
    in-memory options dict that nothing read.

    Carries no message content. It answers "how did Orion decide", not "what was
    said", so it is safe to query freely for rate comparisons.
    """

    __tablename__ = "routing_decision"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    record_id = Column(String, index=True, nullable=True)
    correlation_id = Column(String, index=True, nullable=True)
    session_id = Column(String, index=True, nullable=True)
    decided_at = Column(DateTime, index=True, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    source = Column(String, nullable=True)
    reason = Column(String, nullable=True)

    # Before and after the gate. They differ exactly when it demoted the turn.
    execution_depth_before_gate = Column(Integer, nullable=True)
    execution_depth = Column(Integer, nullable=True)
    primary_verb = Column(String, nullable=True)

    # Both sides of the comparison, so a later reader can tell a threshold
    # change from a confidence change. Orion only controls the threshold.
    decision_confidence = Column(Float, nullable=True)
    routing_threshold = Column(Float, nullable=True)

    # Indexed: the rate of this over time is the outcome a routing mutation
    # claims to move, and the thing the post-adoption monitor needs to read.
    gate_demoted = Column(Boolean, index=True, nullable=True)
