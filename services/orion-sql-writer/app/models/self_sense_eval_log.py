from sqlalchemy import Column, DateTime, Index, Integer, String, Text
from sqlalchemy.sql import func

from app.db import Base


class SelfSenseEvalLogSQL(Base):
    """One persisted answer+scores per fixed question per self-sense eval run.
    Producer schema: orion/schemas/self_sense.py `SelfSenseEvalV1`."""

    __tablename__ = "self_sense_eval_log"

    entry_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    question_key = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    answer_source = Column(String, nullable=False)
    correlation_id = Column(String, nullable=True)
    self_label_score = Column(Integer, nullable=False)
    grounded_record_score = Column(Integer, nullable=False)
    self_definition_version = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_self_sense_eval_log_created_at", "created_at"),
        Index("idx_self_sense_eval_log_run_id", "run_id"),
    )
