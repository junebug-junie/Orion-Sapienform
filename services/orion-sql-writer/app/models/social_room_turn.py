from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class SocialRoomTurnSQL(Base):
    __tablename__ = "social_room_turns"

    turn_id = Column(String, primary_key=True)
    correlation_id = Column(String, index=True, nullable=True)
    session_id = Column(String, index=True, nullable=True)
    user_id = Column(String, nullable=True)
    source = Column(String, nullable=False)
    profile = Column(String, nullable=False, default="social_room")
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    text = Column(Text, nullable=False)
    recall_profile = Column(String, nullable=True)
    trace_verb = Column(String, nullable=True)
    tags = Column(JSONB, nullable=True)
    concept_evidence = Column(JSONB, nullable=True)
    grounding_state = Column(JSONB, nullable=True)
    redaction = Column(JSONB, nullable=True)
    client_meta = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
