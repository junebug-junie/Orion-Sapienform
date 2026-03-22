from sqlalchemy import Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class ExternalRoomParticipantSQL(Base):
    __tablename__ = "external_room_participants"

    participant_ref = Column(String, primary_key=True)
    platform = Column(String, nullable=False, index=True)
    room_id = Column(String, nullable=False, index=True)
    participant_id = Column(String, nullable=False)
    participant_name = Column(String, nullable=True)
    participant_kind = Column(String, nullable=False)
    last_message_id = Column(String, nullable=True)
    last_seen_at = Column(String, nullable=False)
    metadata_ = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
