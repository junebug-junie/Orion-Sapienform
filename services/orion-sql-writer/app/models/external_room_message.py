from sqlalchemy import Boolean, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class ExternalRoomMessageSQL(Base):
    __tablename__ = "external_room_messages"

    event_id = Column(String, primary_key=True)
    correlation_id = Column(String, index=True, nullable=True)
    platform = Column(String, nullable=False, index=True)
    room_id = Column(String, nullable=False, index=True)
    thread_id = Column(String, nullable=True)
    direction = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    transport_message_id = Column(String, nullable=False, index=True)
    reply_to_message_id = Column(String, nullable=True)
    sender_id = Column(String, nullable=False)
    sender_name = Column(String, nullable=True)
    sender_kind = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    source = Column(String, nullable=False)
    observed_at = Column(String, nullable=False)
    transport_ts = Column(String, nullable=True)
    raw_payload = Column(JSONB, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)
    delivery_ok = Column(Boolean, nullable=True)
    delivery_error = Column(Text, nullable=True)
    skip_reason = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
