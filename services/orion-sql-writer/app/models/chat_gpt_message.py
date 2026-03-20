from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base


class ChatGptMessageSQL(Base):
    __tablename__ = "chat_gpt_message"

    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    correlation_id = Column(String, index=True, nullable=True)
    trace_id = Column(String, index=True, nullable=True)
    memory_status = Column(String, index=True, nullable=True)
    memory_tier = Column(String, index=True, nullable=True)
    user_id = Column(String, nullable=True)
    role = Column(String)
    content = Column(Text)
    model = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True))
    meta = Column("metadata", JSONB, nullable=True)
