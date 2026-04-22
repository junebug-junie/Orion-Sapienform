from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base


class ChatGptConversationSQL(Base):
    __tablename__ = "chat_gpt_conversation"

    conversation_id = Column(String, primary_key=True)
    import_run_id = Column(String, primary_key=True)
    session_id = Column(String, index=True, nullable=True)
    title = Column(String, nullable=True)
    create_time = Column(Float, nullable=True)
    update_time = Column(Float, nullable=True)
    current_node_id = Column(String, nullable=True)
    message_count = Column(Integer, nullable=False, default=0)
    turn_count = Column(Integer, nullable=False, default=0)
    branch_count = Column(Integer, nullable=False, default=0)
    meta = Column("metadata", JSONB, nullable=True)
    ingested_at = Column(DateTime(timezone=True), nullable=True)
