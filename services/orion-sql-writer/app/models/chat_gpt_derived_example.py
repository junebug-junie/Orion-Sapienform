from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base


class ChatGptDerivedExampleSQL(Base):
    __tablename__ = "chat_gpt_derived_example"

    example_id = Column(String, primary_key=True)
    import_run_id = Column(String, index=True, nullable=False)
    conversation_id = Column(String, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=True)
    user_message_id = Column(String, nullable=True)
    assistant_message_id = Column(String, nullable=True)
    turn_id = Column(String, index=True, nullable=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    prompt_role = Column(String, nullable=False, default="user")
    response_role = Column(String, nullable=False, default="assistant")
    prompt_timestamp = Column(DateTime(timezone=True), nullable=True)
    response_timestamp = Column(DateTime(timezone=True), nullable=True)
    model = Column(String, nullable=True)
    provider = Column(String, nullable=True)
    tags = Column(JSONB, nullable=True)
    meta = Column("metadata", JSONB, nullable=True)
