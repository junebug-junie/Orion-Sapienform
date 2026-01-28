from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base

class ChatMessageSQL(Base):
    __tablename__ = "chat_message"

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
    # We use 'meta' as the attribute name to avoid conflict with Base.metadata
    # The worker may need adjustment to map 'metadata' from payload to 'meta' here, 
    # or rely on the underlying column name if using core insert, but the worker uses ORM kwargs.
    meta = Column("metadata", JSONB, nullable=True)
