from sqlalchemy import Column, Integer, Date, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.db import Base

class Dream(Base):
    __tablename__ = "dreams"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dream_date = Column(Date, nullable=False, index=True, unique=True)

    tldr = Column(Text, nullable=True)
    themes = Column(JSONB, nullable=True)   # List[str]
    symbols = Column(JSONB, nullable=True)  # Dict[str, str]
    narrative = Column(Text, nullable=True)

    fragments = Column(JSONB, nullable=True) # List[DreamFragmentMeta]
    metrics = Column(JSONB, nullable=True)   # DreamMetrics

    created_at = Column(DateTime(timezone=False), server_default=func.now())
