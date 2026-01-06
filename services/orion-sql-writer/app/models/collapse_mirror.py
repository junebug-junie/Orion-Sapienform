from sqlalchemy import Column, String, Text
from app.db import Base

class CollapseMirror(Base):
    __tablename__ = "collapse_mirror"

    id = Column(String, primary_key=True, index=True)
    correlation_id = Column(String, index=True, nullable=True)
    observer = Column(String)
    trigger = Column(Text)
    observer_state = Column(String)
    field_resonance = Column(Text)
    type = Column(String)
    emergent_entity = Column(Text)
    summary = Column(Text)
    mantra = Column(Text)
    causal_echo = Column(Text, nullable=True)
    timestamp = Column(String, nullable=True)
    environment = Column(String, nullable=True)
