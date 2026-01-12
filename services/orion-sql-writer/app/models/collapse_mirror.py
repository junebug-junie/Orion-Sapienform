from sqlalchemy import Column, String, Text, Boolean, Float
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base

class CollapseMirror(Base):
    __tablename__ = "collapse_mirror"

    # --- V1 Fields (Existing) ---
    id = Column(String, primary_key=True, index=True) # Maps to Pydantic event_id
    correlation_id = Column(String, index=True, nullable=True)
    observer = Column(String)
    trigger = Column(Text)

    # Note: V2 Pydantic sends a List[str] here.
    observer_state = Column(String)

    field_resonance = Column(Text)
    type = Column(String)
    emergent_entity = Column(Text)
    summary = Column(Text)
    mantra = Column(Text)
    causal_echo = Column(Text, nullable=True)
    timestamp = Column(String, nullable=True)
    environment = Column(String, nullable=True)

    # --- V2 Fields (New) ---
    snapshot_kind = Column(String, default="baseline")
    is_causally_dense = Column(Boolean, default=False)
    epistemic_status = Column(String, default="observed")
    visibility = Column(String, default="internal")
    redaction_level = Column(String, default="low")
    source_service = Column(String, nullable=True)
    source_node = Column(String, nullable=True)
    what_changed_summary = Column(Text, nullable=True)
    pattern_candidate = Column(String, nullable=True)
    resonance_signature = Column(String, nullable=True)
    change_type = Column(String, nullable=True)

    # --- Complex Structures (JSONB) ---
    numeric_sisters = Column(JSONB, nullable=True)
    causal_density = Column(JSONB, nullable=True)
    state_snapshot = Column(JSONB, nullable=True)
    what_changed = Column(JSONB, nullable=True)
    tags = Column(JSONB, default=list)
    tag_scores = Column(JSONB, default=dict)
    change_type_scores = Column(JSONB, default=dict)
