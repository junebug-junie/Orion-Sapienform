from sqlalchemy import Column, String, Text, Boolean, Float
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base

class CollapseMirror(Base):
    __tablename__ = "collapse_mirror"

    # --- V1 Fields (Existing) ---
    id = Column(String, primary_key=True, index=True)
    correlation_id = Column(String, index=True, nullable=True)
    observer = Column(String)
    trigger = Column(Text)
    observer_state = Column(String)  # Kept for legacy, V2 uses state_snapshot
    field_resonance = Column(Text)
    type = Column(String)
    emergent_entity = Column(Text)
    summary = Column(Text)
    mantra = Column(Text)
    causal_echo = Column(Text, nullable=True)
    timestamp = Column(String, nullable=True)
    environment = Column(String, nullable=True)

    # --- V2 Fields (New) ---
    snapshot_kind = Column(String, default="baseline")  # baseline | event
    is_causally_dense = Column(Boolean, default=False)

    # Complex Structures (JSONB)
    numeric_sisters = Column(JSONB, nullable=True)      # Valence, Arousal, Clarity
    causal_density = Column(JSONB, nullable=True)       # Score + Rationale
    state_snapshot = Column(JSONB, nullable=True)       # Full telemetry snapshot
    what_changed = Column(JSONB, nullable=True)         # Diff summary
    tags = Column(JSONB, default=list)                  # List[str]
    tag_scores = Column(JSONB, default=dict)            # Dict[str, float]
