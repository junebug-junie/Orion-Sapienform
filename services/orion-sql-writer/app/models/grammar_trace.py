from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from app.db import Base


class GrammarTraceSQL(Base):
    __tablename__ = "grammar_traces"

    trace_id = Column(String, primary_key=True)
    trace_type = Column(String, nullable=False)
    session_id = Column(String, nullable=True)
    turn_id = Column(String, nullable=True)
    root_event_id = Column(String, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String, nullable=False, default="open")
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)


class GrammarEventSQL(Base):
    __tablename__ = "grammar_events"

    event_id = Column(String, primary_key=True)
    trace_id = Column(String, nullable=False)
    parent_event_id = Column(String, nullable=True)
    root_event_id = Column(String, nullable=True)
    event_kind = Column(String, nullable=False)
    session_id = Column(String, nullable=True)
    turn_id = Column(String, nullable=True)
    correlation_id = Column(String, nullable=True)
    layer = Column(String, nullable=True)
    dimensions = Column(ARRAY(String), nullable=False, default=list)
    emitted_at = Column(DateTime(timezone=True), nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=True)
    source_service = Column(String, nullable=False)
    source_component = Column(String, nullable=True)
    source_event_id = Column(String, nullable=True)
    payload_ref = Column(String, nullable=True)
    event_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_grammar_events_trace_id", "trace_id"),
        Index("idx_grammar_events_session_turn", "session_id", "turn_id"),
    )


class GrammarAtomSQL(Base):
    __tablename__ = "grammar_atoms"

    atom_id = Column(String, primary_key=True)
    trace_id = Column(String, nullable=False)
    atom_type = Column(String, nullable=False)
    semantic_role = Column(String, nullable=False)
    layer = Column(String, nullable=False)
    dimensions = Column(ARRAY(String), nullable=False, default=list)
    summary = Column(Text, nullable=False)
    text_value = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    salience = Column(Float, nullable=True)
    uncertainty = Column(Float, nullable=True)
    time_start = Column(DateTime(timezone=True), nullable=True)
    time_end = Column(DateTime(timezone=True), nullable=True)
    source_event_id = Column(String, nullable=True)
    payload_ref = Column(String, nullable=True)
    renderer_hint = Column(String, nullable=True)
    atom_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_grammar_atoms_trace_id", "trace_id"),
        Index("idx_grammar_atoms_layer", "layer"),
        Index("idx_grammar_atoms_dimensions", "dimensions", postgresql_using="gin"),
    )


class GrammarEdgeSQL(Base):
    __tablename__ = "grammar_edges"

    edge_id = Column(String, primary_key=True)
    trace_id = Column(String, nullable=False)
    from_atom_id = Column(String, nullable=False)
    to_atom_id = Column(String, nullable=False)
    relation_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    salience = Column(Float, nullable=True)
    layer_from = Column(String, nullable=True)
    layer_to = Column(String, nullable=True)
    temporal_relation = Column(String, nullable=True)
    evidence_event_ids = Column(ARRAY(String), nullable=False, default=list)
    edge_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_grammar_edges_trace_id", "trace_id"),
        Index("idx_grammar_edges_from", "from_atom_id"),
        Index("idx_grammar_edges_to", "to_atom_id"),
    )


class GrammarTemporalHopSQL(Base):
    __tablename__ = "grammar_temporal_hops"

    hop_id = Column(String, primary_key=True)
    trace_id = Column(String, nullable=False)
    from_atom_id = Column(String, nullable=False)
    to_atom_id = Column(String, nullable=True)
    hop_type = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    reason = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    turn_distance = Column(Integer, nullable=True)
    session_distance = Column(Integer, nullable=True)
    target_time_start = Column(DateTime(timezone=True), nullable=True)
    target_time_end = Column(DateTime(timezone=True), nullable=True)
    hop_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("idx_grammar_temporal_hops_trace_id", "trace_id"),)


class GrammarCompactionSQL(Base):
    __tablename__ = "grammar_compactions"

    compaction_id = Column(String, primary_key=True)
    trace_id = Column(String, nullable=False)
    source_atom_ids = Column(ARRAY(String), nullable=False)
    output_atom_id = Column(String, nullable=False)
    compaction_type = Column(String, nullable=False)
    method = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    preserves = Column(ARRAY(String), nullable=False, default=list)
    drops = Column(ARRAY(String), nullable=False, default=list)
    confidence = Column(Float, nullable=True)
    compaction_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class GrammarProjectionSQL(Base):
    __tablename__ = "grammar_projections"

    projection_id = Column(String, primary_key=True)
    trace_id = Column(String, nullable=False)
    source_atom_ids = Column(ARRAY(String), nullable=False)
    projection_type = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    projected_atom_id = Column(String, nullable=True)
    projection_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
