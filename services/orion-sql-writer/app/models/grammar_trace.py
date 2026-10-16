from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Index, Integer, String, Text
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
        Index("idx_grammar_events_source_created", "source_service", "created_at", "event_id"),
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


class GrammarActSQL(Base):
    """One row per act (orion.schemas.grammar.ActV1): opened, then closed by act_id.

    Written by orion/grammar/act_ledger.py. The row is the efference copy: what
    the act expected (``predicted`` is the EffectPosterior mean stamped at
    open), what happened, and the error. A closed act is also graded into
    substrate_action_outcomes / substrate_action_effect_posterior
    (``outcome_row_id`` points at that row).
    """

    __tablename__ = "grammar_acts"

    act_id = Column(String, primary_key=True)
    parent_act_id = Column(String, nullable=True)
    trace_id = Column(String, nullable=False)
    level = Column(String, nullable=False)
    actor = Column(String, nullable=False)
    verb = Column(String, nullable=False)
    object_ref = Column(Text, nullable=True)
    test_before = Column(Text, nullable=True)
    signal_id = Column(String, nullable=True)
    direction = Column(String, nullable=True)
    horizon_sec = Column(Float, nullable=True)
    expect_basis = Column(Text, nullable=True)
    predicted = Column(Float, nullable=True)
    predicted_n = Column(Integer, nullable=True)
    cold_start = Column(Boolean, nullable=True)
    goal_ref = Column(String, nullable=True)
    workspace_ref = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    correlation_id = Column(String, nullable=True)
    opened_at = Column(DateTime(timezone=True), nullable=True)
    horizon_at = Column(DateTime(timezone=True), nullable=True)
    open_event_id = Column(String, nullable=True)
    cost_json = Column(JSONB, nullable=True)
    outcome_value = Column(Float, nullable=True)
    outcome_source = Column(String, nullable=True)
    outcome_note = Column(Text, nullable=True)
    error = Column(Float, nullable=True)
    surprise_nats = Column(Float, nullable=True)
    outcome_row_id = Column(BigInteger, nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    close_event_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_grammar_acts_open_horizon", "horizon_at", postgresql_where=closed_at.is_(None)),
        Index("idx_grammar_acts_verb_opened", "verb", "opened_at"),
    )
