from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.types import JSON
from sqlalchemy.sql import func

from .db import Base


class SocialParticipantContinuitySQL(Base):
    __tablename__ = "social_participant_continuity"

    peer_key = Column(String, primary_key=True)
    platform = Column(String, nullable=False, index=True)
    room_id = Column(String, nullable=False, index=True)
    participant_id = Column(String, nullable=False)
    participant_name = Column(String, nullable=True)
    aliases = Column(JSON, nullable=True)
    participant_kind = Column(String, nullable=False)
    recent_shared_topics = Column(JSON, nullable=True)
    interaction_tone_summary = Column(Text, nullable=False, default="")
    safe_continuity_summary = Column(Text, nullable=False, default="")
    evidence_refs = Column(JSON, nullable=True)
    evidence_count = Column(Integer, nullable=False, default=0)
    last_seen_at = Column(String, nullable=False)
    confidence = Column(Float, nullable=False, default=0.0)
    trust_tier = Column(String, nullable=False, default="new")
    shared_artifact_scope = Column(String, nullable=False, default="peer_local")
    shared_artifact_status = Column(String, nullable=False, default="unknown")
    shared_artifact_summary = Column(Text, nullable=False, default="")
    shared_artifact_reason = Column(Text, nullable=False, default="")
    shared_artifact_proposal = Column(JSON, nullable=True)
    shared_artifact_revision = Column(JSON, nullable=True)
    shared_artifact_confirmation = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class SocialRoomContinuitySQL(Base):
    __tablename__ = "social_room_continuity"

    room_key = Column(String, primary_key=True)
    platform = Column(String, nullable=False, index=True)
    room_id = Column(String, nullable=False, index=True)
    recurring_topics = Column(JSON, nullable=True)
    active_participants = Column(JSON, nullable=True)
    recent_thread_summary = Column(Text, nullable=False, default="")
    room_tone_summary = Column(Text, nullable=False, default="")
    open_threads = Column(JSON, nullable=True)
    evidence_refs = Column(JSON, nullable=True)
    evidence_count = Column(Integer, nullable=False, default=0)
    last_updated_at = Column(String, nullable=False)
    shared_artifact_scope = Column(String, nullable=False, default="room_local")
    shared_artifact_status = Column(String, nullable=False, default="unknown")
    shared_artifact_summary = Column(Text, nullable=False, default="")
    shared_artifact_reason = Column(Text, nullable=False, default="")
    shared_artifact_proposal = Column(JSON, nullable=True)
    shared_artifact_revision = Column(JSON, nullable=True)
    shared_artifact_confirmation = Column(JSON, nullable=True)
    active_threads = Column(JSON, nullable=True)
    current_thread_key = Column(String, nullable=True)
    current_thread_summary = Column(Text, nullable=False, default="")
    handoff_signal = Column(JSON, nullable=True)
    active_claims = Column(JSON, nullable=True)
    recent_claim_revisions = Column(JSON, nullable=True)
    claim_attributions = Column(JSON, nullable=True)
    claim_consensus_states = Column(JSON, nullable=True)
    claim_divergence_signals = Column(JSON, nullable=True)
    bridge_summary = Column(JSON, nullable=True)
    clarifying_question = Column(JSON, nullable=True)
    deliberation_decision = Column(JSON, nullable=True)
    turn_handoff = Column(JSON, nullable=True)
    closure_signal = Column(JSON, nullable=True)
    floor_decision = Column(JSON, nullable=True)
    active_commitments = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class SocialStanceSnapshotSQL(Base):
    __tablename__ = "social_stance_snapshot"

    stance_id = Column(String, primary_key=True)
    curiosity = Column(Float, nullable=False, default=0.5)
    warmth = Column(Float, nullable=False, default=0.7)
    directness = Column(Float, nullable=False, default=0.6)
    playfulness = Column(Float, nullable=False, default=0.3)
    caution = Column(Float, nullable=False, default=0.4)
    depth_preference = Column(Float, nullable=False, default=0.5)
    recent_social_orientation_summary = Column(Text, nullable=False, default="")
    evidence_refs = Column(JSON, nullable=True)
    evidence_count = Column(Integer, nullable=False, default=0)
    last_updated_at = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class SocialPeerStyleHintSQL(Base):
    __tablename__ = "social_peer_style_hint"

    peer_style_key = Column(String, primary_key=True)
    platform = Column(String, nullable=False, index=True)
    room_id = Column(String, nullable=False, index=True)
    participant_id = Column(String, nullable=False)
    participant_name = Column(String, nullable=True)
    style_hints_summary = Column(Text, nullable=False, default="")
    preferred_directness = Column(Float, nullable=False, default=0.5)
    preferred_depth = Column(Float, nullable=False, default=0.5)
    question_appetite = Column(Float, nullable=False, default=0.5)
    playfulness_tendency = Column(Float, nullable=False, default=0.3)
    formality_tendency = Column(Float, nullable=False, default=0.5)
    summarization_preference = Column(Float, nullable=False, default=0.3)
    evidence_count = Column(Integer, nullable=False, default=0)
    confidence = Column(Float, nullable=False, default=0.0)
    last_updated_at = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class SocialRoomRitualSummarySQL(Base):
    __tablename__ = "social_room_ritual_summary"

    ritual_key = Column(String, primary_key=True)
    platform = Column(String, nullable=False, index=True)
    room_id = Column(String, nullable=False, index=True)
    greeting_style = Column(String, nullable=False, default="warm")
    reentry_style = Column(String, nullable=False, default="grounded")
    thread_revival_style = Column(String, nullable=False, default="direct")
    pause_handoff_style = Column(String, nullable=False, default="brief")
    summary_cadence_preference = Column(Float, nullable=False, default=0.3)
    room_tone_summary = Column(Text, nullable=False, default="")
    culture_summary = Column(Text, nullable=False, default="")
    evidence_count = Column(Integer, nullable=False, default=0)
    confidence = Column(Float, nullable=False, default=0.0)
    last_updated_at = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
