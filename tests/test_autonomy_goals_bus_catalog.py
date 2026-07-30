"""Bus catalog and schema registry coverage for autonomy goal artifacts."""

from __future__ import annotations

from pathlib import Path

import yaml

from orion.core.schemas.drives import AutonomyGoalPlannedV1, GoalProposalV1
from orion.schemas.field_goal import FieldGoalProvenanceV1
from orion.schemas.registry import _REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"


def _channel_entry(name: str) -> dict:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8")) or {}
    for entry in doc.get("channels") or []:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    raise AssertionError(f"Missing channel catalog entry: {name}")


def test_goal_proposal_schema_in_registry() -> None:
    assert "GoalProposalV1" in _REGISTRY
    assert resolve("GoalProposalV1") is GoalProposalV1
    goal = GoalProposalV1.model_validate(
        {
            "artifact_id": "goal-1",
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "memory.goals.proposed.v1",
            "goal_statement": "Clarify autonomy boundaries without executing any new action.",
            "proposal_signature": "sig-1",
            "drive_origin": "autonomy",
            "priority": 0.5,
            "provenance": {
                "intake_channel": "orion:metacognition:tick",
                "source_event_refs": [],
                "evidence_items": [],
                "tension_refs": [],
            },
        }
    )
    assert goal.proposal_status == "proposed"
    assert goal.semantic_source == "template"


def test_field_goal_provenance_schema_in_registry() -> None:
    assert "FieldGoalProvenanceV1" in _REGISTRY
    assert resolve("FieldGoalProvenanceV1") is FieldGoalProvenanceV1
    goal = FieldGoalProvenanceV1.model_validate(
        {
            "artifact_id": "goal-1",
            "subject": "attention",
            "model_layer": "field_attention",
            "entity_id": "node:substrate.biometrics",
            "kind": "memory.field_goals.proposed.v1",
            "field_target_id": "node:substrate.biometrics",
            "target_kind": "node",
            "salience_score": 0.8,
            "source_field_tick_id": "tick-1",
            "source_attention_frame_id": "frame-1",
            "priority": 0.8,
            "provenance": {"intake_channel": "internal.attention_runtime"},
        }
    )
    assert goal.proposal_status == "proposed"


def test_memory_goals_proposed_channel_cataloged() -> None:
    # Repointed 2026-07-30 (Sentience Striving Program sec6 Objective 3):
    # orion-spark-concept-induction's GoalProposalEngine (the channel's only
    # producer until then) was deleted alongside the rest of drive-pressure/
    # goal-generation (orion/sentience_striving_program/README.md sec8),
    # leaving the channel producer-less. New real producer:
    # orion-attention-runtime, publishing FieldGoalProvenanceV1 -- see
    # orion/bus/channels.yaml's comment on this entry and
    # docs/superpowers/specs/2026-07-30-goal-provenance-and-decision-lattice-
    # observability-design.md. GoalProposalV1 the schema is still real
    # (constructed in-memory elsewhere for internal policy evaluation, never
    # published to any channel) -- covered by
    # test_goal_proposal_schema_in_registry above; it is simply no longer
    # this channel's schema.
    entry = _channel_entry("orion:memory:goals:proposed")
    assert entry["schema_id"] == "FieldGoalProvenanceV1"
    assert entry["message_kind"] == "memory.field_goals.proposed.v1"
    assert entry.get("producer_services") == ["orion-attention-runtime"]


def test_autonomy_goal_planned_schema_and_channel_cataloged() -> None:
    assert "AutonomyGoalPlannedV1" in _REGISTRY
    assert resolve("AutonomyGoalPlannedV1") is AutonomyGoalPlannedV1

    entry = _channel_entry("orion:autonomy:goal:planned")
    assert entry["schema_id"] == "AutonomyGoalPlannedV1"
    assert entry["message_kind"] == "autonomy.goal.planned.v1"
    assert "orion-cortex-exec" in (entry.get("producer_services") or [])

    planned = AutonomyGoalPlannedV1.model_validate(
        {
            "goal_artifact_id": "goal-abc",
            "goal_statement": "Archive stale goals",
            "drive_origin": "autonomy",
            "task_id": "goal-task-1",
        }
    )
    assert planned.kind == "autonomy.goal.planned.v1"
