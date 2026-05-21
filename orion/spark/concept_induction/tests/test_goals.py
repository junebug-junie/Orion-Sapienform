from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.drives import DriveStateV1, GoalProposalV1, TensionEventV1
from orion.spark.concept_induction.goals import GoalProposalEngine


def _minimal_goal(**overrides):
    base = dict(
        artifact_id="goal-abc",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="memory.goals.proposed.v1",
        ts=datetime(2026, 5, 21, tzinfo=timezone.utc),
        confidence=0.7,
        provenance={
            "intake_channel": "orion:metacognition:tick",
            "correlation_id": "c1",
            "trace_id": "trace-long-id-12345",
            "source_event_refs": [],
            "evidence_items": [],
            "tension_refs": [],
        },
        goal_statement="Clarify autonomy boundaries without executing any new action.",
        proposal_signature="sig",
        drive_origin="autonomy",
        priority=0.5,
    )
    base.update(overrides)
    return GoalProposalV1.model_validate(base)


def test_goal_proposal_v1_accepts_new_optional_fields():
    goal = _minimal_goal(
        goal_statement_base="Clarify autonomy boundaries without executing any new action.",
        proposal_status="proposed",
        semantic_source="template",
    )
    assert goal.goal_statement_base.startswith("Clarify autonomy")
    assert goal.proposal_status == "proposed"
    assert goal.semantic_source == "template"


class _FakeStore:
    def __init__(self):
        self._cooldowns = {}
        self._goal_slots = {}

    def load_goal_cooldown(self, signature):
        return self._cooldowns.get(signature)

    def save_goal_cooldown(self, signature, until):
        self._cooldowns[signature] = {"cooldown_until": until.isoformat()}

    def record_goal_suppression(self, signature, now):
        rec = self._cooldowns.setdefault(signature, {})
        rec["suppressed_count"] = int(rec.get("suppressed_count", 0)) + 1

    def load_goal_slot(self, subject, drive_origin):
        return self._goal_slots.get(f"{subject}:{drive_origin}", {})

    def save_goal_slot(self, subject, drive_origin, *, signature, artifact_id):
        self._goal_slots[f"{subject}:{drive_origin}"] = {
            "signature": signature,
            "artifact_id": artifact_id,
        }


def _drive_state(trace_id: str) -> DriveStateV1:
    return DriveStateV1.model_validate({
        "subject": "orion",
        "model_layer": "self-model",
        "entity_id": "self:orion",
        "kind": "memory.drives.state.v1",
        "pressures": {"autonomy": 0.9},
        "activations": {"autonomy": True},
        "updated_at": datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
        "trace_id": trace_id,
        "provenance": {
            "intake_channel": "orion:metacognition:tick",
            "source_event_refs": [],
            "evidence_items": [],
            "tension_refs": [],
        },
    })


def test_drive_origin_from_audit_dominant():
    engine = GoalProposalEngine(cooldown_minutes=0)
    drive_state = DriveStateV1.model_validate({
        "subject": "orion",
        "model_layer": "self-model",
        "entity_id": "self:orion",
        "kind": "memory.drives.state.v1",
        "pressures": {"autonomy": 0.95, "relational": 0.4},
        "activations": {},
        "updated_at": datetime.now(timezone.utc),
        "provenance": {
            "intake_channel": "x",
            "source_event_refs": [],
            "evidence_items": [],
            "tension_refs": [],
        },
    })
    origin = engine._drive_origin(drive_state, dominant_drive="relational", source="audit_dominant")
    assert origin == "relational"


def test_signature_stable_when_trace_changes():
    engine = GoalProposalEngine(cooldown_minutes=0)
    store = _FakeStore()
    env = BaseEnvelope(
        kind="metacognition.tick.v1",
        source=ServiceRef(name="test", version="0"),
        correlation_id=uuid4(),
        payload={},
    )
    tensions: list[TensionEventV1] = []
    d1 = engine.propose(env=env, intake_channel="orion:metacognition:tick", drive_state=_drive_state("trace-aaa"), tensions=tensions, store=store)
    d2 = engine.propose(env=env, intake_channel="orion:metacognition:tick", drive_state=_drive_state("trace-bbb"), tensions=tensions, store=store)
    assert d1.proposal is not None
    assert d2.suppressed_signature == d1.proposal.proposal_signature
    assert "trace=" not in d1.proposal.goal_statement
    assert d1.proposal.goal_statement_base == d1.proposal.goal_statement


def test_signature_change_sets_supersedes_artifact_for_same_drive_origin():
    engine = GoalProposalEngine(cooldown_minutes=0)
    store = _FakeStore()
    env = BaseEnvelope(
        kind="metacognition.tick.v1",
        source=ServiceRef(name="test", version="0"),
        correlation_id=uuid4(),
        payload={},
    )
    d1 = engine.propose(
        env=env,
        intake_channel="orion:metacognition:tick",
        drive_state=_drive_state("trace-aaa"),
        tensions=[],
        store=store,
    )
    tension = TensionEventV1.model_validate({
        "artifact_id": "tension-1",
        "subject": "orion",
        "model_layer": "self-model",
        "entity_id": "self:orion",
        "kind": "tension.contradiction.v1",
        "magnitude": 0.8,
        "provenance": {
            "intake_channel": "orion:metacognition:tick",
            "source_event_refs": [],
            "evidence_items": [],
            "tension_refs": [],
        },
    })
    d2 = engine.propose(
        env=env,
        intake_channel="orion:metacognition:tick",
        drive_state=_drive_state("trace-bbb"),
        tensions=[tension],
        store=store,
    )
    assert d1.proposal is not None
    assert d2.proposal is not None
    assert d1.proposal.drive_origin == d2.proposal.drive_origin == "autonomy"
    assert d2.proposal.supersedes_artifact_id == d1.proposal.artifact_id
    assert d2.proposal.proposal_signature != d1.proposal.proposal_signature
