from __future__ import annotations

from datetime import datetime

from orion.autonomy.models import AutonomyEvidenceRefV1, AutonomyStateV1, AutonomyStateV2, upgrade_autonomy_state_v1_to_v2


def test_upgrade_preserves_v1_fields_and_sets_v2_schema() -> None:
    v1 = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        latest_identity_snapshot_id="snap-1",
        latest_drive_audit_id="audit-1",
        latest_goal_ids=["g1"],
        dominant_drive="coherence",
        active_drives=["coherence"],
        drive_pressures={"coherence": 0.4},
        tension_kinds=["tension.x"],
        goal_headlines=[],
        source="graph",
        generated_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    v2 = upgrade_autonomy_state_v1_to_v2(v1)
    assert isinstance(v2, AutonomyStateV2)
    assert v2.schema_version == "autonomy.state.v2"
    assert v2.subject == v1.subject
    assert v2.drive_pressures == v1.drive_pressures
    assert v2.confidence == 0.55
    assert "no_action_outcome_history" in v2.unknowns
    assert "evidence_from_graph_only" in v2.unknowns


def test_upgrade_evidence_ids_stable() -> None:
    v1 = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        latest_identity_snapshot_id="snap-a",
        latest_drive_audit_id="audit-b",
        latest_goal_ids=["g1", "g2"],
        dominant_drive=None,
        active_drives=[],
        drive_pressures={},
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
        generated_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    v2 = upgrade_autonomy_state_v1_to_v2(v1)
    ids = [e.evidence_id for e in v2.evidence_refs]
    assert "identity_snapshot:snap-a" in ids
    assert "drive_audit:audit-b" in ids
    assert "goal_ref:g1" in ids
    assert "goal_ref:g2" in ids


def _merge_evidence_by_id(a: list[AutonomyEvidenceRefV1], b: list[AutonomyEvidenceRefV1]) -> list[AutonomyEvidenceRefV1]:
    by_id: dict[str, AutonomyEvidenceRefV1] = {}
    for ref in a + b:
        by_id[ref.evidence_id] = ref
    return list(by_id.values())


def test_upgrade_merge_no_duplicate_evidence() -> None:
    v1 = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        latest_identity_snapshot_id="snap-1",
        latest_drive_audit_id="audit-1",
        latest_goal_ids=["g1"],
        dominant_drive="coherence",
        active_drives=["coherence"],
        drive_pressures={},
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
        generated_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    u1 = upgrade_autonomy_state_v1_to_v2(v1)
    u2 = upgrade_autonomy_state_v1_to_v2(v1)
    merged = _merge_evidence_by_id(list(u1.evidence_refs), list(u2.evidence_refs))
    assert len(merged) == len(u1.evidence_refs)
    assert {e.evidence_id for e in merged} == {e.evidence_id for e in u1.evidence_refs}


def test_upgrade_unknowns_without_snapshot_audit() -> None:
    v1 = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        latest_identity_snapshot_id=None,
        latest_drive_audit_id=None,
        latest_goal_ids=[],
        dominant_drive=None,
        active_drives=[],
        drive_pressures={},
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
        generated_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    v2 = upgrade_autonomy_state_v1_to_v2(v1)
    assert "no_identity_snapshot" in v2.unknowns
    assert "no_drive_audit" in v2.unknowns


def test_upgrade_attention_when_dominant_or_tensions() -> None:
    flat = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        latest_identity_snapshot_id=None,
        latest_drive_audit_id=None,
        latest_goal_ids=[],
        dominant_drive=None,
        active_drives=[],
        drive_pressures={},
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
        generated_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    assert upgrade_autonomy_state_v1_to_v2(flat).attention_items == []

    with_dom = flat.model_copy(update={"dominant_drive": "coherence"})
    v2 = upgrade_autonomy_state_v1_to_v2(with_dom)
    assert len(v2.attention_items) >= 1
