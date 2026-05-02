from __future__ import annotations

from datetime import datetime

from orion.autonomy.models import ActionOutcomeRefV1, AutonomyEvidenceRefV1, AutonomyStateV1, AutonomyStateV2
from orion.autonomy.reducer import AutonomyReducerInputV1, reduce_autonomy_state


def _base_v2(**overrides: object) -> AutonomyStateV2:
    v = AutonomyStateV2(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        latest_identity_snapshot_id="snap-x",
        latest_drive_audit_id="audit-x",
        source="graph",
        generated_at=datetime(2026, 1, 1, 12, 0, 0),
        schema_version="autonomy.state.v2",
        confidence=0.55,
        unknowns=[],
        evidence_refs=[],
        freshness={},
        attention_items=[],
        candidate_impulses=[],
        inhibited_impulses=[],
        last_action_outcomes=[],
    )
    return v.model_copy(update=overrides)


def test_reducer_cold_start_orion_binding() -> None:
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(subject="orion", previous_state=None, evidence=[], action_outcomes=[], now=fixed)
    )
    assert r.state.confidence == 0.25
    assert "no_previous_state" in r.state.unknowns
    assert r.state.entity_id == "self:orion"


def test_reducer_user_message_and_infra_no_drive_pressure() -> None:
    prior = _base_v2(
        drive_pressures={"coherence": 0.5, "continuity": 0.2},
        dominant_drive="coherence",
    )
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[
                AutonomyEvidenceRefV1(
                    evidence_id="u1",
                    source="user_message",
                    kind="user_turn",
                    summary="hello",
                    confidence=0.9,
                    observed_at=fixed,
                ),
                AutonomyEvidenceRefV1(
                    evidence_id="i1",
                    source="infra",
                    kind="infra_health",
                    summary="autonomy graph availability=available",
                    confidence=1.0,
                    observed_at=fixed,
                ),
            ],
            action_outcomes=[],
            now=fixed,
        )
    )
    assert r.state.drive_pressures.get("coherence", 0.0) == 0.5
    assert r.state.drive_pressures.get("continuity", 0.0) == 0.2


def test_reducer_capability_timeout_unavailable() -> None:
    prior = _base_v2(drive_pressures={"capability": 0.1})
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[
                AutonomyEvidenceRefV1(
                    evidence_id="e1",
                    source="graph",
                    kind="incident",
                    summary="GraphDB timeout unavailable",
                    confidence=0.8,
                    observed_at=fixed,
                )
            ],
            action_outcomes=[],
            now=fixed,
        )
    )
    assert r.state.drive_pressures.get("capability", 0.0) > 0.1
    assert "tension.capability_gap.v1" in r.state.tension_kinds


def test_reducer_polarity_blind_no_contradiction_still_raises_coherence() -> None:
    """Polarity-blind token match: 'contradiction' hits inside 'no contradiction'."""
    prior = _base_v2(drive_pressures={"coherence": 0.1})
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[
                AutonomyEvidenceRefV1(
                    evidence_id="e1",
                    source="graph",
                    kind="note",
                    summary="no contradiction",
                    confidence=0.6,
                    observed_at=fixed,
                )
            ],
            action_outcomes=[],
            now=fixed,
        )
    )
    assert r.state.drive_pressures.get("coherence", 0.0) > 0.1


def test_reducer_all_proxy_inhibition_and_unknown() -> None:
    prior = _base_v2(confidence=0.55, drive_pressures={"coherence": 0.05})
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[
                AutonomyEvidenceRefV1(
                    evidence_id="p1",
                    source="phi",
                    kind="proxy_telemetry",
                    summary="spark bump",
                    confidence=0.4,
                    observed_at=fixed,
                ),
                AutonomyEvidenceRefV1(
                    evidence_id="p2",
                    source="phi",
                    kind="proxy_telemetry",
                    summary="telemetry only",
                    confidence=0.4,
                    observed_at=fixed,
                ),
            ],
            action_outcomes=[],
            now=fixed,
        )
    )
    assert "proxy_only_evidence" in r.state.unknowns
    reasons = {i.inhibition_reason for i in r.state.inhibited_impulses}
    assert "proxy_signal_not_canonical_state" in reasons
    assert r.state.candidate_impulses == []


def test_reducer_high_surprise_outcome_reduces_confidence() -> None:
    prior = _base_v2(confidence=0.8)
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[],
            action_outcomes=[
                ActionOutcomeRefV1(
                    action_id="a1",
                    kind="tool",
                    summary="unexpected result",
                    surprise=0.9,
                    observed_at=fixed,
                )
            ],
            now=fixed,
        )
    )
    assert r.state.confidence < 0.8


def test_reducer_determinism_fixed_now() -> None:
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    v1 = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive="coherence",
        active_drives=["coherence"],
        drive_pressures={"coherence": 0.3},
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
        generated_at=fixed,
    )
    inp = AutonomyReducerInputV1(
        subject="orion",
        previous_state=v1,
        evidence=[
            AutonomyEvidenceRefV1(
                evidence_id="z1",
                source="graph",
                kind="note",
                summary="regression detected",
                confidence=0.8,
                observed_at=fixed,
            )
        ],
        action_outcomes=[],
        now=fixed,
    )
    a = reduce_autonomy_state(inp).state.model_dump(mode="json")
    b = reduce_autonomy_state(inp).state.model_dump(mode="json")
    assert a == b


def test_reducer_evidence_trim_twenty() -> None:
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    evs = [
        AutonomyEvidenceRefV1(
            evidence_id=f"id-{i}",
            source="graph",
            kind="batch",
            summary=f"item-{i}",
            confidence=0.5,
            observed_at=datetime(2026, 5, 2, 12, 0, i),
        )
        for i in range(25)
    ]
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=None,
            evidence=evs,
            action_outcomes=[],
            now=fixed,
        )
    )
    assert len(r.state.evidence_refs) <= 20
