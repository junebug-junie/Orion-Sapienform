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
    assert r.state.confidence == 0.20
    assert "no_previous_state" in r.state.unknowns
    assert r.state.entity_id == "self:orion"


def test_reducer_cold_start_juniper_and_relationship_bindings() -> None:
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    rj = reduce_autonomy_state(
        AutonomyReducerInputV1(subject="juniper", previous_state=None, evidence=[], action_outcomes=[], now=fixed)
    )
    assert rj.state.entity_id == "user:juniper"
    rr = reduce_autonomy_state(
        AutonomyReducerInputV1(subject="relationship", previous_state=None, evidence=[], action_outcomes=[], now=fixed)
    )
    assert rr.state.entity_id == "relationship:orion|juniper"


def test_reducer_no_fresh_evidence_when_merged_empty_only() -> None:
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    cold = reduce_autonomy_state(
        AutonomyReducerInputV1(subject="orion", previous_state=None, evidence=[], action_outcomes=[], now=fixed)
    )
    assert "no_fresh_evidence" in cold.state.unknowns
    assert cold.state.evidence_refs == []

    v1 = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        latest_identity_snapshot_id="snap-1",
        latest_drive_audit_id="audit-1",
        latest_goal_ids=["g1"],
        dominant_drive="coherence",
        active_drives=["coherence"],
        drive_pressures={"coherence": 0.3},
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
        generated_at=fixed,
    )
    with_graph = reduce_autonomy_state(
        AutonomyReducerInputV1(subject="orion", previous_state=v1, evidence=[], action_outcomes=[], now=fixed)
    )
    assert "no_fresh_evidence" not in with_graph.state.unknowns
    assert len(with_graph.state.evidence_refs) > 0


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


def test_reducer_mapped_hazard_moves_relational_pressure() -> None:
    prior = _base_v2(drive_pressures={"relational": 0.05, "coherence": 0.1})
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[
                AutonomyEvidenceRefV1(
                    evidence_id="h1",
                    source="social_bridge",
                    kind="relational_signal",
                    summary="cooldown_active",
                    confidence=0.6,
                    observed_at=fixed,
                    signal_kind="chat_social_hazard",
                    dimension="cooldown_active",
                    value=1.0,
                )
            ],
            action_outcomes=[],
            now=fixed,
        )
    )
    assert r.state.drive_pressures["relational"] > 0.05
    assert abs(r.delta.drive_deltas.get("relational", 0.0)) > 0.0
    assert "latest_direct_evidence_at" in r.state.freshness


def test_reducer_unmapped_hazard_does_not_move_pressures() -> None:
    prior = _base_v2(drive_pressures={"relational": 0.2, "coherence": 0.2})
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    before = dict(prior.drive_pressures)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[
                AutonomyEvidenceRefV1(
                    evidence_id="h1",
                    source="social_bridge",
                    kind="relational_signal",
                    summary="context_excluded:memory",
                    confidence=0.6,
                    observed_at=fixed,
                )
            ],
            action_outcomes=[],
            now=fixed,
        )
    )
    for k in before:
        assert r.state.drive_pressures.get(k, 0.0) == before.get(k, 0.0)


def test_reducer_prose_keywords_no_longer_move_pressures() -> None:
    """Acceptance: keyword tokens must not move pressures."""
    prior = _base_v2(drive_pressures={"coherence": 0.1, "capability": 0.1, "relational": 0.1})
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
                    summary="frustration repair contradiction timeout unavailable stale",
                    confidence=0.8,
                    observed_at=fixed,
                )
            ],
            action_outcomes=[],
            now=fixed,
        )
    )
    assert r.state.drive_pressures["coherence"] == 0.1
    assert r.state.drive_pressures["capability"] == 0.1
    assert r.state.drive_pressures["relational"] == 0.1
    # Tension kinds must not OR in from the prose blob either.
    assert "tension.coherence_break.v1" not in r.state.tension_kinds
    assert "tension.capability_gap.v1" not in r.state.tension_kinds
    assert "tension.relational_repair.v1" not in r.state.tension_kinds


def test_reducer_reasoning_fallback_moves_coherence() -> None:
    prior = _base_v2(drive_pressures={"coherence": 0.05, "predictive": 0.05})
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[
                AutonomyEvidenceRefV1(
                    evidence_id="reasoning:fallback_recommended",
                    source="reasoning",
                    kind="reasoning_quality",
                    summary="reasoning fallback recommended",
                    confidence=0.6,
                    observed_at=fixed,
                    signal_kind="chat_reasoning_quality",
                    dimension="fallback",
                    value=1.0,
                )
            ],
            action_outcomes=[],
            now=fixed,
        )
    )
    assert r.state.drive_pressures["coherence"] > 0.05
    assert r.state.drive_pressures["predictive"] > 0.05


def test_reducer_tension_kinds_from_pressure_thresholds_only() -> None:
    prior = _base_v2(drive_pressures={"coherence": 0.30})
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    # No incoming evidence; prior pressure already at threshold.
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[],
            action_outcomes=[],
            now=fixed,
        )
    )
    assert "tension.coherence_break.v1" in r.state.tension_kinds


def test_reducer_no_keyword_helpers_in_module() -> None:
    import inspect
    from orion.autonomy import reducer as reducer_mod

    src = inspect.getsource(reducer_mod)
    for banned in (
        "_apply_single_evidence_pressures",
        '"contradiction"',
        '"frustration"',
        '"stale" in blob',
        '"timeout" in blob',
    ):
        assert banned not in src, banned


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


def test_reducer_determinism_fixed_now_typed() -> None:
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    prior = _base_v2(drive_pressures={"relational": 0.0})
    inp = AutonomyReducerInputV1(
        subject="orion",
        previous_state=prior,
        evidence=[
            AutonomyEvidenceRefV1(
                evidence_id="z1",
                source="social_bridge",
                kind="relational_signal",
                summary="self_message_loop",
                confidence=0.6,
                observed_at=fixed,
                signal_kind="chat_social_hazard",
                dimension="self_message_loop",
                value=1.0,
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
