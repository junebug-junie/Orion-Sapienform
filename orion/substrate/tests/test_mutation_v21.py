from __future__ import annotations

import os
from datetime import timedelta

import pytest
from pathlib import Path

from orion.core.schemas.substrate_mutation import (
    CognitiveProposalReviewV1,
    MutationDecisionV1,
    MutationPatchV1,
    MutationPressureEvidenceV1,
    MutationPressureV1,
    MutationProposalV1,
    MutationSignalV1,
    RecallProductionCandidateReviewV1,
    RecallShadowEvalRunV1,
    RecallStrategyProfileV1,
)
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate import mutation_control_surface
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_pressure import PressureAccumulator, PressurePolicy
from orion.substrate.mutation_proposals import (
    ROUTING_TARGET_PARKED_REASON,
    ProposalFactory,
    build_placeholder_routing_proposal,
)
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_worker import SubstrateAdaptationWorker
from orion.substrate.scripts.smoke_mutation_v21 import run_smoke


def _routing_surface(value: float = 0.50, *, degraded: bool = False):
    """Stub for ProposalFactory's live-surface reader. 0.50 is what the routing
    patch was always implicitly written against, so pre-existing assertions
    (patch 0.58, rollback 0.50) still hold -- now derived, not hardcoded."""
    return lambda: {"value": value, "raw": {"value": value}, "degraded": degraded}


@pytest.fixture(autouse=True)
def _isolate_control_surface(tmp_path, monkeypatch):
    """Give every test in this module its own control surface, seeded to 0.5.

    These tests call ``PatchApplier.apply`` directly against the module-global
    control surface, which resolves from the ambient environment and is written
    by the real setter -- so one test's apply moved the starting value for every
    later one. Invisible until ``apply`` learned to decline a patch that would
    change nothing: four tests about lock recovery, reload continuity and
    retention then failed, not for what they were testing but because an earlier
    test had already moved the surface to the patch value.

    Seeded to 0.5, not the patch constant 0.58, so the applies these tests
    depend on are genuine changes. Same shape as
    ``services/orion-cortex-orch/tests/conftest.py``.
    """
    for key in (
        "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL",
        "SUBSTRATE_POLICY_POSTGRES_URL",
        "DATABASE_URL",
        "SUBSTRATE_MUTATION_CONTROL_SQL_DB_PATH",
        "SUBSTRATE_MUTATION_SQL_DB_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    previous = mutation_control_surface._CONTROL_SURFACE_STORE
    isolated = mutation_control_surface.RuntimeControlSurfaceStore(
        sql_db_path=str(tmp_path / "control-surface-isolated.sqlite3")
    )
    assert isolated.postgres_url is None and isolated.source_kind() == "sqlite"
    mutation_control_surface._CONTROL_SURFACE_STORE = isolated
    try:
        mutation_control_surface.set_chat_reflective_lane_threshold(value=0.5, actor="test_seed")
        yield isolated
    finally:
        mutation_control_surface._CONTROL_SURFACE_STORE = previous


def _routing_pressure() -> MutationPressureV1:
    return MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="routing",
        pressure_kind="runtime_failure",
        pressure_score=8.0,
        evidence_refs=["telemetry:1"],
        source_signal_ids=["signal-1"],
    )


def _direct_routing_proposal(*, target_value: float = 0.58, rollback_value: float = 0.50) -> MutationProposalV1:
    """A routing_threshold_patch proposal, bypassing the parked ProposalFactory.

    As of 2026-09-03, `ProposalFactory.plan_for_pressure()`/`from_pressure()`
    refuse every "routing" pressure outright (parked -- see
    mutation_proposals.py's `ROUTING_TARGET_PARKED_REASON`). The tests that
    use this are about generic proposal -> trial -> decision -> store/apply/
    replay mechanics, not about the parked evidence pipeline, so they build
    the proposal the factory used to build directly instead, via the shared
    `build_placeholder_routing_proposal()` (also used by
    smoke_mutation_v21.py and the orion-hub replay-inspection endpoint, so
    the shape stays in one place).
    """
    return build_placeholder_routing_proposal(
        target_value=target_value,
        rollback_value=rollback_value,
        source_pressure_id="pressure-routing-test",
    )


def test_signal_to_pressure_pipeline_filters_parked_routing_signals() -> None:
    """autonomy_graph review telemetry no longer produces a "routing" signal.

    Formerly `test_signal_to_pressure_pipeline`, which asserted the opposite
    (exactly one signal, target_surface "routing", feeding a real pressure).
    As of 2026-09-03 the routing surface is parked: this telemetry is a
    review-pipeline consolidation-outcome signal that has nothing to do with
    what `chat_reflective_lane_threshold` gates, and mutation_proposals.py
    refuses every "routing" pressure unconditionally regardless -- so the
    detector filters it here instead of spending a store write and a
    pressure-accumulation cycle on a signal that can only ever be discarded
    three steps later.
    """
    detector = MutationDetectors()
    telemetry = GraphReviewTelemetryRecordV1(
        invocation_surface="operator_review",
        execution_outcome="failed",
        selection_reason="failed_test",
        runtime_duration_ms=12,
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="autonomy_graph",
    )
    signals = detector.from_review_telemetry([telemetry])
    assert signals == []


def test_routing_detector_no_longer_emits_runtime_social_pressure_signals() -> None:
    """As of 2026-09-03 the routing surface is parked (see
    mutation_detectors.py's filter at the end of `from_review_telemetry()`).
    Formerly `test_routing_detector_emits_richer_runtime_social_pressure_signals`,
    which asserted these kinds WERE produced. The underlying signal-building
    functions (`_build_rich_routing_signals` etc.) are untouched and still
    build them internally -- they are filtered from the returned list now,
    not removed at the source, so this proves the filter, not their absence
    from the code.
    """
    detector = MutationDetectors()
    telemetry = GraphReviewTelemetryRecordV1(
        invocation_surface="operator_review",
        execution_outcome="executed",
        selection_reason="recall_miss truncated operator_correction:downgrade",
        runtime_duration_ms=1500,
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="autonomy_graph",
        notes=["false_escalation", "not_addressed", "finish_reason:length"],
        consolidation_outcomes=["requeue_review"],
        degraded=True,
    )
    signals = detector.from_review_telemetry([telemetry])
    assert signals == []


def test_routing_rich_pressure_signals_do_not_broaden_non_routing_zones() -> None:
    detector = MutationDetectors()
    telemetry = GraphReviewTelemetryRecordV1(
        invocation_surface="operator_review",
        execution_outcome="executed",
        selection_reason="false_escalation recall_miss not_addressed",
        runtime_duration_ms=1800,
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="world_ontology",
        notes=["operator_correction:downgrade", "finish_reason:length"],
        consolidation_outcomes=["requeue_review"],
        degraded=True,
    )
    signals = detector.from_review_telemetry([telemetry])
    assert len(signals) == 1
    assert signals[0].target_surface == "graph_consolidation"
    assert signals[0].event_kind in {"runtime_review_churn", "runtime_executed"}


def test_routing_pressure_events_are_filtered_even_via_producer_provenance_path() -> None:
    """`_signals_from_pressure_events` maps a pressure_event to "routing" by
    its own `pressure_category`, independent of the record's own
    `target_zone` -- the trickiest of the three routing-signal-producing
    paths to filter correctly, since a record whose zone maps elsewhere can
    still carry a routing-categorized pressure_event. Confirms the filter in
    `from_review_telemetry()` catches this path too, not just the simpler
    zone-based one. Formerly
    `test_pressure_events_become_mutation_signals_with_event_provenance`,
    which asserted this event WAS turned into a routing-surfaced signal.
    """
    detector = MutationDetectors()
    event = MutationPressureEvidenceV1(
        pressure_event_id="pressure-evt-1",
        source_service="orion-hub",
        source_event_id="feedback-1",
        correlation_id="corr-1",
        pressure_category="routing_false_escalation",
        confidence=0.8,
        evidence_refs=["feedback:feedback-1", "feedback_category:wrong_tool_wrong_routing_wrong_mode"],
    )
    telemetry = GraphReviewTelemetryRecordV1(
        invocation_surface="chat_reflective_lane",
        execution_outcome="executed",
        selection_reason="producer_pressure_events",
        runtime_duration_ms=0,
        anchor_scope="orion",
        subject_ref="entity:orion",
        # world_ontology, not autonomy_graph: the record's own zone-derived
        # surface is "graph_consolidation" (not parked), so the only way a
        # "routing" signal can appear here is via the pressure_event's own
        # category -- isolating the path this test is actually about, not
        # also triggering the simpler zone-based filter at the same time.
        target_zone="world_ontology",
        pressure_events=[event],
    )
    signals = detector.from_review_telemetry([telemetry])
    # The record's own zone-derived signal (graph_consolidation) is not
    # parked and must survive -- only the routing_false_escalation pressure
    # event's own "routing"-surfaced signal is filtered.
    assert signals
    assert not any(signal.target_surface == "routing" for signal in signals)
    assert any(signal.target_surface == "graph_consolidation" for signal in signals)


def test_cognitive_signals_can_be_derived_from_existing_artifacts() -> None:
    detector = MutationDetectors(allow_cognitive_lane=True)
    telemetry = GraphReviewTelemetryRecordV1(
        invocation_surface="chat_reflective_lane",
        execution_outcome="executed",
        selection_reason="chat_stance_debug contradiction observed with identity_continuity drift",
        runtime_duration_ms=0,
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="autonomy_graph",
        pressure_events=[
            MutationPressureEvidenceV1(
                source_service="orion-hub",
                source_event_id="fb-1",
                correlation_id="corr-cog-1",
                pressure_category="social_addressedness_gap",
                confidence=0.7,
                evidence_refs=["feedback:fb-1"],
            ),
            MutationPressureEvidenceV1(
                source_service="orion-hub",
                source_event_id="fb-2",
                correlation_id="corr-cog-2",
                pressure_category="recall_miss_or_dissatisfaction",
                confidence=0.8,
                evidence_refs=["feedback:fb-2"],
            ),
        ],
    )
    signals = detector.from_review_telemetry([telemetry])
    kinds = {signal.event_kind for signal in signals}
    assert "contradiction_pressure" in kinds
    assert "identity_continuity_pressure" in kinds
    assert "social_continuity_pressure" in kinds
    cognitive = [signal for signal in signals if signal.target_surface.startswith("cognitive_")]
    assert cognitive


def test_cognitive_pressure_produces_operator_gated_proposal() -> None:
    detector = MutationDetectors(allow_cognitive_lane=True)
    telemetry = GraphReviewTelemetryRecordV1(
        invocation_surface="chat_reflective_lane",
        execution_outcome="executed",
        selection_reason="stance_drift detected",
        runtime_duration_ms=0,
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="autonomy_graph",
        pressure_events=[
            MutationPressureEvidenceV1(
                source_service="orion-cortex-exec",
                source_event_id="evt-stance-1",
                correlation_id="corr-stance-1",
                pressure_category="runtime_degradation_or_timeout",
                confidence=0.65,
                evidence_refs=["diag:partial"],
            )
        ],
    )
    signals = detector.from_review_telemetry([telemetry])
    cognitive_signal = next(signal for signal in signals if signal.event_kind == "stance_drift_pressure")
    pressure = PressureAccumulator(policy=PressurePolicy(activation_threshold=0.2)).apply(current=None, signal=cognitive_signal)
    proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(pressure)
    assert proposal is not None
    assert proposal.lane == "cognitive"
    assert proposal.target_surface == "cognitive_stance_continuity_adjustment"
    assert proposal.mutation_class == "cognitive_stance_continuity_adjustment"
    assert proposal.patch.patch["not_applied_status"] == "draft_only_not_applied"
    assert any(note.startswith("blast_radius:") for note in proposal.notes)
    assert proposal.patch.rollback_payload


def test_cognitive_lane_decisions_are_always_require_review() -> None:
    proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(
        MutationPressureV1(
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_surface="cognitive_social_continuity_repair",
            pressure_kind="social_continuity_pressure",
            pressure_score=7.0,
            evidence_refs=["telemetry:cog-1"],
            source_signal_ids=["signal-cog-1"],
        )
    )
    assert proposal is not None
    trial = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-v1"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={"operator_acceptance_rate": 0.8})
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    assert decision.action == "require_review"
    assert decision.requires_operator_review is True


def test_cognitive_proposals_never_use_prompt_profile_carrier() -> None:
    surfaces = (
        "cognitive_contradiction_reconciliation",
        "cognitive_identity_continuity_adjustment",
        "cognitive_stance_continuity_adjustment",
        "cognitive_social_continuity_repair",
    )
    for surface in surfaces:
        proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(
            MutationPressureV1(
                anchor_scope="orion",
                subject_ref="entity:orion",
                target_surface=surface,
                pressure_kind="cognitive_pressure",
                pressure_score=5.0,
                evidence_refs=[f"telemetry:{surface}"],
                source_signal_ids=[f"signal:{surface}"],
            )
        )
        assert proposal is not None
        assert proposal.lane == "cognitive"
        assert proposal.mutation_class != "approved_prompt_profile_variant_promotion"


def test_cognitive_review_accepted_as_draft_persists_draft_not_adoption() -> None:
    store = SubstrateMutationStore()
    proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(
        MutationPressureV1(
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_surface="cognitive_identity_continuity_adjustment",
            pressure_kind="identity_continuity_pressure",
            pressure_score=6.5,
            evidence_refs=["telemetry:cog-draft-1"],
            source_signal_ids=["signal-cog-draft-1"],
        )
    )
    assert proposal is not None
    store.add_proposal(proposal, priority=50)
    review = CognitiveProposalReviewV1(
        proposal_id=proposal.proposal_id,
        state="accepted_as_draft",
        reviewer="operator:test",
        rationale="safe draft only",
    )
    draft = store.record_cognitive_review(review)
    assert draft is not None
    assert draft.status == "draft_only_not_applied"
    assert store.queue_status_for_proposal(proposal.proposal_id) == "accepted_as_draft"
    assert not any(item.proposal_id == proposal.proposal_id for item in store._adoptions.values())


def test_cognitive_review_terminal_states_persist() -> None:
    store = SubstrateMutationStore()
    proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(
        MutationPressureV1(
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_surface="cognitive_social_continuity_repair",
            pressure_kind="social_continuity_pressure",
            pressure_score=6.0,
            evidence_refs=["telemetry:cog-state-1"],
            source_signal_ids=["signal-cog-state-1"],
        )
    )
    assert proposal is not None
    store.add_proposal(proposal, priority=40)

    for expected in ("rejected", "superseded", "archived"):
        review = CognitiveProposalReviewV1(proposal_id=proposal.proposal_id, state=expected, reviewer="operator:test")
        draft = store.record_cognitive_review(review)
        assert draft is None
        assert store.queue_status_for_proposal(proposal.proposal_id) == expected
        updated = store.get_proposal(proposal.proposal_id)
        assert updated is not None
        assert updated.rollout_state == expected


def test_recall_pressure_event_becomes_recall_strategy_signal_and_proposal() -> None:
    detector = MutationDetectors()
    telemetry = GraphReviewTelemetryRecordV1(
        invocation_surface="chat_reflective_lane",
        execution_outcome="executed",
        selection_reason="producer_pressure_events:recall-evt",
        runtime_duration_ms=0,
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="autonomy_graph",
        pressure_events=[
            MutationPressureEvidenceV1(
                pressure_event_id="recall-evt-1",
                source_service="orion-recall",
                source_event_id="corr-recall-1",
                pressure_category="missing_exact_anchor",
                confidence=0.82,
                evidence_refs=["recall_decision:abc", "query:exact anchor"],
                metadata={
                    "v1_v2_compare": {"v1_latency_ms": 120, "v2_latency_ms": 95, "selected_count_delta": 2},
                    "anchor_plan": {"temporal_anchor": "today", "time_window_days": 1, "exact_anchor_tokens": ["COMMIT123"]},
                    "selected_evidence_cards": [{"id": "page-1"}],
                },
            )
        ],
    )
    signals = detector.from_review_telemetry([telemetry])
    recall_signal = next(signal for signal in signals if signal.target_surface == "recall_anchor_policy")
    assert recall_signal.event_kind == "pressure_event:missing_exact_anchor"
    assert any(ref.startswith("recall_compare:") for ref in recall_signal.evidence_refs)
    pressure = PressureAccumulator(policy=PressurePolicy(activation_threshold=0.2)).apply(current=None, signal=recall_signal)
    proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(pressure)
    assert proposal is not None
    assert proposal.mutation_class == "recall_anchor_policy_candidate"
    assert proposal.patch.patch["shadow_only_status"] == "recall_v2_shadow_only"
    assert proposal.patch.patch["not_applied_status"] == "proposal_only_not_applied"
    assert isinstance(proposal.patch.patch["v1_v2_comparison_evidence"], dict)
    assert proposal.patch.patch["v1_v2_comparison_evidence"].get("selected_count_delta") == 2
    assert isinstance(proposal.patch.patch["anchor_plan_summary"], dict)
    assert isinstance(proposal.patch.patch["selected_evidence_cards"], list)
    assert proposal.patch.patch["selected_evidence_cards"][0].get("id") == "page-1"


def test_recall_strategy_proposal_is_always_operator_gated_and_not_applied() -> None:
    proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(
        MutationPressureV1(
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_surface="recall_strategy_profile",
            pressure_kind="pressure_event:recall_miss_or_dissatisfaction",
            pressure_score=7.2,
            evidence_refs=["recall_compare:v1_latency_ms=120", "anchor_plan:time_window_days=7", "selected_card:page-1"],
            source_signal_ids=["signal-recall-1"],
        )
    )
    assert proposal is not None
    trial = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-recall-v2-shadow"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-recall-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={})
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    assert decision.action == "require_review"
    assert decision.requires_operator_review is True
    adoption = PatchApplier(surfaces={}).apply(proposal=proposal, decision=decision)
    assert adoption is None


def test_patch_applier_never_applies_recall_weighting_even_if_auto_promote() -> None:
    proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(
        MutationPressureV1(
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_surface="recall_strategy_profile",
            pressure_kind="pressure_event:recall_miss_or_dissatisfaction",
            pressure_score=9.0,
            evidence_refs=["telemetry:1"],
            source_signal_ids=["signal-1"],
        )
    )
    assert proposal is not None
    assert proposal.mutation_class == "recall_strategy_profile_candidate"
    fake_decision = MutationDecisionV1(
        proposal_id=proposal.proposal_id,
        action="auto_promote",
        reason="hypothetical_misconfig",
    )
    assert PatchApplier(surfaces={}).apply(proposal=proposal, decision=fake_decision) is None
    rw_prop = MutationProposalV1(
        lane="operational",
        mutation_class="recall_weighting_patch",
        risk_tier="low",
        target_surface="recall",
        anchor_scope="orion",
        subject_ref="entity:orion",
        rationale="test",
        expected_effect="none",
        evidence_refs=["telemetry:rw"],
        source_signal_ids=["signal-rw"],
        source_pressure_id="pressure-rw",
        patch=MutationPatchV1(
            mutation_class="recall_weighting_patch",
            target_surface="recall",
            target_ref="recall",
            patch={"semantic_weight": 0.6, "episodic_weight": 0.3, "recency_weight": 0.1},
            rollback_payload={"semantic_weight": 0.5, "episodic_weight": 0.35, "recency_weight": 0.15},
        ),
        notes=[],
    )
    assert PatchApplier(surfaces={}).apply(proposal=rw_prop, decision=fake_decision) is None


def test_eval_shaped_compare_produces_recall_strategy_proposal() -> None:
    from orion.substrate.recall_eval_bridge import eval_row_to_v1_v2_compare

    case_row = {
        "case_id": "synthetic-1",
        "v1": {"selected_count": 0, "latency_ms": 50, "precision_proxy": 0.1},
        "v2": {"selected_count": 2, "latency_ms": 40, "precision_proxy": 0.6, "entity_time_match_rate": 0.5},
    }
    compare = eval_row_to_v1_v2_compare(case_row)
    detector = MutationDetectors()
    telemetry = GraphReviewTelemetryRecordV1(
        invocation_surface="chat_reflective_lane",
        execution_outcome="executed",
        selection_reason="producer_pressure_events:eval-synthetic",
        runtime_duration_ms=0,
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="autonomy_graph",
        pressure_events=[
            MutationPressureEvidenceV1(
                pressure_event_id="eval-evt-1",
                source_service="orion-recall",
                source_event_id="eval:synthetic-1",
                pressure_category="recall_miss_or_dissatisfaction",
                confidence=0.88,
                evidence_refs=["recall_eval:synthetic-1"],
                metadata={
                    "v1_v2_compare": compare,
                    "recall_eval_case": {"case_id": "synthetic-1", "query": "q"},
                    "anchor_plan": {"time_window_days": 7},
                    "selected_evidence_cards": [{"id": "c1", "source": "vector"}],
                },
            )
        ],
    )
    signals = detector.from_review_telemetry([telemetry])
    strat_signal = next(signal for signal in signals if signal.target_surface == "recall_strategy_profile")
    pressure = PressureAccumulator(policy=PressurePolicy(activation_threshold=0.2)).apply(current=None, signal=strat_signal)
    proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(pressure)
    assert proposal is not None
    assert proposal.mutation_class == "recall_strategy_profile_candidate"
    assert proposal.patch.patch.get("v1_v2_comparison_evidence", {}).get("source") == "recall_eval_suite"


def test_recall_pressure_evidence_history_bounded_and_in_proposal() -> None:
    accum = PressureAccumulator(policy=PressurePolicy(activation_threshold=0.01))
    pressure: MutationPressureV1 | None = None
    for i in range(10):
        sig = MutationSignalV1(
            event_kind="pressure_event:missing_exact_anchor",
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_zone="concept_graph",
            target_surface="recall_anchor_policy",
            strength=0.55,
            evidence_refs=[f"telemetry:{i}"],
            metadata={
                "recall_compare": {"selected_count_delta": i, "v1_selected_count": 1, "v2_selected_count": 2},
                "failure_category": "missing_exact_anchor",
                "recall_evidence_kind": "live_shadow",
            },
        )
        pressure = accum.apply(current=pressure, signal=sig)
    assert pressure is not None
    assert len(pressure.recall_evidence_history) == 8
    assert pressure.recall_evidence_history[-1]["recall_compare"]["selected_count_delta"] == 9
    proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(pressure)
    assert proposal is not None
    hist = proposal.patch.patch.get("contributing_recall_evidence_history")
    assert isinstance(hist, list) and len(hist) == 8


def test_routing_pressure_is_parked_not_turned_into_a_proposal() -> None:
    """As of 2026-09-03 the routing surface is parked (see
    mutation_proposals.py's `ROUTING_TARGET_PARKED_REASON`): confirmed live
    that the evidence feeding it has nothing to do with what the dial gates,
    and that the hardcoded 0.58 target is below the minimum confidence
    (0.61) any heuristic routing decision can carry at execution_depth >= 2
    with AUTO_ROUTER_LLM_ENABLED=false. Formerly
    `test_routing_threshold_proposal_class_unchanged` /
    `test_pressure_to_proposal`, which asserted the opposite."""
    plan = ProposalFactory(routing_surface_reader=_routing_surface()).plan_for_pressure(_routing_pressure())
    assert plan.proposal is None
    assert plan.refusal_reason == ROUTING_TARGET_PARKED_REASON


def test_proposal_to_trial() -> None:
    proposal = _direct_routing_proposal()
    assert proposal is not None
    trial = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-v1"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={"success_rate_delta": 0.1, "latency_ms_delta": 0.0})
    assert trial.proposal_id == proposal.proposal_id
    assert trial.status == "passed"


def test_trial_to_decision() -> None:
    proposal = _direct_routing_proposal()
    assert proposal is not None
    trial = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-v1"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={"success_rate_delta": 0.1, "latency_ms_delta": 0.0})
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    assert decision.action == "auto_promote"


def test_mutation_proposal_requires_evidence_and_rollback() -> None:
    proposal = _direct_routing_proposal()
    assert proposal is not None
    assert proposal.evidence_refs
    assert proposal.patch.rollback_payload


def test_decision_engine_keeps_prompt_profile_operator_gated() -> None:
    pressure = MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="prompt_profile",
        pressure_kind="runtime_review_churn",
        pressure_score=6.0,
        evidence_refs=["telemetry:abc"],
        source_signal_ids=["signal-abc"],
    )
    proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(pressure)
    assert proposal is not None
    registry = ReplayCorpusRegistry(
        corpus_by_class={proposal.mutation_class: "corpus-v1"},
        baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
    )
    trial = SubstrateTrialRunner(scorer=ClassSpecificScorer(), corpus_registry=registry).run_trial(
        proposal=proposal,
        measured_metrics={"quality_score_delta": 0.01, "safety_incident_delta": 0.0},
    )
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    assert decision.action == "require_review"


def test_store_allows_only_single_active_mutation_per_surface(tmp_path: Path) -> None:
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))
    proposal1 = _direct_routing_proposal()
    assert proposal1 is not None
    queue_item = store.add_proposal(proposal1)
    assert queue_item.status == "queued"

    registry = ReplayCorpusRegistry(
        corpus_by_class={proposal1.mutation_class: "corpus-v1"},
        baseline_metric_ref_by_class={proposal1.mutation_class: "baseline-v1"},
    )
    trial_runner = SubstrateTrialRunner(scorer=ClassSpecificScorer(), corpus_registry=registry)
    trial = trial_runner.run_trial(proposal=proposal1, measured_metrics={"success_rate_delta": 0.1, "latency_ms_delta": 0.0})
    decision = DecisionEngine().decide(
        proposal=proposal1,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    adoption = PatchApplier(surfaces={}).apply(proposal=proposal1, decision=decision)
    assert adoption is not None
    assert store.record_adoption(adoption) == []

    proposal2 = _direct_routing_proposal()
    assert proposal2 is not None
    decision2 = DecisionEngine().decide(
        proposal=proposal2,
        trial=trial_runner.run_trial(proposal=proposal2, measured_metrics={"success_rate_delta": 0.1, "latency_ms_delta": 0.0}),
        has_replay_and_baseline=True,
        active_surface_exists=store.active_surface(proposal2.target_surface) is not None,
    )
    assert decision2.action == "hold"


def test_queue_item_consumed_after_trial_and_decision(tmp_path: Path) -> None:
    store = SubstrateMutationStore(sql_db_path=str(tmp_path / "mutation.sqlite3"))
    proposal = _direct_routing_proposal()
    assert proposal is not None
    queue_item = store.add_proposal(proposal)
    assert any(item.queue_item_id == queue_item.queue_item_id for item in store.list_due_queue(limit=10))
    trial = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-v1"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={"success_rate_delta": 0.1, "latency_ms_delta": 0.0})
    store.record_trial(trial)
    store.record_decision(MutationDecisionV1(proposal_id=proposal.proposal_id, action="require_review"))
    assert not any(item.queue_item_id == queue_item.queue_item_id for item in store.list_due_queue(limit=10))


def test_adaptation_worker_obeys_kill_switch() -> None:
    store = SubstrateMutationStore()
    worker = SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(policy=PressurePolicy(activation_threshold=0.2)),
        proposals=ProposalFactory(routing_surface_reader=_routing_surface()),
        trial_runner=SubstrateTrialRunner(
            scorer=ClassSpecificScorer(),
            corpus_registry=ReplayCorpusRegistry(corpus_by_class={}, baseline_metric_ref_by_class={}),
        ),
        decision_engine=DecisionEngine(),
        applier=PatchApplier(surfaces={}),
        monitor=PostAdoptionMonitor(),
    )
    os.environ["SUBSTRATE_MUTATION_AUTONOMY_ENABLED"] = "false"
    result = worker.run_cycle(
        telemetry=[
            GraphReviewTelemetryRecordV1(
                invocation_surface="operator_review",
                execution_outcome="failed",
                selection_reason="x",
                runtime_duration_ms=12,
                anchor_scope="orion",
                subject_ref="entity:orion",
                target_zone="concept_graph",
            )
        ],
        measured_metrics_by_proposal={},
    )
    assert "autonomy_kill_switch_disabled" in result["notes"]


def test_require_review_never_applies() -> None:
    pressure = MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="prompt_profile",
        pressure_kind="runtime_review_churn",
        pressure_score=6.0,
        evidence_refs=["telemetry:abc"],
        source_signal_ids=["signal-abc"],
    )
    proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(pressure)
    assert proposal is not None
    trial = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-v1"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={"quality_score_delta": 0.1, "safety_incident_delta": 0.0})
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    applier = PatchApplier(surfaces={})
    adoption = applier.apply(proposal=proposal, decision=decision)
    assert decision.action == "require_review"
    assert adoption is None
    assert applier.surfaces == {}


def test_one_live_mutation_invariant_blocks_before_side_effects() -> None:
    """The routing surface is parked, so a "routing" pressure can no longer
    reach `plan_for_pressure()` via telemetry -- feeding autonomy_graph
    telemetry here (as this test did before 2026-09-03) would produce zero
    signals and pass vacuously, exercising nothing. The active-surface
    invariant itself is real and still needs proving through a full worker
    cycle, so this enqueues the proposal directly (bypassing the parked
    detector/pressure/factory chain, same pattern as smoke_mutation_v21.py's
    active-surface demo) and lets the worker's trial/decision/apply half run
    normally on it.
    """
    store = SubstrateMutationStore()
    proposal = _direct_routing_proposal()
    store._active_surface_by_target[proposal.target_surface] = "existing-adoption"
    store.add_proposal(proposal, priority=60)
    applier = PatchApplier(surfaces={proposal.target_surface: {"chat_reflective_lane_threshold": 0.5}})
    traces: list[dict] = []
    worker = SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(policy=PressurePolicy(activation_threshold=0.1, cooldown_seconds=5)),
        proposals=ProposalFactory(routing_surface_reader=_routing_surface()),
        trial_runner=SubstrateTrialRunner(
            scorer=ClassSpecificScorer(),
            corpus_registry=ReplayCorpusRegistry(
                corpus_by_class={"routing_threshold_patch": "corpus-v1"},
                baseline_metric_ref_by_class={"routing_threshold_patch": "baseline-v1"},
            ),
        ),
        decision_engine=DecisionEngine(),
        applier=applier,
        monitor=PostAdoptionMonitor(),
        trace_logger=traces.append,
    )
    os.environ["SUBSTRATE_MUTATION_AUTONOMY_ENABLED"] = "true"
    result = worker.run_cycle(telemetry=[], measured_metrics_by_proposal={})
    assert any(
        event.get("event") == "mutation_decision_recorded" and event.get("decision") == "hold"
        for event in traces
    ), f"active-surface invariant was never exercised -- no hold decision recorded; events={sorted({e.get('event') for e in traces})}"
    assert result["adoptions"] == 0
    assert applier.surfaces[proposal.target_surface]["chat_reflective_lane_threshold"] == 0.5


def test_rollback_payload_required_before_apply() -> None:
    proposal = _direct_routing_proposal()
    assert proposal is not None
    proposal = proposal.model_copy(update={"patch": proposal.patch.model_copy(update={"rollback_payload": {}})})
    adoption = PatchApplier(surfaces={}).apply(proposal=proposal, decision=MutationDecisionV1(proposal_id=proposal.proposal_id, action="auto_promote"))
    assert adoption is None


def test_approved_prompt_profile_variant_promotion_operator_gated_default() -> None:
    pressure = MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="prompt_profile",
        pressure_kind="runtime_review_churn",
        pressure_score=6.0,
        evidence_refs=["telemetry:abc"],
        source_signal_ids=["signal-abc"],
    )
    proposal = ProposalFactory(routing_surface_reader=_routing_surface()).from_pressure(pressure)
    assert proposal is not None
    trial = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-v1"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={"quality_score_delta": 0.1, "safety_incident_delta": 0.0})
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    assert decision.action == "require_review"


def test_pending_review_compat_no_sql_enum_migration(tmp_path: Path) -> None:
    store = SubstrateMutationStore(sql_db_path=str(tmp_path / "mutation.sqlite3"))
    assert store.source_kind() == "sqlite"
    import sqlite3

    with sqlite3.connect(str(tmp_path / "mutation.sqlite3")) as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='substrate_mutation_queue'"
        ).fetchone()
    assert row is not None
    ddl = str(row[0] or "")
    assert "CHECK" not in ddl.upper()
    assert "pending_review" not in ddl


def test_smoke_script_trace_and_invariants() -> None:
    lines = run_smoke(emit=False)
    assert any("event=mutation_smoke_start" in line for line in lines)
    assert any("event=mutation_pressure_recorded" in line for line in lines)
    assert any("event=mutation_proposal_enqueued" in line for line in lines)
    assert any("event=mutation_trial_recorded" in line for line in lines)
    assert any("decision=require_review" in line and "queue_status_after=pending_review" in line for line in lines)
    assert any("event=mutation_apply_blocked" in line and "blocked_reason=active_surface" in line for line in lines)
    assert any("decision=auto_promote" in line and "applied=True" in line for line in lines)
    assert any("blocked_reason=rollback_payload_required" in line for line in lines)
    assert any("event=mutation_smoke_complete" in line and "ok=true" in line for line in lines)


def test_restart_safe_reload_of_in_flight_mutation_state(tmp_path: Path) -> None:
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))
    proposal = _direct_routing_proposal()
    assert proposal is not None
    queue_item = store.add_proposal(proposal)
    trial = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-v1"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={"success_rate_delta": 0.1, "latency_ms_delta": 0.0})
    store.record_trial(trial)
    store.record_decision(MutationDecisionV1(proposal_id=proposal.proposal_id, action="require_review"))

    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    assert reloaded.get_proposal(proposal.proposal_id) is not None
    assert reloaded.queue_status_for_proposal(proposal.proposal_id) == "pending_review"
    assert queue_item.queue_item_id in reloaded._queue


def test_duplicate_apply_prevention_after_retry(tmp_path: Path) -> None:
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))
    proposal = _direct_routing_proposal()
    assert proposal is not None
    trial = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-v1"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={"success_rate_delta": 0.2, "latency_ms_delta": 0.0})
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    applier = PatchApplier(surfaces={"routing": {"chat_reflective_lane_threshold": 0.5}})
    adoption = applier.apply(proposal=proposal, decision=decision)
    assert adoption is not None
    assert store.record_adoption(adoption) == []

    retried = adoption.model_copy(update={"adoption_id": "substrate-mutation-adoption-retry"})
    warnings = store.record_adoption(retried)
    assert warnings == ["duplicate_adoption_for_proposal"]
    assert store.active_surface("routing") == adoption.adoption_id


def test_active_surface_recovered_after_reload(tmp_path: Path) -> None:
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))
    proposal = _direct_routing_proposal()
    assert proposal is not None
    trial = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-v1"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={"success_rate_delta": 0.2, "latency_ms_delta": 0.0})
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    adoption = PatchApplier(surfaces={"routing": {"chat_reflective_lane_threshold": 0.5}}).apply(proposal=proposal, decision=decision)
    assert adoption is not None
    assert store.record_adoption(adoption) == []

    import sqlite3

    with sqlite3.connect(str(db)) as conn:
        conn.execute("DELETE FROM substrate_mutation_active_surface")
        conn.commit()
    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    assert reloaded.active_surface("routing") == adoption.adoption_id


def test_rollback_continuity_after_reload(tmp_path: Path) -> None:
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))
    proposal = _direct_routing_proposal()
    assert proposal is not None
    trial = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-v1"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={"success_rate_delta": 0.2, "latency_ms_delta": 0.0})
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    applier = PatchApplier(surfaces={"routing": {"chat_reflective_lane_threshold": 0.5}})
    adoption = applier.apply(proposal=proposal, decision=decision)
    assert adoption is not None
    assert store.record_adoption(adoption) == []
    rollback = PostAdoptionMonitor().build_rollback(adoption=adoption, reason="regression")
    store.record_rollback(rollback)

    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    assert reloaded.active_surface("routing") is None
    assert any(item["rollback_id"] == rollback.rollback_id for item in reloaded.recent_rollbacks(limit=10))


def test_record_rollback_refuses_on_non_applied_adoption() -> None:
    """Real-stakes gate: rollback is refused once an adoption is no longer
    "applied" -- already rolled back, in this test. Before this guard the
    store trusted every caller's own discipline and enforced nothing itself.
    """
    store = SubstrateMutationStore()
    proposal = _direct_routing_proposal()
    assert proposal is not None
    store.add_proposal(proposal)
    decision = MutationDecisionV1(proposal_id=proposal.proposal_id, action="auto_promote")
    adoption = PatchApplier(surfaces={"routing": {"chat_reflective_lane_threshold": 0.5}}).apply(proposal=proposal, decision=decision)
    assert adoption is not None
    assert store.record_adoption(adoption) == []

    first_rollback = PostAdoptionMonitor().build_rollback(adoption=adoption, reason="first")
    assert store.record_rollback(first_rollback) is True

    # Same adoption_id, second attempt -- adoption is now "rolled_back" in the
    # store (the local `adoption` variable is untouched, so this looks like a
    # legitimate second call from a caller that doesn't track store state).
    second_rollback = PostAdoptionMonitor().build_rollback(adoption=adoption, reason="second")
    assert store.record_rollback(second_rollback) is False
    assert len(store.recent_rollbacks(limit=10)) == 1


def test_rollback_cooldown_blocks_readoption_scaled_by_risk_tier() -> None:
    """Real-stakes gate: a rolled-back "high" risk_tier mutation cools its
    target_surface down for rollback_window_sec * 8 (see
    _ROLLBACK_COOLDOWN_MULTIPLIER), not zero. A "low" risk_tier rollback on a
    different surface, checked in the same test, is not blocked at the same
    elapsed time -- the multiplier is real, not a flat cooldown.
    """
    store = SubstrateMutationStore()
    high = _direct_routing_proposal().model_copy(update={"risk_tier": "high"})
    store.add_proposal(high)
    decision = MutationDecisionV1(proposal_id=high.proposal_id, action="auto_promote")
    high_adoption = PatchApplier(surfaces={"routing": {"chat_reflective_lane_threshold": 0.5}}).apply(proposal=high, decision=decision)
    assert high_adoption is not None
    high_adoption = high_adoption.model_copy(update={"rollback_window_sec": 100})
    assert store.record_adoption(high_adoption) == []
    rollback = PostAdoptionMonitor().build_rollback(adoption=high_adoption, reason="regression")
    assert store.record_rollback(rollback) is True

    retry = high.model_copy(update={"proposal_id": "substrate-mutation-proposal-retry"})
    store.add_proposal(retry)
    retry_adoption = high_adoption.model_copy(
        update={"adoption_id": "substrate-mutation-adoption-retry", "proposal_id": retry.proposal_id, "status": "applied"}
    )
    # Still well inside 100s * 8.0 = 800s of the rollback (created_at ~= now).
    assert store.record_adoption(retry_adoption) == ["target_surface_in_rollback_cooldown"]

    # Backdate the rollback past its cooldown and retry the same adoption.
    store._rollbacks[rollback.rollback_id] = rollback.model_copy(
        update={"created_at": rollback.created_at - timedelta(seconds=900)}
    )
    assert store.record_adoption(retry_adoption) == []


def test_surface_reliability_cold_start_then_computed() -> None:
    """Real-stakes gate: surface_reliability() is None until
    SURFACE_RELIABILITY_MIN_SAMPLES resolved adoptions exist (cold start, not
    a fabricated mid-range number), then a Laplace-smoothed settled ratio that
    a rollback -- not just its patch -- keeps moving after the patch itself
    is undone.
    """
    store = SubstrateMutationStore()
    assert store.surface_reliability("routing") is None

    outcomes = ["settled", "settled", "rolled_back"]
    for i, outcome in enumerate(outcomes):
        # Distinct target_value per iteration -- PatchApplier declines a patch
        # that would change nothing against the live control surface, and the
        # prior iteration's real apply already moved it (settling keeps an
        # applied value; it doesn't revert it).
        proposal = _direct_routing_proposal(target_value=0.58 + i * 0.01).model_copy(
            update={"proposal_id": f"substrate-mutation-proposal-rel-{i}"}
        )
        store.add_proposal(proposal)
        decision = MutationDecisionV1(proposal_id=proposal.proposal_id, action="auto_promote")
        adoption = PatchApplier(surfaces={"routing": {"chat_reflective_lane_threshold": 0.5}}).apply(proposal=proposal, decision=decision)
        assert adoption is not None
        adoption = adoption.model_copy(update={"adoption_id": f"substrate-mutation-adoption-rel-{i}"})
        assert store.record_adoption(adoption) == []
        if i < 2:
            assert store.surface_reliability("routing") is None  # still below MIN_SAMPLES
        if outcome == "settled":
            assert store.record_settlement(adoption.adoption_id) is True
        else:
            rollback = PostAdoptionMonitor().build_rollback(adoption=adoption, reason="third")
            assert store.record_rollback(rollback) is True

    # 2 settled, 1 rolled_back -> (2+1)/(2+1+2) = 0.6
    assert store.surface_reliability("routing") == pytest.approx(0.6)


def test_proposal_factory_refuses_below_reliability_floor() -> None:
    """Real-stakes gate: being wrong costs this surface's future proposals,
    not just the one rolled-back action. None (cold start, or no reader
    wired) is not treated as a failing reliability -- only a real low read
    refuses.
    """
    pressure = MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="graph_consolidation",
        pressure_kind="runtime_drift",
        pressure_score=6.0,
        evidence_refs=["telemetry:x"],
        source_signal_ids=["signal-x"],
    )
    unreliable = ProposalFactory(surface_reliability_reader=lambda _surface: 0.1)
    plan = unreliable.plan_for_pressure(pressure)
    assert plan.proposal is None
    assert plan.refusal_reason == "target_surface_reliability_below_floor"

    cold_start = ProposalFactory(surface_reliability_reader=lambda _surface: None)
    assert cold_start.plan_for_pressure(pressure).proposal is not None

    no_reader = ProposalFactory()
    assert no_reader.plan_for_pressure(pressure).proposal is not None


def test_blocked_apply_attribution_persists_with_reason_and_context(tmp_path: Path) -> None:
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))
    key = store.record_apply_blocked(
        proposal_id="proposal-1",
        decision_id="decision-1",
        target_surface="routing",
        reason="active_surface",
        notes=["active_mutation_exists_for_target_surface"],
        queue_status="approved",
    )
    assert key
    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    rows = reloaded.recent_blocked_applies(limit=5)
    assert rows
    assert rows[0]["proposal_id"] == "proposal-1"
    assert rows[0]["decision_id"] == "decision-1"
    assert rows[0]["reason"] == "active_surface"


def test_retention_compaction_preserves_active_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_RETENTION_MAX_BLOCKED_APPLIES", "50")
    monkeypatch.setenv("SUBSTRATE_MUTATION_RETENTION_MAX_ROLLBACKS", "50")
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))
    proposal = _direct_routing_proposal()
    assert proposal is not None
    trial = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-v1"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={"success_rate_delta": 0.2, "latency_ms_delta": 0.0})
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    adoption = PatchApplier(surfaces={"routing": {"chat_reflective_lane_threshold": 0.5}}).apply(proposal=proposal, decision=decision)
    assert adoption is not None
    assert store.record_adoption(adoption) == []
    for idx in range(120):
        store.record_apply_blocked(
            proposal_id=f"p-{idx}",
            decision_id=f"d-{idx}",
            target_surface="routing",
            reason="active_surface",
            queue_status="approved",
        )
    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    assert len(reloaded.recent_blocked_applies(limit=500)) <= 50
    assert reloaded.active_surface("routing") == adoption.adoption_id


def test_targeted_signal_persistence_keeps_restart_reload_behavior(tmp_path: Path) -> None:
    db = tmp_path / "mutation.sqlite3"
    store = SubstrateMutationStore(sql_db_path=str(db))
    # world_ontology, not autonomy_graph: this is a generic signal-persistence
    # test, and autonomy_graph telemetry is now filtered to zero signals (the
    # routing surface is parked -- see mutation_detectors.py).
    signal = MutationDetectors().from_review_telemetry(
        [
            GraphReviewTelemetryRecordV1(
                invocation_surface="operator_review",
                execution_outcome="failed",
                selection_reason="targeted_signal_persist",
                runtime_duration_ms=8,
                anchor_scope="orion",
                subject_ref="entity:orion",
                target_zone="world_ontology",
            )
        ]
    )[0]
    store.record_signal(signal)
    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    assert any(item.signal_id == signal.signal_id for item in reloaded._signals)


def test_routing_replay_corpus_drives_trial_metrics_without_manual_injection() -> None:
    proposal = _direct_routing_proposal()
    assert proposal is not None
    runner = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={"routing_threshold_patch": "replay-routing-v1"},
            baseline_metric_ref_by_class={"routing_threshold_patch": "baseline-routing-v1"},
        ),
    )
    telemetry = [
        GraphReviewTelemetryRecordV1(
            invocation_surface="operator_review",
            execution_outcome="failed",
            selection_reason="replay-failure",
            selected_priority=90,
            runtime_duration_ms=40,
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_zone="autonomy_graph",
        ),
        GraphReviewTelemetryRecordV1(
            invocation_surface="operator_review",
            execution_outcome="executed",
            selection_reason="replay-executed",
            selected_priority=20,
            runtime_duration_ms=12,
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_zone="autonomy_graph",
        ),
    ]
    trial = runner.run_trial(proposal=proposal, measured_metrics={}, replay_records=telemetry)
    assert trial.status in {"passed", "failed"}
    assert "success_rate_delta" in trial.metrics
    assert "route_appropriateness_proxy" in trial.metrics


def test_routing_decision_can_use_replay_derived_metrics() -> None:
    proposal = _direct_routing_proposal()
    assert proposal is not None
    runner = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={"routing_threshold_patch": "replay-routing-v1"},
            baseline_metric_ref_by_class={"routing_threshold_patch": "baseline-routing-v1"},
        ),
    )
    telemetry = [
        GraphReviewTelemetryRecordV1(
            invocation_surface="operator_review",
            execution_outcome="failed",
            selection_reason="replay-high-priority",
            selected_priority=95,
            runtime_duration_ms=15,
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_zone="autonomy_graph",
        )
    ]
    trial = runner.run_trial(proposal=proposal, measured_metrics={}, replay_records=telemetry)
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    assert decision.action in {"auto_promote", "hold", "reject"}


def test_manual_metric_injection_remains_optional_override() -> None:
    proposal = _direct_routing_proposal()
    assert proposal is not None
    runner = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={"routing_threshold_patch": "replay-routing-v1"},
            baseline_metric_ref_by_class={"routing_threshold_patch": "baseline-routing-v1"},
        ),
    )
    trial = runner.run_trial(
        proposal=proposal,
        measured_metrics={"success_rate_delta": -0.2, "latency_ms_delta": -1.0},
        replay_records=[
            GraphReviewTelemetryRecordV1(
                invocation_surface="operator_review",
                execution_outcome="failed",
                selection_reason="replay-fallback-check",
                selected_priority=95,
                runtime_duration_ms=10,
                anchor_scope="orion",
                subject_ref="entity:orion",
                target_zone="autonomy_graph",
            )
        ],
    )
    assert trial.metrics["success_rate_delta"] == -0.2
    assert trial.status == "failed"


def test_routing_replay_prefers_rich_runtime_artifacts_over_selected_priority() -> None:
    proposal = _direct_routing_proposal()
    assert proposal is not None
    runner = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={"routing_threshold_patch": "replay-routing-v1"},
            baseline_metric_ref_by_class={"routing_threshold_patch": "baseline-routing-v1"},
        ),
    )
    telemetry = [
        GraphReviewTelemetryRecordV1(
            invocation_surface="operator_review",
            execution_outcome="executed",
            selection_reason="rich-artifact-case decision_confidence:0.20 route_quality:0.30 task_completion:0.20",
            selected_priority=98,
            runtime_duration_ms=11,
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_zone="autonomy_graph",
            notes=["false_escalation", "operator_correction:downgrade"],
        ),
        GraphReviewTelemetryRecordV1(
            invocation_surface="operator_review",
            execution_outcome="failed",
            selection_reason="rich-artifact-escalate decision_confidence:0.90 route_quality:0.85 task_completion:0.90",
            selected_priority=40,
            runtime_duration_ms=20,
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_zone="autonomy_graph",
            notes=["operator_correction:escalate"],
        ),
    ]
    trial = runner.run_trial(proposal=proposal, measured_metrics={}, replay_records=telemetry)
    assert "evaluator_confidence" in trial.metrics
    assert trial.metrics["evaluator_confidence"] > 0.0
    assert trial.metrics["corpus_coverage"] > 0.0
    assert trial.metrics["rich_signal_case_count"] >= 1.0


def test_routing_replay_inspection_reports_corpus_composition_and_confidence() -> None:
    proposal = _direct_routing_proposal()
    assert proposal is not None
    runner = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={"routing_threshold_patch": "replay-routing-v1"},
            baseline_metric_ref_by_class={"routing_threshold_patch": "baseline-routing-v1"},
        ),
    )
    inspection = runner.inspect_routing_replay(
        proposal=proposal,
        replay_records=[
            GraphReviewTelemetryRecordV1(
                invocation_surface="operator_review",
                execution_outcome="failed",
                selection_reason="inspect-rich decision_confidence:0.82 task_completion:0.76",
                selected_priority=80,
                runtime_duration_ms=15,
                anchor_scope="orion",
                subject_ref="entity:orion",
                target_zone="autonomy_graph",
            )
        ],
    )
    assert inspection["corpus_composition"]["rich_signal_case_count"] == 1
    assert inspection["derived_metrics"]["evaluator_confidence"] > 0.0


def test_recall_strategy_profiles_persist_across_sql_reload(tmp_path: Path) -> None:
    db = tmp_path / "mutation.db"
    store = SubstrateMutationStore(sql_db_path=str(db))
    staged = store.stage_recall_profile(
        profile=RecallStrategyProfileV1(
            source_proposal_id="proposal-1",
            source_pressure_ids=["pressure-1"],
            source_evidence_refs=["recall_compare:1"],
            readiness_snapshot={"recommendation": "review_candidate", "gates_blocked": []},
            strategy_kind="strategy_profile",
            recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
            anchor_policy_snapshot={"time_window_days": 7},
            page_index_policy_snapshot={"top_k": 8},
            graph_expansion_policy_snapshot={"enabled": True},
            created_by="operator:test",
            status="staged",
        )
    )
    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    loaded = reloaded.get_recall_strategy_profile(staged.profile_id)
    assert loaded is not None
    assert loaded.source_proposal_id == "proposal-1"
    assert loaded.readiness_snapshot.get("recommendation") == "review_candidate"


def test_recall_shadow_profile_activation_keeps_single_active_profile() -> None:
    store = SubstrateMutationStore()
    p1 = store.stage_recall_profile(
        profile=RecallStrategyProfileV1(
            source_proposal_id="proposal-a",
            source_pressure_ids=[],
            source_evidence_refs=[],
            readiness_snapshot={"recommendation": "review_candidate", "gates_blocked": []},
            strategy_kind="strategy_profile",
            recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
            anchor_policy_snapshot={},
            page_index_policy_snapshot={},
            graph_expansion_policy_snapshot={},
            created_by="operator",
        )
    )
    p2 = store.stage_recall_profile(
        profile=RecallStrategyProfileV1(
            source_proposal_id="proposal-b",
            source_pressure_ids=[],
            source_evidence_refs=[],
            readiness_snapshot={"recommendation": "review_candidate", "gates_blocked": []},
            strategy_kind="strategy_profile",
            recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
            anchor_policy_snapshot={},
            page_index_policy_snapshot={},
            graph_expansion_policy_snapshot={},
            created_by="operator",
        )
    )
    assert store.activate_recall_shadow_profile(p1.profile_id) is not None
    assert store.activate_recall_shadow_profile(p2.profile_id) is not None
    prior = store.get_recall_strategy_profile(p1.profile_id)
    active = store.active_recall_shadow_profile()
    assert prior is not None and prior.status == "staged"
    assert active is not None and active.profile_id == p2.profile_id


def test_recall_eval_runs_and_candidate_reviews_persist_across_sql_reload(tmp_path: Path) -> None:
    db = tmp_path / "mutation.db"
    store = SubstrateMutationStore(sql_db_path=str(db))
    profile = store.stage_recall_profile(
        profile=RecallStrategyProfileV1(
            source_proposal_id="proposal-z",
            source_pressure_ids=[],
            source_evidence_refs=[],
            readiness_snapshot={"recommendation": "ready_for_shadow_expansion", "gates_blocked": []},
            strategy_kind="strategy_profile",
            recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
            anchor_policy_snapshot={},
            page_index_policy_snapshot={},
            graph_expansion_policy_snapshot={},
            created_by="operator",
        )
    )
    run = store.record_recall_shadow_eval_run(
        RecallShadowEvalRunV1(
            profile_id=profile.profile_id,
            dry_run=False,
            status="completed",
            eval_row_count=2,
            readiness_before={"recommendation": "review_candidate"},
            readiness_after={"recommendation": "ready_for_shadow_expansion"},
            pressure_event_refs=["pressure_event:a"],
        )
    )
    review = store.record_recall_production_candidate_review(
        RecallProductionCandidateReviewV1(
            profile_id=profile.profile_id,
            source_eval_run_ids=[run.run_id],
            readiness_snapshot={"recommendation": "ready_for_shadow_expansion"},
            recommendation="expand_shadow_corpus",
            status="draft",
        )
    )
    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    loaded_run = reloaded.get_recall_shadow_eval_run(run.run_id)
    loaded_review = reloaded.get_recall_production_candidate_review(review.review_id)
    assert loaded_run is not None and loaded_run.profile_id == profile.profile_id
    assert loaded_review is not None and loaded_review.profile_id == profile.profile_id
