from __future__ import annotations

import os
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
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_pressure import PressureAccumulator, PressurePolicy
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_worker import SubstrateAdaptationWorker
from orion.substrate.scripts.smoke_mutation_v21 import run_smoke


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


def test_signal_to_pressure_pipeline() -> None:
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
    assert len(signals) == 1
    pressure = PressureAccumulator(policy=PressurePolicy(activation_threshold=0.2)).apply(current=None, signal=signals[0])
    assert pressure.pressure_score > 0
    assert pressure.target_surface == "routing"


def test_routing_detector_emits_richer_runtime_social_pressure_signals() -> None:
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
    kinds = {item.event_kind for item in signals}
    assert "routing_decision_mismatch" in kinds
    assert "routing_recall_dissatisfaction" in kinds
    assert "routing_runtime_degradation" in kinds
    provenance_signal = next(item for item in signals if item.event_kind == "routing_decision_mismatch")
    assert provenance_signal.target_surface == "routing"
    assert provenance_signal.metadata["source_kind"] == "routing_mismatch_signal"
    assert provenance_signal.metadata["derived_signal_kind"] == "routing_decision_mismatch"
    assert provenance_signal.metadata["confidence"] == provenance_signal.strength
    assert any(ref.startswith("telemetry:") for ref in provenance_signal.evidence_refs)
    assert any(ref.startswith("source_kind:") for ref in provenance_signal.evidence_refs)


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


def test_pressure_events_become_mutation_signals_with_event_provenance() -> None:
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
        target_zone="autonomy_graph",
        pressure_events=[event],
    )
    signals = detector.from_review_telemetry([telemetry])
    matched = [item for item in signals if item.metadata.get("source_kind") == "producer_pressure_event"]
    assert len(matched) == 1
    signal = matched[0]
    assert signal.target_surface == "routing"
    assert signal.metadata["pressure_event_id"] == "pressure-evt-1"
    assert signal.metadata["source_event_id"] == "feedback-1"
    assert any(ref == "pressure_event:pressure-evt-1" for ref in signal.evidence_refs)


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
    proposal = ProposalFactory().from_pressure(pressure)
    assert proposal is not None
    assert proposal.lane == "cognitive"
    assert proposal.target_surface == "cognitive_stance_continuity_adjustment"
    assert proposal.mutation_class == "cognitive_stance_continuity_adjustment"
    assert proposal.patch.patch["not_applied_status"] == "draft_only_not_applied"
    assert any(note.startswith("blast_radius:") for note in proposal.notes)
    assert proposal.patch.rollback_payload


def test_cognitive_lane_decisions_are_always_require_review() -> None:
    proposal = ProposalFactory().from_pressure(
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
        proposal = ProposalFactory().from_pressure(
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
    proposal = ProposalFactory().from_pressure(
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
    proposal = ProposalFactory().from_pressure(
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
    proposal = ProposalFactory().from_pressure(pressure)
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
    proposal = ProposalFactory().from_pressure(
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
    proposal = ProposalFactory().from_pressure(
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
    proposal = ProposalFactory().from_pressure(pressure)
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
    proposal = ProposalFactory().from_pressure(pressure)
    assert proposal is not None
    hist = proposal.patch.patch.get("contributing_recall_evidence_history")
    assert isinstance(hist, list) and len(hist) == 8


def test_routing_threshold_proposal_class_unchanged() -> None:
    proposal = ProposalFactory().from_pressure(_routing_pressure())
    assert proposal is not None
    assert proposal.mutation_class == "routing_threshold_patch"


def test_pressure_to_proposal() -> None:
    proposal = ProposalFactory().from_pressure(_routing_pressure())
    assert proposal is not None
    assert proposal.patch.target_surface == "routing"
    assert proposal.source_pressure_id


def test_proposal_to_trial() -> None:
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(pressure)
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
    proposal1 = ProposalFactory().from_pressure(_routing_pressure())
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

    proposal2 = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
        proposals=ProposalFactory(),
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
    proposal = ProposalFactory().from_pressure(pressure)
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
    store = SubstrateMutationStore()
    proposal = ProposalFactory().from_pressure(_routing_pressure())
    assert proposal is not None
    store._active_surface_by_target[proposal.target_surface] = "existing-adoption"
    applier = PatchApplier(surfaces={proposal.target_surface: {"chat_reflective_lane_threshold": 0.5}})
    worker = SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(policy=PressurePolicy(activation_threshold=0.1, cooldown_seconds=5)),
        proposals=ProposalFactory(),
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
    )
    os.environ["SUBSTRATE_MUTATION_AUTONOMY_ENABLED"] = "true"
    result = worker.run_cycle(
        telemetry=[
            GraphReviewTelemetryRecordV1(
                invocation_surface="operator_review",
                execution_outcome="failed",
                selection_reason="x",
                runtime_duration_ms=12,
                anchor_scope="orion",
                subject_ref="entity:orion",
                target_zone="autonomy_graph",
            )
        ],
        measured_metrics_by_proposal={},
    )
    assert result["adoptions"] == 0
    assert applier.surfaces[proposal.target_surface]["chat_reflective_lane_threshold"] == 0.5


def test_rollback_payload_required_before_apply() -> None:
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(pressure)
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
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    signal = MutationDetectors().from_review_telemetry(
        [
            GraphReviewTelemetryRecordV1(
                invocation_surface="operator_review",
                execution_outcome="failed",
                selection_reason="targeted_signal_persist",
                runtime_duration_ms=8,
                anchor_scope="orion",
                subject_ref="entity:orion",
                target_zone="autonomy_graph",
            )
        ]
    )[0]
    store.record_signal(signal)
    reloaded = SubstrateMutationStore(sql_db_path=str(db))
    assert any(item.signal_id == signal.signal_id for item in reloaded._signals)


def test_routing_replay_corpus_drives_trial_metrics_without_manual_injection() -> None:
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
    proposal = ProposalFactory().from_pressure(_routing_pressure())
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
