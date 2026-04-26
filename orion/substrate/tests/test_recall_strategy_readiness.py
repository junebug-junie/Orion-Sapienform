from __future__ import annotations

from orion.core.schemas.substrate_mutation import (
    MutationDecisionV1,
    MutationPressureEvidenceV1,
    MutationPressureV1,
)
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner
from orion.substrate.recall_strategy_readiness import (
    compute_recall_strategy_readiness,
    readiness_from_telemetry_records,
    readiness_for_pressure,
)


def _row(
    *,
    case_id: str,
    v1p: float = 0.2,
    v2p: float = 0.38,
    v1_lat: float = 100.0,
    v2_lat: float = 130.0,
    cousin: float = 0.15,
    source: str = "recall_eval_suite",
) -> dict[str, object]:
    return {
        "source": source,
        "case_id": case_id,
        "v1_latency_ms": v1_lat,
        "v2_latency_ms": v2_lat,
        "v1_precision_proxy": v1p,
        "v2_precision_proxy": v2p,
        "v2_irrelevant_cousin_rate": cousin,
        "v2_entity_time_match_rate": 0.6,
        "v2_explainability_completeness": 0.85,
    }


def test_weak_evidence_returns_not_ready() -> None:
    rows = [_row(case_id="only-one")]
    r = compute_recall_strategy_readiness(
        compare_rows=rows,
        failure_categories=["recall_miss_or_dissatisfaction"],
        corpus_total_cases=10,
        minimum_evidence_cases_required=3,
    )
    assert r.recommendation == "not_ready"
    assert "insufficient_evidence" in r.gates_blocked
    assert r.minimum_evidence_met is False


def test_strong_eval_can_reach_ready_for_operator_promotion() -> None:
    rows = [_row(case_id=f"c{i}") for i in range(10)]
    r = compute_recall_strategy_readiness(
        compare_rows=rows,
        failure_categories=["recall_miss_or_dissatisfaction"] * 10,
        corpus_total_cases=10,
        minimum_evidence_cases_required=3,
    )
    assert r.recommendation == "ready_for_operator_promotion"
    assert r.minimum_evidence_met is True
    assert r.corpus_coverage >= 0.75
    assert not r.gates_blocked


def test_intermediate_corpus_coverage_is_review_candidate() -> None:
    rows = [_row(case_id=f"c{i}") for i in range(6)]
    r = compute_recall_strategy_readiness(
        compare_rows=rows,
        failure_categories=["recall_miss_or_dissatisfaction"] * 6,
        corpus_total_cases=10,
        minimum_evidence_cases_required=3,
    )
    assert r.corpus_coverage == 0.6
    assert r.recommendation == "review_candidate"


def test_low_corpus_coverage_triggers_ready_for_shadow_expansion() -> None:
    rows = [_row(case_id=f"c{i}") for i in range(5)]
    r = compute_recall_strategy_readiness(
        compare_rows=rows,
        failure_categories=["recall_miss_or_dissatisfaction"] * 5,
        corpus_total_cases=20,
        minimum_evidence_cases_required=3,
    )
    assert r.corpus_coverage == 0.25
    assert r.minimum_evidence_met is True
    assert r.recommendation == "ready_for_shadow_expansion"


def test_high_irrelevant_cousin_rate_blocks_readiness() -> None:
    rows = [_row(case_id=f"c{i}", cousin=0.5) for i in range(5)]
    r = compute_recall_strategy_readiness(
        compare_rows=rows,
        failure_categories=["irrelevant_semantic_neighbor"] * 5,
        corpus_total_cases=10,
        minimum_evidence_cases_required=3,
    )
    assert r.recommendation == "not_ready"
    assert "high_irrelevant_cousin_rate" in r.gates_blocked


def test_latency_regression_blocks_readiness() -> None:
    rows = [_row(case_id=f"c{i}", v1_lat=50.0, v2_lat=400.0) for i in range(5)]
    r = compute_recall_strategy_readiness(
        compare_rows=rows,
        failure_categories=["recall_miss_or_dissatisfaction"] * 5,
        corpus_total_cases=10,
        minimum_evidence_cases_required=3,
    )
    assert r.recommendation == "not_ready"
    assert "latency_regression" in r.gates_blocked


def test_readiness_from_telemetry_records_uses_eval_case_ids_for_coverage() -> None:
    ev = MutationPressureEvidenceV1(
        source_service="orion-recall-eval",
        source_event_id="run:a:0",
        pressure_category="recall_miss_or_dissatisfaction",
        confidence=0.8,
        evidence_refs=["recall_eval_case:0"],
        metadata={
            "v1_v2_compare": _row(case_id="suite-0"),
            "recall_eval_case": {"case_id": "suite-0"},
        },
    )
    rec = GraphReviewTelemetryRecordV1(
        invocation_surface="operator_review",
        execution_outcome="executed",
        selection_reason="recall_eval",
        runtime_duration_ms=1,
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="autonomy_graph",
        pressure_events=[ev],
    )
    r = readiness_from_telemetry_records([rec], corpus_total_cases=4)
    assert r.evidence_observation_count == 1
    assert r.corpus_coverage == 0.25
    assert any("eval_suite_cases_observed" in n for n in r.readiness_notes)


def test_recall_proposal_payload_contains_readiness() -> None:
    snap = {
        "failure_category": "recall_miss_or_dissatisfaction",
        "recall_compare": _row(case_id="p1"),
    }
    pressure = MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="recall_strategy_profile",
        pressure_kind="pressure_event:recall_miss_or_dissatisfaction",
        pressure_score=6.0,
        evidence_refs=["recall_compare:1"],
        source_signal_ids=["sig-1"],
        recall_evidence_snapshot=snap,
        recall_evidence_history=[],
    )
    proposal = ProposalFactory().from_pressure(pressure)
    assert proposal is not None
    readiness = proposal.patch.patch.get("recall_strategy_readiness")
    assert isinstance(readiness, dict)
    assert readiness.get("recommendation") in {
        "not_ready",
        "review_candidate",
        "ready_for_shadow_expansion",
        "ready_for_operator_promotion",
    }
    assert "evidence_observation_count" in readiness


def test_recall_lineage_contains_readiness_under_pressure_lineage() -> None:
    snap = {"failure_category": "missing_exact_anchor", "recall_compare": _row(case_id="l1", source="shadow")}
    pressure = MutationPressureV1(
        pressure_id="pressure-readiness-lineage-1",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="recall_strategy_profile",
        pressure_kind="pressure_event:missing_exact_anchor",
        pressure_score=5.0,
        evidence_refs=["recall_compare:x"],
        source_signal_ids=["sig-lineage"],
        recall_evidence_snapshot=snap,
        recall_evidence_history=[{"failure_category": "missing_exact_anchor", "recall_compare": _row(case_id="l2", source="shadow")}],
    )
    proposal = ProposalFactory().from_pressure(pressure)
    assert proposal is not None
    store = SubstrateMutationStore()
    store.record_pressure(pressure)
    store.add_proposal(proposal)
    life = store.lifecycle_for_proposal(proposal.proposal_id)
    assert life is not None
    lineage = life.get("recall_pressure_evidence_lineage")
    assert isinstance(lineage, dict)
    assert "recall_strategy_readiness" in lineage
    assert lineage["recall_strategy_readiness"]["recommendation"] in {
        "not_ready",
        "review_candidate",
        "ready_for_shadow_expansion",
        "ready_for_operator_promotion",
    }


def test_recall_candidate_still_no_adoption_under_auto_promote() -> None:
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
    fake_promote = MutationDecisionV1(
        proposal_id=proposal.proposal_id,
        action="auto_promote",
        reason="hypothetical_misconfig",
    )
    assert PatchApplier(surfaces={}).apply(proposal=proposal, decision=fake_promote) is None
    assert decision.action == "require_review"


def test_readiness_for_pressure_aggregates_history() -> None:
    snap = {"failure_category": "stale_memory_selected", "recall_compare": _row(case_id="h0")}
    hist = [
        {"failure_category": "stale_memory_selected", "recall_compare": _row(case_id="h1")},
        {"failure_category": "stale_memory_selected", "recall_compare": _row(case_id="h2")},
    ]
    pressure = MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="recall_anchor_policy",
        pressure_kind="pressure_event:stale_memory_selected",
        pressure_score=4.0,
        evidence_refs=["recall_compare:a"],
        source_signal_ids=["sig-h"],
        recall_evidence_snapshot=snap,
        recall_evidence_history=hist,
    )
    r = readiness_for_pressure(pressure, corpus_total_cases=10)
    assert r.evidence_observation_count == 3


def test_precision_only_regression_yields_review_candidate_not_ready_false() -> None:
    rows = [
        {
            "source": "recall_eval_suite",
            "case_id": f"p{i}",
            "v1_latency_ms": 100,
            "v2_latency_ms": 110,
            "v1_precision_proxy": 0.7,
            "v2_precision_proxy": 0.2,
            "v2_irrelevant_cousin_rate": 0.1,
        }
        for i in range(5)
    ]
    r = compute_recall_strategy_readiness(
        compare_rows=rows,
        failure_categories=["unsupported_memory_claim"] * 5,
        corpus_total_cases=10,
        minimum_evidence_cases_required=3,
    )
    assert "precision_regression" in r.gates_blocked
    assert r.recommendation == "review_candidate"
