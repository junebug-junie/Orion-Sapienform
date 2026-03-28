from __future__ import annotations

from orion.spark.concept_induction.parity_evidence import (
    ParityReadinessThresholds,
    configure_parity_evidence_store,
    get_parity_evidence_snapshot,
    record_parity_evidence,
    reset_parity_evidence_store,
)


def setup_function() -> None:
    reset_parity_evidence_store()


def test_parity_aggregation_rolls_up_by_consumer_and_mismatch_class() -> None:
    configure_parity_evidence_store(
        thresholds=ParityReadinessThresholds(min_comparisons=2, max_mismatch_rate=0.5, max_unavailable_rate=0.5),
        summary_interval=10,
    )

    record_parity_evidence(
        consumer="concept_induction_pass",
        subject_outcomes=[
            {"subject": "orion", "mismatch_classes": [], "graph_unavailable": False, "empty_on_local_only": False, "empty_on_graph_only": False},
            {"subject": "juniper", "mismatch_classes": ["revision_mismatch"], "graph_unavailable": False, "empty_on_local_only": False, "empty_on_graph_only": False},
        ],
    )

    snap = get_parity_evidence_snapshot()
    evidence = snap["consumers"]["concept_induction_pass"]
    assert evidence["total_comparisons"] == 2
    assert evidence["exact_matches"] == 1
    assert evidence["mismatches"] == 1
    assert evidence["mismatch_class_counts"]["revision_mismatch"] == 1


def test_graph_unavailable_aggregation_counts_without_altering_local_return_semantics() -> None:
    configure_parity_evidence_store(
        thresholds=ParityReadinessThresholds(min_comparisons=1, max_mismatch_rate=1.0, max_unavailable_rate=1.0),
        summary_interval=10,
    )

    record_parity_evidence(
        consumer="chat_stance",
        subject_outcomes=[
            {
                "subject": "orion",
                "mismatch_classes": ["graph_unavailable"],
                "graph_unavailable": True,
                "empty_on_local_only": False,
                "empty_on_graph_only": False,
            }
        ],
    )

    snap = get_parity_evidence_snapshot()
    evidence = snap["consumers"]["chat_stance"]
    assert evidence["graph_unavailable_count"] == 1
    assert evidence["mismatches"] == 1


def test_readiness_evaluation_false_then_true_under_thresholds() -> None:
    configure_parity_evidence_store(
        thresholds=ParityReadinessThresholds(
            min_comparisons=3,
            max_mismatch_rate=0.34,
            max_unavailable_rate=0.2,
            critical_mismatch_classes=("query_error",),
        ),
        summary_interval=10,
    )

    record_parity_evidence(
        consumer="concept_induction_pass",
        subject_outcomes=[
            {"subject": "orion", "mismatch_classes": ["revision_mismatch"], "graph_unavailable": False, "empty_on_local_only": False, "empty_on_graph_only": False},
            {"subject": "juniper", "mismatch_classes": [], "graph_unavailable": False, "empty_on_local_only": False, "empty_on_graph_only": False},
        ],
    )
    snap = get_parity_evidence_snapshot()
    assert snap["readiness"]["concept_induction_pass"]["ready"] is False
    assert snap["readiness"]["concept_induction_pass"]["reason"] == "insufficient_samples"

    record_parity_evidence(
        consumer="concept_induction_pass",
        subject_outcomes=[
            {"subject": "relationship", "mismatch_classes": [], "graph_unavailable": False, "empty_on_local_only": False, "empty_on_graph_only": False}
        ],
    )
    snap2 = get_parity_evidence_snapshot()
    assert snap2["readiness"]["concept_induction_pass"]["ready"] is True


def test_inspectable_surface_returns_expected_shape_and_consumer_separation() -> None:
    configure_parity_evidence_store(
        thresholds=ParityReadinessThresholds(min_comparisons=1, max_mismatch_rate=1.0, max_unavailable_rate=1.0),
        summary_interval=1,
    )

    record_parity_evidence(
        consumer="concept_induction_pass",
        subject_outcomes=[
            {"subject": "orion", "mismatch_classes": [], "graph_unavailable": False, "empty_on_local_only": False, "empty_on_graph_only": False}
        ],
    )
    record_parity_evidence(
        consumer="chat_stance",
        subject_outcomes=[
            {"subject": "juniper", "mismatch_classes": ["concept_count_mismatch"], "graph_unavailable": False, "empty_on_local_only": False, "empty_on_graph_only": False}
        ],
    )

    snap = get_parity_evidence_snapshot()
    assert "thresholds" in snap
    assert "consumers" in snap
    assert "readiness" in snap
    assert snap["consumers"]["concept_induction_pass"]["total_comparisons"] == 1
    assert snap["consumers"]["chat_stance"]["total_comparisons"] == 1
    assert snap["consumers"]["chat_stance"]["mismatch_class_counts"]["concept_count_mismatch"] == 1
