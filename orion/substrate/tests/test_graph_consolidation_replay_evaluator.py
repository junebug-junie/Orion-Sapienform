"""graph_consolidation_param_patch's trial metrics, derived from real telemetry.

Before this, a graph_consolidation_param_patch proposal's trial always came
back `inconclusive` -- ClassSpecificScorer.evaluate() requires
queue_resolution_delta/requeue_rate_delta (mutation_contracts.py's contract),
and nothing anywhere ever computed either one; smoke_mutation_v21.py hand-typed
constants for its own demo, and no real code path produced them. Confirmed
live 2026-09-08: the first real graph_consolidation proposal ever generated
after the prior-cycle-wiring fix (PR #2162) landed exactly on
`trial_inconclusive` for this reason.

GraphConsolidationReplayEvaluator is the real producer: split real replay
telemetry into an older and newer half, compare resolution/requeue rates
between them. Not a counterfactual simulation of the proposed patch values
(unlike routing's evaluator) -- see the class docstring in mutation_trials.py
for why that isn't honestly possible here.

Code review caught two real bugs in the first pass, both fixed here:
"reinforce" was wrongly counted as a resolved outcome (the scheduler treats
it identically to keep_provisional -- neither removes the item from the
queue, only retire/noop do); and two identical outcome distributions
produced a hollow {0.0, 0.0} that metric_passed()'s `>= 0.0` floor reads as
passing -- auto-promoting on zero evidence, which was the literal historical
norm (100% keep_provisional, every record, until 2026-09-08).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.substrate_mutation import MutationPatchV1, MutationProposalV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import (
    GraphConsolidationReplayEvaluator,
    ReplayCorpusRegistry,
    SubstrateTrialRunner,
)


def _telemetry(*, outcomes: list[str], at: datetime) -> GraphReviewTelemetryRecordV1:
    return GraphReviewTelemetryRecordV1(
        invocation_surface="operator_review",
        target_zone="world_ontology",
        selection_reason="eligible_item_selected",
        execution_outcome="executed",
        runtime_duration_ms=5,
        selected_at=at,
        consolidation_outcomes=outcomes,
    )


def _graph_consolidation_proposal() -> MutationProposalV1:
    return MutationProposalV1(
        lane="operational",
        mutation_class="graph_consolidation_param_patch",
        risk_tier="medium",
        target_surface="graph_consolidation",
        anchor_scope="orion",
        subject_ref="entity:orion",
        rationale="test",
        expected_effect="reduce_runtime_failure",
        evidence_refs=["telemetry:placeholder"],
        source_signal_ids=["signal:placeholder"],
        source_pressure_id="pressure-test",
        patch=MutationPatchV1(
            mutation_class="graph_consolidation_param_patch",
            target_surface="graph_consolidation",
            target_ref="graph_consolidation",
            patch={"query_limit_nodes": 96},
            rollback_payload={"query_limit_nodes": 64},
        ),
    )


def test_derive_metrics_is_none_below_the_minimum_window() -> None:
    evaluator = GraphConsolidationReplayEvaluator()
    base = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    records = [_telemetry(outcomes=["keep_provisional"], at=base + timedelta(minutes=i)) for i in range(4)]

    assert evaluator.derive_metrics(replay_records=records) is None


def test_reinforce_is_not_counted_as_resolved() -> None:
    """GraphReviewScheduler._schedule_for_decision() groups "reinforce" with
    "keep_provisional" under the same bounded-monitoring cadence -- neither
    removes the item from the queue. Only "retire"/"noop" actually do
    (they return queue_item=None, so the item is never re-upserted). A
    shift from keep_provisional to reinforce is therefore NOT resolution,
    even though it's real, positive information (proven separately by PR
    #2162's own regression tests) -- it must not move this metric.
    """
    evaluator = GraphConsolidationReplayEvaluator()
    base = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    older = [_telemetry(outcomes=["keep_provisional"], at=base + timedelta(minutes=i)) for i in range(3)]
    newer = [_telemetry(outcomes=["reinforce"], at=base + timedelta(minutes=10 + i)) for i in range(3)]

    # Both halves are "0% resolved, 0% requeued" under the corrected
    # definition -- identical distributions -- so this must read as
    # insufficient evidence, not a fake win.
    assert evaluator.derive_metrics(replay_records=older + newer) is None


def test_derive_metrics_reads_improving_resolution_as_positive() -> None:
    """Older half all keep_provisional (0% resolved), newer half all
    retire (100% resolved) -- retire is the one outcome that actually
    empties the queue (review_schedule.py)."""
    evaluator = GraphConsolidationReplayEvaluator()
    base = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    older = [_telemetry(outcomes=["keep_provisional"], at=base + timedelta(minutes=i)) for i in range(3)]
    newer = [_telemetry(outcomes=["retire"], at=base + timedelta(minutes=10 + i)) for i in range(3)]

    metrics = evaluator.derive_metrics(replay_records=older + newer)

    assert metrics is not None
    assert metrics["queue_resolution_delta"] == 1.0
    assert metrics["requeue_rate_delta"] == 0.0
    assert metrics["graph_consolidation_replay_case_count"] == 6.0


def test_derive_metrics_reads_a_dropping_requeue_rate_as_positive() -> None:
    """Older half all requeue_review, newer half all retire -- requeue rate
    drops from 100% to 0%. That IMPROVEMENT must read as a positive delta,
    even though the raw newer-minus-older requeue rate is negative --
    metric_passed() treats every metric as "higher is better"."""
    evaluator = GraphConsolidationReplayEvaluator()
    base = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    older = [_telemetry(outcomes=["requeue_review"], at=base + timedelta(minutes=i)) for i in range(3)]
    newer = [_telemetry(outcomes=["retire"], at=base + timedelta(minutes=10 + i)) for i in range(3)]

    metrics = evaluator.derive_metrics(replay_records=older + newer)

    assert metrics is not None
    assert metrics["requeue_rate_delta"] == 1.0


def test_derive_metrics_is_none_when_both_halves_are_identical() -> None:
    """The actual historical condition, confirmed live: 100% keep_provisional,
    every record, for this system's entire history until 2026-09-08. Two
    identical distributions must read as "no evidence of change", not a
    flat-but-passing 0.0/0.0 -- metric_passed()'s `>= 0.0` floor would
    otherwise auto-promote a proposal that measured literally nothing.
    """
    evaluator = GraphConsolidationReplayEvaluator()
    base = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    records = [_telemetry(outcomes=["keep_provisional"], at=base + timedelta(minutes=i)) for i in range(6)]

    assert evaluator.derive_metrics(replay_records=records) is None


def test_derive_metrics_ignores_rows_with_no_real_consolidation() -> None:
    evaluator = GraphConsolidationReplayEvaluator()
    base = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    noise = [
        GraphReviewTelemetryRecordV1(
            invocation_surface="operator_review",
            target_zone="world_ontology",
            selection_reason="noop",
            execution_outcome="noop",
            runtime_duration_ms=0,
            selected_at=base + timedelta(minutes=i),
            consolidation_outcomes=[],
        )
        for i in range(10)
    ]
    real = [_telemetry(outcomes=["keep_provisional"], at=base + timedelta(minutes=20 + i)) for i in range(3)]

    assert evaluator.derive_metrics(replay_records=noise + real) is None


def test_trial_runner_produces_a_real_status_for_graph_consolidation_given_enough_replay() -> None:
    """The actual regression: before GraphConsolidationReplayEvaluator existed,
    this trial always came back inconclusive (missing_class_metrics) no
    matter what replay_records held, because nothing ever populated
    queue_resolution_delta/requeue_rate_delta."""
    runner = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={"graph_consolidation_param_patch": "replay-consolidation-v1"},
            baseline_metric_ref_by_class={"graph_consolidation_param_patch": "baseline-consolidation-v1"},
        ),
    )
    base = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    older = [_telemetry(outcomes=["keep_provisional"], at=base + timedelta(minutes=i)) for i in range(4)]
    newer = [_telemetry(outcomes=["retire"], at=base + timedelta(minutes=10 + i)) for i in range(4)]

    trial = runner.run_trial(
        proposal=_graph_consolidation_proposal(),
        measured_metrics={},
        replay_records=older + newer,
    )

    assert trial.status == "passed"
    assert trial.metrics["queue_resolution_delta"] == 1.0
    assert "missing_class_metrics" not in trial.notes


def test_trial_runner_stays_inconclusive_with_too_little_replay() -> None:
    runner = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={"graph_consolidation_param_patch": "replay-consolidation-v1"},
            baseline_metric_ref_by_class={"graph_consolidation_param_patch": "baseline-consolidation-v1"},
        ),
    )

    trial = runner.run_trial(
        proposal=_graph_consolidation_proposal(),
        measured_metrics={},
        replay_records=[],
    )

    assert trial.status == "inconclusive"
    assert trial.notes == ["missing_class_metrics"]


def test_trial_runner_stays_inconclusive_when_replay_shows_no_real_change() -> None:
    """The exact bug the second review pass caught: without the
    identical-distributions guard, this would have returned
    status="passed" from telemetry that shows zero real difference."""
    runner = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={"graph_consolidation_param_patch": "replay-consolidation-v1"},
            baseline_metric_ref_by_class={"graph_consolidation_param_patch": "baseline-consolidation-v1"},
        ),
    )
    base = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    records = [_telemetry(outcomes=["keep_provisional"], at=base + timedelta(minutes=i)) for i in range(8)]

    trial = runner.run_trial(
        proposal=_graph_consolidation_proposal(),
        measured_metrics={},
        replay_records=records,
    )

    assert trial.status == "inconclusive"
