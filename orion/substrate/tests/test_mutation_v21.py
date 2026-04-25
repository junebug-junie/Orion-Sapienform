from __future__ import annotations

import os
from pathlib import Path

from orion.core.schemas.substrate_mutation import MutationPressureV1, MutationSignalV1
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


def test_mutation_proposal_requires_evidence_and_rollback() -> None:
    pressure = MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="routing",
        pressure_kind="runtime_failure",
        pressure_score=8.0,
        evidence_refs=["telemetry:1"],
        source_signal_ids=["signal-1"],
    )
    proposal = ProposalFactory().from_pressure(pressure)
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
    pressure = MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="routing",
        pressure_kind="runtime_failure",
        pressure_score=8.0,
        evidence_refs=["telemetry:1"],
        source_signal_ids=["signal-1"],
    )
    proposal1 = ProposalFactory().from_pressure(pressure)
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

    proposal2 = ProposalFactory().from_pressure(pressure)
    assert proposal2 is not None
    decision2 = DecisionEngine().decide(
        proposal=proposal2,
        trial=trial_runner.run_trial(proposal=proposal2, measured_metrics={"success_rate_delta": 0.1, "latency_ms_delta": 0.0}),
        has_replay_and_baseline=True,
        active_surface_exists=store.active_surface(proposal2.target_surface) is not None,
    )
    assert decision2.action == "hold"


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
