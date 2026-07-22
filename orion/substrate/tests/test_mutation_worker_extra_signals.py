from __future__ import annotations

import os
from datetime import datetime, timezone

from orion.core.schemas.substrate_mutation import MutationSignalV1
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_pressure import PressureAccumulator, PressurePolicy
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_worker import AdaptationCycleBudget, SubstrateAdaptationWorker


def _continuity_pressure_signal(strength: float = 0.3) -> MutationSignalV1:
    """A literal stand-in for what orion/substrate/mutation_self_revision.py's
    prediction_error_mutation_signals() used to produce for a continuity_pressure
    self-model dimension (deleted 2026-07-22, SelfStateV1 burn -- that module had
    zero real production callers, only these tests). These tests exist to cover
    SubstrateAdaptationWorker.run_cycle()'s generic extra_signals folding/budget
    behavior, which is independent of whatever produced the signal."""
    return MutationSignalV1(
        event_kind="self_model_drift:continuity_pressure",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="cognitive_identity_continuity_adjustment",
        target_zone="concept_graph",
        strength=strength,
        evidence_refs=["self_state:ss-revision-worker", "self_dimension:continuity_pressure"],
        source_ref="self_state:ss-revision-worker",
        metadata={
            "source_kind": "self_model_prediction_error",
            "self_dimension_id": "continuity_pressure",
            "prediction_error": round(strength, 6),
            "trajectory": 0.0,
        },
    )


def _build_worker(store: SubstrateMutationStore) -> SubstrateAdaptationWorker:
    return SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(policy=PressurePolicy()),
        proposals=ProposalFactory(),
        trial_runner=SubstrateTrialRunner(
            scorer=ClassSpecificScorer(),
            corpus_registry=ReplayCorpusRegistry(corpus_by_class={}, baseline_metric_ref_by_class={}),
        ),
        decision_engine=DecisionEngine(),
        applier=PatchApplier(surfaces={}),
        monitor=PostAdoptionMonitor(),
        budget=AdaptationCycleBudget(max_signals=64, max_proposals=8, max_trials=8, max_adoptions=2),
    )


def test_run_cycle_folds_in_extra_signals_and_sustains_to_a_proposal() -> None:
    os.environ["SUBSTRATE_MUTATION_AUTONOMY_ENABLED"] = "true"
    store = SubstrateMutationStore()
    worker = _build_worker(store)
    now = datetime.now(timezone.utc)

    signal = _continuity_pressure_signal(0.3)

    result = worker.run_cycle(
        telemetry=[],
        measured_metrics_by_proposal={},
        extra_signals=[signal],
        now=now,
    )
    assert result["signals"] == 1
    # one weak tick is not enough to draft a proposal yet
    assert result["proposals"] == 0

    proposal_created = False
    for _ in range(9):
        result = worker.run_cycle(
            telemetry=[],
            measured_metrics_by_proposal={},
            extra_signals=[signal],
            now=now,
        )
        assert result["signals"] == 1
        if result["proposals"] >= 1:
            proposal_created = True
            break
    assert proposal_created

    cognitive_proposals = [p for p in store._proposals.values() if p.lane == "cognitive"]
    assert cognitive_proposals
    assert cognitive_proposals[0].mutation_class == "cognitive_identity_continuity_adjustment"
    assert cognitive_proposals[0].patch.patch["not_applied_status"] == "draft_only_not_applied"


def test_run_cycle_extra_signals_respect_max_signals_kill_lever() -> None:
    """extra_signals must fold into the same operator budget as telemetry
    signals, so max_signals=0 (the routing-proposals-disabled kill lever)
    also silences self-revision signals rather than bypassing it."""
    os.environ["SUBSTRATE_MUTATION_AUTONOMY_ENABLED"] = "true"
    store = SubstrateMutationStore()
    worker = SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(policy=PressurePolicy()),
        proposals=ProposalFactory(),
        trial_runner=SubstrateTrialRunner(
            scorer=ClassSpecificScorer(),
            corpus_registry=ReplayCorpusRegistry(corpus_by_class={}, baseline_metric_ref_by_class={}),
        ),
        decision_engine=DecisionEngine(),
        applier=PatchApplier(surfaces={}),
        monitor=PostAdoptionMonitor(),
        budget=AdaptationCycleBudget(max_signals=0, max_proposals=8, max_trials=8, max_adoptions=2),
    )
    now = datetime.now(timezone.utc)
    signal = _continuity_pressure_signal(0.3)

    result = worker.run_cycle(
        telemetry=[],
        measured_metrics_by_proposal={},
        extra_signals=[signal],
        now=now,
    )
    assert result["signals"] == 0


def test_run_cycle_extra_signals_share_budget_with_telemetry_signals() -> None:
    """When telemetry signals already consume the budget, extra_signals get
    only what remains rather than an independent fixed slice."""
    os.environ["SUBSTRATE_MUTATION_AUTONOMY_ENABLED"] = "true"
    store = SubstrateMutationStore()
    worker = SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(policy=PressurePolicy()),
        proposals=ProposalFactory(),
        trial_runner=SubstrateTrialRunner(
            scorer=ClassSpecificScorer(),
            corpus_registry=ReplayCorpusRegistry(corpus_by_class={}, baseline_metric_ref_by_class={}),
        ),
        decision_engine=DecisionEngine(),
        applier=PatchApplier(surfaces={}),
        monitor=PostAdoptionMonitor(),
        budget=AdaptationCycleBudget(max_signals=1, max_proposals=8, max_trials=8, max_adoptions=2),
    )
    now = datetime.now(timezone.utc)
    signal = _continuity_pressure_signal(0.3)

    result = worker.run_cycle(
        telemetry=[],
        measured_metrics_by_proposal={},
        extra_signals=[signal, signal, signal],
        now=now,
    )
    assert result["signals"] == 1


def test_run_cycle_extra_signals_default_none_behaves_like_today() -> None:
    os.environ["SUBSTRATE_MUTATION_AUTONOMY_ENABLED"] = "true"
    store = SubstrateMutationStore()
    worker = _build_worker(store)
    now = datetime.now(timezone.utc)

    result_without_kwarg = worker.run_cycle(
        telemetry=[],
        measured_metrics_by_proposal={},
        now=now,
    )
    result_with_none = worker.run_cycle(
        telemetry=[],
        measured_metrics_by_proposal={},
        extra_signals=None,
        now=now,
    )
    assert result_without_kwarg["signals"] == 0
    assert result_with_none["signals"] == 0
    assert result_without_kwarg["proposals"] == 0
    assert result_with_none["proposals"] == 0
