from __future__ import annotations

import os
from datetime import datetime, timezone

from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_pressure import PressureAccumulator, PressurePolicy
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_self_revision import prediction_error_mutation_signals
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_worker import AdaptationCycleBudget, SubstrateAdaptationWorker


def _self_state(prediction_error_scores: dict[str, float]) -> SelfStateV1:
    now = datetime.now(timezone.utc)
    dims = {
        dim: SelfStateDimensionV1(dimension_id=dim, score=0.5, confidence=0.7)
        for dim in prediction_error_scores
    }
    return SelfStateV1(
        self_state_id="ss-revision-worker",
        generated_at=now,
        source_field_tick_id="ft",
        source_field_generated_at=now,
        source_attention_frame_id="af",
        source_attention_generated_at=now,
        overall_condition="strained",
        overall_intensity=0.6,
        overall_confidence=0.6,
        dimensions=dims,
        prediction_error_scores=prediction_error_scores,
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

    signal = prediction_error_mutation_signals(
        _self_state({"continuity_pressure": 0.3}), min_error=0.3
    )[0]

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
    signal = prediction_error_mutation_signals(
        _self_state({"continuity_pressure": 0.3}), min_error=0.3
    )[0]

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
    signal = prediction_error_mutation_signals(
        _self_state({"continuity_pressure": 0.3}), min_error=0.3
    )[0]

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
