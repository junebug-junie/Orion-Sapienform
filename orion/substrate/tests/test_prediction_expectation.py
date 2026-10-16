"""Rung 6: Orion revising their own surprise settings.

Pure scoring (prediction_expectation.py) plus the full governed loop through the
real SubstrateAdaptationWorker: tuner proposes -> trial on held-out readings ->
auto-promote -> live control surface moves -> post-adoption replay -> rollback.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from orion.bus.ewma import compute_ewma_update
from orion.core.schemas.substrate_mutation import MutationPatchV1, MutationProposalV1
from orion.substrate import mutation_control_surface
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_control_surface import (
    RuntimeControlSurfaceStore,
    get_prediction_expectation,
    set_prediction_expectation,
)
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_pressure import PressureAccumulator
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner
from orion.substrate.mutation_worker import SubstrateAdaptationWorker
from orion.substrate.prediction_error import execution_prediction_error, execution_raw_reading
from orion.substrate.prediction_expectation import (
    WARMUP_READINGS,
    Expectation,
    assess,
    candidate_grid,
    domain_specs,
    holdout_gain,
    paired_gain,
    per_reading_log_scores,
)
from orion.substrate.prediction_expectation_tuner import PredictionExpectationTuner

# Anchored to the wall clock: queue items come due at real time (add_proposal).
T0 = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=2400)
SPEC = domain_specs()["execution"]
DEFAULT = SPEC.default


def _iid(n: int, *, seed: int, mu: float = 0.02, sd: float = 0.005) -> list[float]:
    r = random.Random(seed)
    return [abs(mu + r.gauss(0, sd)) for _ in range(n)]


def _drift(n: int, *, seed: int) -> list[float]:
    r = random.Random(seed)
    level, out = 0.02, []
    for _ in range(n):
        level = max(0.0001, level + r.gauss(0, 0.003))
        out.append(abs(level + r.gauss(0, 0.0005)))
    return out


def _timed(values: list[float], *, start: datetime = T0) -> list[tuple[datetime, float]]:
    return [(start + timedelta(minutes=i), v) for i, v in enumerate(values)]


# --- pure scoring ---------------------------------------------------------


def test_replay_reproduces_the_live_ewma_update() -> None:
    values = _iid(50, seed=1)
    exp = Expectation(alpha=0.2, min_variance=1e-6)
    scores = per_reading_log_scores(values, exp)
    ewma, var, n = 0.0, 0.0, 0
    for value, score in zip(values, scores):
        update = compute_ewma_update(prev_ewma=ewma, prev_variance=var, prev_count=n, value=value, alpha=0.2, min_variance=1e-6)
        if n == 0:
            assert score is None
        else:
            # z from the live helper and the replay's own deviation agree.
            spread = max(var, 1e-6)
            assert update.zscore == pytest.approx((value - ewma) / spread**0.5)
        ewma, var, n = update.ewma, update.variance, n + 1


def test_going_numb_loses_under_the_log_score() -> None:
    """A huge variance floor makes nothing surprising; a proper score must not reward it."""
    values = _iid(600, seed=2)
    gain = paired_gain(values, candidate=Expectation(0.2, 1.0), reference=DEFAULT, score_from=WARMUP_READINGS)
    assert gain is not None and gain.upper_bound < 0.0


def test_assess_finds_a_better_setting_and_it_survives_holdout() -> None:
    readings = _timed(_iid(1500, seed=2))
    a = assess(SPEC, readings, current=DEFAULT)
    assert a is not None
    assert a.candidate.alpha < DEFAULT.alpha  # steady data: slower tracking is better calibrated
    assert a.candidate.min_variance == DEFAULT.min_variance  # floor never binds -> not changed
    assert a.fit_gain.lower_bound > 0.0
    held = holdout_gain(readings, candidate=a.candidate, reference=DEFAULT, after=a.fit_window_end, min_readings=150)
    assert held is not None and held.lower_bound > 0.0


def test_holdout_rejects_a_candidate_that_only_fit_the_past() -> None:
    readings = _timed(_drift(2500, seed=3))
    a = assess(SPEC, readings, current=DEFAULT)
    assert a is not None
    held = holdout_gain(readings, candidate=a.candidate, reference=DEFAULT, after=a.fit_window_end, min_readings=150)
    assert held is not None and held.lower_bound <= 0.0


def test_too_little_history_proposes_nothing() -> None:
    assert assess(SPEC, _timed(_iid(400, seed=4)), current=DEFAULT) is None


def test_candidates_stay_inside_domain_bounds() -> None:
    for candidate in candidate_grid(SPEC, DEFAULT):
        assert SPEC.in_bounds(candidate)
        assert candidate != DEFAULT


def test_execution_prediction_error_uses_the_supplied_expectation() -> None:
    from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1

    def projection(load: float) -> ExecutionTrajectoryProjectionV1:
        return ExecutionTrajectoryProjectionV1.model_validate(
            {
                "projection_id": "p",
                "generated_at": T0,
                "runs": {
                    "t": {
                        "trace_id": "t",
                        "correlation_id": "c",
                        "node_id": "athena",
                        "pressure_hints": {"reasoning_load": load},
                        "last_updated_at": T0,
                    }
                },
                "prediction_error_baseline_ewma": 0.001,
                "prediction_error_baseline_ewma_var": 1e-6,
                "prediction_error_baseline_ewma_n": 10,
            }
        )

    prev = projection(0.0)
    assert execution_raw_reading(prev, projection(0.02)) == pytest.approx(0.005)  # 1 of 4 hint keys moved
    tight = execution_prediction_error(prev, projection(0.02), expectation=Expectation(0.2, 1e-10))
    loose = execution_prediction_error(prev, projection(0.02), expectation=Expectation(0.2, 1e-3))
    assert tight > loose


# --- control surface ------------------------------------------------------


@pytest.fixture
def isolated_surface(tmp_path, monkeypatch):
    store = RuntimeControlSurfaceStore(sql_db_path=str(tmp_path / "control.sqlite3"))
    assert store.source_kind() == "sqlite"
    monkeypatch.setattr(mutation_control_surface, "_CONTROL_SURFACE_STORE", store)
    return store


def test_unset_or_out_of_bounds_surface_reads_as_the_calibrated_default(isolated_surface) -> None:
    assert get_prediction_expectation("execution") == DEFAULT
    isolated_surface.upsert(key=SPEC.control_surface_key, value={"alpha": 0.99, "min_variance": 1e-10})
    assert get_prediction_expectation("execution") == DEFAULT
    with pytest.raises(mutation_control_surface.ControlSurfaceWriteError):
        set_prediction_expectation("execution", alpha=0.99, min_variance=1e-10, actor="test")


# --- full governed loop ---------------------------------------------------


class _Readings:
    def __init__(self, readings: list[tuple[datetime, float]]) -> None:
        self.readings = readings

    def __call__(self, domain: str, since: datetime) -> list[tuple[datetime, float]]:
        if domain != "execution":
            return []
        return [r for r in self.readings if r[0] >= since]


def _worker(store: SubstrateMutationStore, tuner: PredictionExpectationTuner) -> SubstrateAdaptationWorker:
    corpus = ReplayCorpusRegistry(
        corpus_by_class={"prediction_expectation_patch": "replay-prediction-raw-reading-v1"},
        baseline_metric_ref_by_class={"prediction_expectation_patch": "baseline-prediction-expectation-live-v1"},
    )
    return SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(),
        proposals=ProposalFactory(),
        trial_runner=SubstrateTrialRunner(
            scorer=ClassSpecificScorer(),
            corpus_registry=corpus,
            prediction_expectation_evaluator=tuner.trial_metrics,
        ),
        decision_engine=DecisionEngine(),
        applier=PatchApplier(surfaces={}),
        monitor=PostAdoptionMonitor(regression_threshold=-0.05),
    )


def test_orion_adopts_a_better_expectation_then_rolls_it_back_when_the_world_changes(
    isolated_surface, monkeypatch
) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    history = _iid(1500, seed=2)
    reader = _Readings(_timed(history))
    tuner = PredictionExpectationTuner(read_readings=reader, read_live=get_prediction_expectation)
    store = SubstrateMutationStore()
    now = T0 + timedelta(minutes=1500)

    proposals = tuner.propose(store=store, now=now)
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.patch.rollback_payload == DEFAULT.as_patch()
    assert proposal.patch.patch != DEFAULT.as_patch()  # derived from evidence, not a constant
    # Rate-limited and one-at-a-time: a second ask in the same hour adds nothing.
    assert tuner.propose(store=store, now=now + timedelta(minutes=1)) == []

    # The queue item came due at the wall-clock moment it was enqueued.
    tick = datetime.now(timezone.utc)
    result = _worker(store, tuner).run_cycle(telemetry=[], measured_metrics_by_proposal={}, now=tick)
    assert result["adoptions"] == 1, (result, store.latest_trials_by_proposal())
    trial = store.latest_trials_by_proposal()[proposal.proposal_id]
    assert trial.status == "passed" and trial.metrics["holdout_log_score_gain_lower_bound"] > 0.0
    adopted = get_prediction_expectation("execution")
    assert adopted.as_patch() == proposal.patch.patch
    assert store.active_surface(SPEC.target_surface) is not None

    # The world stops being steady and starts drifting: the slow tracking Orion
    # adopted is now confidently worse than the setting it replaced.
    adoption = next(iter(store._adoptions.values()))
    reader.readings += _timed(_drift(600, seed=11), start=adoption.applied_at + timedelta(seconds=1))
    deltas = tuner.post_adoption_deltas(store=store)
    assert deltas[proposal.proposal_id] <= -0.05, deltas
    result = _worker(store, tuner).run_cycle(
        telemetry=[],
        measured_metrics_by_proposal={},
        post_adoption_delta_by_proposal=deltas,
        now=tick + timedelta(minutes=5),
    )
    assert f"rolled_back:{proposal.proposal_id}" in result["notes"]
    assert get_prediction_expectation("execution") == DEFAULT
    assert store.active_surface(SPEC.target_surface) is None


def test_trial_without_enough_holdout_is_inconclusive_not_passed(isolated_surface) -> None:
    tuner = PredictionExpectationTuner(read_readings=lambda d, s: [], read_live=get_prediction_expectation)
    proposal = MutationProposalV1(
        mutation_class="prediction_expectation_patch",
        risk_tier="low",
        target_surface=SPEC.target_surface,
        anchor_scope="orion",
        subject_ref="entity:orion",
        evidence_refs=["x"],
        source_signal_ids=["s"],
        source_pressure_id="direct-evidence:s",
        patch=MutationPatchV1(
            mutation_class="prediction_expectation_patch",
            target_surface=SPEC.target_surface,
            target_ref="execution",
            patch={"alpha": 0.1, "min_variance": DEFAULT.min_variance},
            rollback_payload=DEFAULT.as_patch(),
        ),
        notes=[f"fit_window_start:{T0.isoformat()}", f"fit_window_end:{T0.isoformat()}"],
    )
    runner = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={"prediction_expectation_patch": "c"},
            baseline_metric_ref_by_class={"prediction_expectation_patch": "b"},
        ),
        prediction_expectation_evaluator=tuner.trial_metrics,
    )
    assert runner.run_trial(proposal=proposal, measured_metrics={}).status == "inconclusive"


def test_graph_consolidation_is_retired_and_cannot_be_decided() -> None:
    proposal = MutationProposalV1(
        mutation_class="graph_consolidation_param_patch",
        risk_tier="medium",
        target_surface="graph_consolidation",
        anchor_scope="orion",
        subject_ref="orion",
        evidence_refs=["x"],
        source_signal_ids=["s"],
        source_pressure_id="p",
        patch=MutationPatchV1(
            mutation_class="graph_consolidation_param_patch",
            target_surface="graph_consolidation",
            target_ref="graph_consolidation",
            patch={"query_limit_nodes": 96},
            rollback_payload={"query_limit_nodes": 64},
        ),
    )
    decision = DecisionEngine().decide(proposal=proposal, trial=None, has_replay_and_baseline=True, active_surface_exists=False)
    assert decision.action == "reject" and decision.reason == "mutation_class_retired"


def test_world_ontology_review_telemetry_no_longer_produces_signals() -> None:
    from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1

    record = GraphReviewTelemetryRecordV1.model_validate(
        {
            "invocation_surface": "operator_review",
            "execution_outcome": "executed",
            "selection_reason": "test",
            "runtime_duration_ms": 10,
            "anchor_scope": "orion",
            "subject_ref": "orion",
            "target_zone": "world_ontology",
        }
    )
    assert MutationDetectors().from_review_telemetry([record]) == []


def _expectation_proposal(alpha: float = 0.1) -> MutationProposalV1:
    return MutationProposalV1(
        mutation_class="prediction_expectation_patch",
        risk_tier="low",
        target_surface=SPEC.target_surface,
        anchor_scope="orion",
        subject_ref="entity:orion",
        evidence_refs=["x"],
        source_signal_ids=["s"],
        source_pressure_id="direct-evidence:s",
        patch=MutationPatchV1(
            mutation_class="prediction_expectation_patch",
            target_surface=SPEC.target_surface,
            target_ref="execution",
            patch={"alpha": alpha, "min_variance": DEFAULT.min_variance},
            rollback_payload=DEFAULT.as_patch(),
        ),
        notes=[f"fit_window_start:{T0.isoformat()}", f"fit_window_end:{T0.isoformat()}"],
    )


def test_apply_refuses_when_the_live_value_cannot_be_read(isolated_surface, monkeypatch) -> None:
    """A failed read must not be recorded as "live was the default" -- that
    would make a later rollback restore the wrong value."""
    from orion.core.schemas.substrate_mutation import MutationDecisionV1

    def broken(key):
        raise mutation_control_surface.ControlSurfaceReadError("db blip")

    monkeypatch.setattr(isolated_surface, "read", broken)
    proposal = _expectation_proposal()
    decision = MutationDecisionV1(proposal_id=proposal.proposal_id, action="auto_promote", reason="test")
    assert PatchApplier(surfaces={}).apply(proposal=proposal, decision=decision) is None


def test_a_failed_trial_read_is_inconclusive_not_a_crash(isolated_surface) -> None:
    def broken(domain, since):
        raise RuntimeError("relation does not exist")

    tuner = PredictionExpectationTuner(read_readings=broken, read_live=get_prediction_expectation)
    runner = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={"prediction_expectation_patch": "c"},
            baseline_metric_ref_by_class={"prediction_expectation_patch": "b"},
        ),
        prediction_expectation_evaluator=tuner.trial_metrics,
    )
    assert runner.run_trial(proposal=_expectation_proposal(), measured_metrics={}).status == "inconclusive"


def test_a_confident_but_tiny_gain_does_not_pass_the_trial() -> None:
    status, notes = ClassSpecificScorer().evaluate(
        mutation_class="prediction_expectation_patch",
        metrics={"holdout_log_score_gain_lower_bound": 0.001, "holdout_log_score_gain_mean": 0.004},
    )
    assert status == "failed" and "metric_failed:holdout_log_score_gain_mean" in notes


def test_proposals_for_a_domain_are_spaced_a_day_apart(isolated_surface) -> None:
    reader = _Readings(_timed(_iid(1500, seed=2)))
    tuner = PredictionExpectationTuner(read_readings=reader, read_live=get_prediction_expectation)
    store = SubstrateMutationStore()
    now = T0 + timedelta(minutes=1500)
    first = tuner.propose(store=store, now=now)
    assert len(first) == 1
    store.set_queue_status(store.queue_item_id_for_proposal(first[0].proposal_id), "rejected")
    assert tuner.propose(store=store, now=now + timedelta(hours=2)) == []
    assert len(tuner.propose(store=store, now=now + timedelta(hours=25))) == 1
