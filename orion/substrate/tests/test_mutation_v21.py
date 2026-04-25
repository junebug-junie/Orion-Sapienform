from __future__ import annotations

import os
from pathlib import Path

from orion.core.schemas.substrate_mutation import MutationDecisionV1, MutationPressureV1
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
