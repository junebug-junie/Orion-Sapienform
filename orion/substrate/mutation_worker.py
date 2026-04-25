from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from typing import Any, Callable
from uuid import uuid4

from orion.core.schemas.substrate_mutation import MutationDecisionV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_pressure import PressureAccumulator
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner
from orion.substrate.mutation_scoring import ClassSpecificScorer


@dataclass
class AdaptationCycleBudget:
    max_signals: int = 64
    max_proposals: int = 8
    max_trials: int = 8
    max_adoptions: int = 2


@dataclass
class SubstrateAdaptationWorker:
    store: SubstrateMutationStore
    detectors: MutationDetectors
    pressure: PressureAccumulator
    proposals: ProposalFactory
    trial_runner: SubstrateTrialRunner
    decision_engine: DecisionEngine
    applier: PatchApplier
    monitor: PostAdoptionMonitor
    budget: AdaptationCycleBudget = field(default_factory=AdaptationCycleBudget)
    kill_switch_env: str = "SUBSTRATE_MUTATION_AUTONOMY_ENABLED"
    lock_id: int = 914257
    trace_logger: Callable[[dict[str, Any]], None] | None = None

    def run_cycle(
        self,
        *,
        telemetry: list[GraphReviewTelemetryRecordV1],
        measured_metrics_by_proposal: dict[str, dict[str, float]],
        measured_metrics_by_class: dict[str, dict[str, float]] | None = None,
        post_adoption_delta_by_proposal: dict[str, float] | None = None,
        post_adoption_delta_by_target_surface: dict[str, float] | None = None,
        replay_telemetry: list[GraphReviewTelemetryRecordV1] | None = None,
        now: datetime | None = None,
    ) -> dict[str, int | list[str]]:
        t = now or datetime.now(timezone.utc)
        cycle_id = f"mutation-cycle-{uuid4()}"
        if not self._enabled():
            return {"signals": 0, "proposals": 0, "trials": 0, "adoptions": 0, "notes": ["autonomy_kill_switch_disabled"]}
        lock_ctx = self._acquire_leader_lock()
        acquired = bool(lock_ctx.get("acquired"))
        self._trace(event="mutation_lock_acquire", cycle_id=cycle_id, lock_acquired=acquired)
        if not acquired:
            return {"signals": 0, "proposals": 0, "trials": 0, "adoptions": 0, "notes": ["leader_lock_not_acquired"]}

        notes: list[str] = []
        try:
            self._trace(event="mutation_cycle_start", cycle_id=cycle_id, lock_acquired=True)
            signals = self.detectors.from_review_telemetry(telemetry[: self.budget.max_signals])
            for signal in signals:
                self._trace(
                    event="mutation_signal_emitted",
                    cycle_id=cycle_id,
                    signal_id=signal.signal_id,
                    lineage_id=signal.signal_id,
                    surface_key=signal.target_surface,
                    notes=[signal.event_kind],
                )
                self.store.record_signal(signal)
                key = f"{signal.anchor_scope}|{signal.subject_ref}|{signal.target_surface}"
                existing = self.store._pressures.get(key)  # bounded internal state access
                updated = self.pressure.apply(current=existing, signal=signal, now=t)
                self.store.record_pressure(updated)
                self._trace(
                    event="mutation_pressure_recorded",
                    cycle_id=cycle_id,
                    signal_id=signal.signal_id,
                    lineage_id=(updated.source_signal_ids[0] if updated.source_signal_ids else signal.signal_id),
                    pressure_key=key,
                    surface_key=signal.target_surface,
                    notes=[signal.event_kind],
                )

            proposal_count = 0
            for pressure in list(self.store._pressures.values()):
                if proposal_count >= self.budget.max_proposals:
                    break
                ready = self.pressure.ready_for_proposal(pressure, now=t)
                if ready:
                    self._trace(
                        event="mutation_pressure_threshold_crossed",
                        cycle_id=cycle_id,
                        pressure_id=pressure.pressure_id,
                        lineage_id=(pressure.source_signal_ids[0] if pressure.source_signal_ids else pressure.pressure_id),
                        pressure_key=self.store.pressure_key_for(
                            anchor_scope=pressure.anchor_scope,
                            subject_ref=pressure.subject_ref,
                            target_surface=pressure.target_surface,
                        ),
                        surface_key=pressure.target_surface,
                        notes=[f"score={pressure.pressure_score:.4f}"],
                    )
                if not ready:
                    continue
                proposal = self.proposals.from_pressure(pressure)
                if proposal is None:
                    continue
                queue_item = self.store.add_proposal(proposal, priority=60)
                cooled = self.pressure.mark_proposal_emitted(pressure, now=t)
                self.store.record_pressure(cooled)
                proposal_count += 1
                self._trace(
                    event="mutation_proposal_enqueued",
                    cycle_id=cycle_id,
                    queue_item_id=queue_item.queue_item_id,
                    proposal_id=proposal.proposal_id,
                    lineage_id=(proposal.source_signal_ids[0] if proposal.source_signal_ids else proposal.proposal_id),
                    pressure_key=self.store.pressure_key_for(
                        anchor_scope=proposal.anchor_scope,
                        subject_ref=proposal.subject_ref,
                        target_surface=proposal.target_surface,
                    ),
                    surface_key=proposal.target_surface,
                )

            trial_count = 0
            adoption_count = 0
            for queue_item in self.store.list_due_queue(now=t, limit=self.budget.max_trials):
                proposal = self.store.get_proposal(queue_item.proposal_id)
                if proposal is None:
                    continue
                self._trace(
                    event="mutation_trial_started",
                    cycle_id=cycle_id,
                    queue_item_id=queue_item.queue_item_id,
                    proposal_id=proposal.proposal_id,
                    lineage_id=(proposal.source_signal_ids[0] if proposal.source_signal_ids else proposal.proposal_id),
                    surface_key=proposal.target_surface,
                )
                metrics = measured_metrics_by_proposal.get(proposal.proposal_id)
                if metrics is None and measured_metrics_by_class is not None:
                    metrics = measured_metrics_by_class.get(proposal.mutation_class)
                if metrics is None:
                    metrics = {}
                trial = self.trial_runner.run_trial(
                    proposal=proposal,
                    measured_metrics=metrics,
                    replay_records=replay_telemetry or telemetry,
                )
                self.store.record_trial(trial)
                trial_count += 1
                self._trace(
                    event="mutation_trial_recorded",
                    cycle_id=cycle_id,
                    queue_item_id=queue_item.queue_item_id,
                    proposal_id=proposal.proposal_id,
                    trial_id=trial.trial_id,
                    lineage_id=(proposal.source_signal_ids[0] if proposal.source_signal_ids else proposal.proposal_id),
                    surface_key=proposal.target_surface,
                    notes=list(trial.notes),
                )
                has_replay = self.trial_runner.corpus_registry.ready_for_class(proposal.mutation_class)
                active_surface_exists = self.store.active_surface(proposal.target_surface) is not None
                queue_status_before = self.store.queue_status_for_proposal(proposal.proposal_id)
                decision: MutationDecisionV1 = self.decision_engine.decide(
                    proposal=proposal,
                    trial=trial,
                    has_replay_and_baseline=has_replay,
                    active_surface_exists=active_surface_exists,
                )
                self.store.record_decision(decision)
                queue_status_after = self.store.queue_status_for_proposal(proposal.proposal_id)
                self._trace(
                    event="mutation_decision_recorded",
                    cycle_id=cycle_id,
                    queue_item_id=queue_item.queue_item_id,
                    proposal_id=proposal.proposal_id,
                    trial_id=trial.trial_id,
                    lineage_id=(proposal.source_signal_ids[0] if proposal.source_signal_ids else proposal.proposal_id),
                    decision=decision.action,
                    queue_status_before=queue_status_before,
                    queue_status_after=queue_status_after,
                    surface_key=proposal.target_surface,
                    applied=False,
                    notes=list(decision.notes),
                )
                if queue_status_before != queue_status_after:
                    self._trace(
                        event="mutation_queue_transition",
                        cycle_id=cycle_id,
                        queue_item_id=queue_item.queue_item_id,
                        proposal_id=proposal.proposal_id,
                        lineage_id=(proposal.source_signal_ids[0] if proposal.source_signal_ids else proposal.proposal_id),
                        queue_status_before=queue_status_before,
                        queue_status_after=queue_status_after,
                        surface_key=proposal.target_surface,
                    )
                if decision.action != "auto_promote":
                    continue
                # Re-check one-live-mutation-per-surface before apply to avoid side effects when invariant is violated.
                if self.store.active_surface(proposal.target_surface) is not None:
                    notes.append("active_mutation_exists_for_target_surface")
                    self.store.record_apply_blocked(
                        proposal_id=proposal.proposal_id,
                        decision_id=decision.decision_id,
                        target_surface=proposal.target_surface,
                        reason="active_surface",
                        notes=["active_mutation_exists_for_target_surface"],
                        queue_status=self.store.queue_status_for_proposal(proposal.proposal_id),
                    )
                    self._trace(
                        event="mutation_apply_blocked",
                        cycle_id=cycle_id,
                        queue_item_id=queue_item.queue_item_id,
                        proposal_id=proposal.proposal_id,
                        lineage_id=(proposal.source_signal_ids[0] if proposal.source_signal_ids else proposal.proposal_id),
                        decision=decision.action,
                        surface_key=proposal.target_surface,
                        blocked_reason="active_surface",
                        applied=False,
                    )
                    continue
                adoption = self.applier.apply(proposal=proposal, decision=decision)
                if adoption is not None:
                    warnings = self.store.record_adoption(adoption)
                    if warnings:
                        notes.extend(warnings)
                        self.store.record_apply_blocked(
                            proposal_id=proposal.proposal_id,
                            decision_id=decision.decision_id,
                            target_surface=proposal.target_surface,
                            reason=";".join(warnings),
                            notes=list(warnings),
                            queue_status=self.store.queue_status_for_proposal(proposal.proposal_id),
                        )
                        self._trace(
                            event="mutation_apply_blocked",
                            cycle_id=cycle_id,
                            queue_item_id=queue_item.queue_item_id,
                            proposal_id=proposal.proposal_id,
                            lineage_id=(proposal.source_signal_ids[0] if proposal.source_signal_ids else proposal.proposal_id),
                            decision=decision.action,
                            surface_key=proposal.target_surface,
                            blocked_reason=";".join(warnings),
                            applied=False,
                        )
                    else:
                        adoption_count += 1
                        self._trace(
                            event="mutation_apply_succeeded",
                            cycle_id=cycle_id,
                            queue_item_id=queue_item.queue_item_id,
                            proposal_id=proposal.proposal_id,
                            lineage_id=(proposal.source_signal_ids[0] if proposal.source_signal_ids else proposal.proposal_id),
                            decision=decision.action,
                            surface_key=proposal.target_surface,
                            queue_status_after=self.store.queue_status_for_proposal(proposal.proposal_id),
                            applied=True,
                        )
                        self._trace(
                            event="mutation_monitoring_window_opened",
                            cycle_id=cycle_id,
                            proposal_id=proposal.proposal_id,
                            lineage_id=(proposal.source_signal_ids[0] if proposal.source_signal_ids else proposal.proposal_id),
                            surface_key=proposal.target_surface,
                            notes=[f"rollback_window_sec={adoption.rollback_window_sec}"],
                        )
                if adoption_count >= self.budget.max_adoptions:
                    notes.append("adoption_budget_reached")
                    break

            for adoption in list(self.store._adoptions.values()):
                adopted_proposal = self.store.get_proposal(adoption.proposal_id)
                adopted_lineage_id = (
                    adopted_proposal.source_signal_ids[0]
                    if adopted_proposal is not None and adopted_proposal.source_signal_ids
                    else adoption.proposal_id
                )
                delta = (post_adoption_delta_by_proposal or {}).get(adoption.proposal_id)
                if delta is None and post_adoption_delta_by_target_surface is not None:
                    delta = post_adoption_delta_by_target_surface.get(adoption.target_surface)
                if delta is None:
                    continue
                self._trace(
                    event="mutation_monitoring_checked",
                    cycle_id=cycle_id,
                    proposal_id=adoption.proposal_id,
                    lineage_id=adopted_lineage_id,
                    surface_key=adoption.target_surface,
                    notes=[f"delta_score={delta:.4f}"],
                )
                if self.monitor.should_rollback(delta_score=delta):
                    self._trace(
                        event="mutation_rollback_triggered",
                        cycle_id=cycle_id,
                        proposal_id=adoption.proposal_id,
                        lineage_id=adopted_lineage_id,
                        surface_key=adoption.target_surface,
                        notes=[f"delta_score={delta:.4f}"],
                    )
                    rollback = self.monitor.build_rollback(adoption=adoption, reason=f"post_adoption_regression:{delta:.4f}")
                    self.applier.rollback(adoption=adoption)
                    self.store.record_rollback(rollback)
                    notes.append(f"rolled_back:{adoption.proposal_id}")
                    self._trace(
                        event="mutation_rollback_recorded",
                        cycle_id=cycle_id,
                        proposal_id=adoption.proposal_id,
                        rollback_id=rollback.rollback_id,
                        lineage_id=adopted_lineage_id,
                        surface_key=adoption.target_surface,
                        notes=[rollback.reason],
                    )

            result = {
                "signals": len(signals),
                "proposals": proposal_count,
                "trials": trial_count,
                "adoptions": adoption_count,
                "notes": notes,
            }
            self._trace(event="mutation_cycle_complete", cycle_id=cycle_id, lock_acquired=True, notes=notes)
            return result
        finally:
            self._release_leader_lock(lock_ctx)
            self._trace(event="mutation_lock_released", cycle_id=cycle_id, lock_released=True)

    def _enabled(self) -> bool:
        return str(os.getenv(self.kill_switch_env, "false")).strip().lower() in {"1", "true", "yes", "on"}

    def _acquire_leader_lock(self) -> dict[str, Any]:
        if not self.store.postgres_url:
            return {"acquired": True, "engine": None, "conn": None}
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(self.store.postgres_url)
            conn = engine.connect()
            row = conn.execute(text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": self.lock_id}).fetchone()
            acquired = bool(row and row[0])
            if not acquired:
                conn.close()
                engine.dispose()
            return {"acquired": acquired, "engine": engine, "conn": conn if acquired else None}
        except Exception:
            return {"acquired": False, "engine": None, "conn": None}

    def _release_leader_lock(self, lock_ctx: dict[str, Any]) -> None:
        conn = lock_ctx.get("conn")
        engine = lock_ctx.get("engine")
        if conn is not None:
            try:
                from sqlalchemy import text

                conn.execute(text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": self.lock_id})
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass

    def _trace(self, **fields: Any) -> None:
        if self.trace_logger is not None:
            self.trace_logger(fields)


def build_default_worker(*, store: SubstrateMutationStore) -> SubstrateAdaptationWorker:
    corpus = ReplayCorpusRegistry(
        corpus_by_class={
            "routing_threshold_patch": "replay-routing-v1",
            "recall_weighting_patch": "replay-recall-v1",
            "graph_consolidation_param_patch": "replay-consolidation-v1",
            "approved_prompt_profile_variant_promotion": "replay-prompt-profile-v1",
        },
        baseline_metric_ref_by_class={
            "routing_threshold_patch": "baseline-routing-v1",
            "recall_weighting_patch": "baseline-recall-v1",
            "graph_consolidation_param_patch": "baseline-consolidation-v1",
            "approved_prompt_profile_variant_promotion": "baseline-prompt-profile-v1",
        },
    )
    return SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(),
        proposals=ProposalFactory(),
        trial_runner=SubstrateTrialRunner(
            scorer=ClassSpecificScorer(),
            corpus_registry=corpus,
        ),
        decision_engine=DecisionEngine(),
        applier=PatchApplier(surfaces={}),
        monitor=PostAdoptionMonitor(),
    )
