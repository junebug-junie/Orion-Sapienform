from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os

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
    budget: AdaptationCycleBudget = AdaptationCycleBudget()
    kill_switch_env: str = "SUBSTRATE_MUTATION_AUTONOMY_ENABLED"
    lock_id: int = 914257

    def run_cycle(
        self,
        *,
        telemetry: list[GraphReviewTelemetryRecordV1],
        measured_metrics_by_proposal: dict[str, dict[str, float]],
        post_adoption_delta_by_proposal: dict[str, float] | None = None,
        now: datetime | None = None,
    ) -> dict[str, int | list[str]]:
        t = now or datetime.now(timezone.utc)
        if not self._enabled():
            return {"signals": 0, "proposals": 0, "trials": 0, "adoptions": 0, "notes": ["autonomy_kill_switch_disabled"]}
        if not self._acquire_leader_lock():
            return {"signals": 0, "proposals": 0, "trials": 0, "adoptions": 0, "notes": ["leader_lock_not_acquired"]}

        notes: list[str] = []
        signals = self.detectors.from_review_telemetry(telemetry[: self.budget.max_signals])
        for signal in signals:
            self.store.record_signal(signal)
            key = f"{signal.anchor_scope}|{signal.subject_ref}|{signal.target_surface}"
            existing = self.store._pressures.get(key)  # bounded internal state access
            updated = self.pressure.apply(current=existing, signal=signal, now=t)
            self.store.record_pressure(updated)

        proposal_count = 0
        for pressure in list(self.store._pressures.values()):
            if proposal_count >= self.budget.max_proposals:
                break
            if not self.pressure.ready_for_proposal(pressure, now=t):
                continue
            proposal = self.proposals.from_pressure(pressure)
            if proposal is None:
                continue
            self.store.add_proposal(proposal, priority=60)
            proposal_count += 1

        trial_count = 0
        adoption_count = 0
        for queue_item in self.store.list_due_queue(now=t, limit=self.budget.max_trials):
            proposal = self.store.get_proposal(queue_item.proposal_id)
            if proposal is None:
                continue
            metrics = measured_metrics_by_proposal.get(proposal.proposal_id, {})
            trial = self.trial_runner.run_trial(proposal=proposal, measured_metrics=metrics)
            self.store.record_trial(trial)
            trial_count += 1
            has_replay = self.trial_runner.corpus_registry.ready_for_class(proposal.mutation_class)
            decision: MutationDecisionV1 = self.decision_engine.decide(
                proposal=proposal,
                trial=trial,
                has_replay_and_baseline=has_replay,
                active_surface_exists=self.store.active_surface(proposal.target_surface) is not None,
            )
            self.store.record_decision(decision)
            adoption = self.applier.apply(proposal=proposal, decision=decision)
            if adoption is not None:
                warnings = self.store.record_adoption(adoption)
                if warnings:
                    notes.extend(warnings)
                else:
                    adoption_count += 1
            if adoption_count >= self.budget.max_adoptions:
                notes.append("adoption_budget_reached")
                break

        for adoption in list(self.store._adoptions.values()):
            delta = (post_adoption_delta_by_proposal or {}).get(adoption.proposal_id)
            if delta is None:
                continue
            if self.monitor.should_rollback(delta_score=delta):
                rollback = self.monitor.build_rollback(adoption=adoption, reason=f"post_adoption_regression:{delta:.4f}")
                self.applier.rollback(adoption=adoption)
                self.store.record_rollback(rollback)
                notes.append(f"rolled_back:{adoption.proposal_id}")

        return {
            "signals": len(signals),
            "proposals": proposal_count,
            "trials": trial_count,
            "adoptions": adoption_count,
            "notes": notes,
        }

    def _enabled(self) -> bool:
        return str(os.getenv(self.kill_switch_env, "false")).strip().lower() in {"1", "true", "yes", "on"}

    def _acquire_leader_lock(self) -> bool:
        if not self.store.postgres_url:
            return True
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(self.store.postgres_url)
            with engine.begin() as conn:
                row = conn.execute(text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": self.lock_id}).fetchone()
                return bool(row and row[0])
        except Exception:
            return False


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
