from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import os
import sqlite3
import threading
from typing import Any

from orion.core.schemas.substrate_mutation import (
    CognitiveProposalDraftV1,
    CognitiveDraftRecommendationV1,
    CognitiveProposalReviewV1,
    CognitiveStanceNoteV1,
    MutationAdoptionV1,
    MutationDecisionV1,
    MutationPressureV1,
    MutationProposalV1,
    MutationQueueItemV1,
    MutationRollbackV1,
    MutationSignalV1,
    MutationTrialV1,
    RecallCanaryJudgmentRecordV1,
    RecallCanaryReviewArtifactV1,
    RecallCanaryRunV1,
    RecallProductionCandidateReviewV1,
    RecallShadowEvalRunV1,
    RecallStrategyProfileV1,
)
from orion.substrate.recall_strategy_readiness import readiness_for_pressure


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# 2026-09-04 (real-stakes gate, docs/superpowers/specs/2026-09-04-orion-
# emergent-choice-seams-brainstorm.md): a rolled-back mutation used to be a
# free retry -- the surface lock released the moment record_rollback() ran,
# so the same target_surface was eligible for a fresh proposal on the very
# next cycle regardless of what just went wrong. This multiplies the SAME
# rollback_window_sec the adoption already carried, rather than a fresh
# constant, so the penalty tracks whatever window that specific adoption
# declared for itself -- and scales with the risk_tier the proposal itself
# declared, so claiming "high" risk and being wrong costs more downtime than
# claiming "low" risk and being wrong. Disclosed, uncalibrated first-cut
# values -- same convention as every other first-cut constant in this
# codebase (see orion/proposals/scoring.py's DIMENSION_PRECISION_MIN_VARIANCE
# comments) -- revisit against real post-deploy cooldown/re-proposal data.
_ROLLBACK_COOLDOWN_MULTIPLIER: dict[str, float] = {
    "low": 1.0,
    "medium": 3.0,
    "high": 8.0,
}

# Below this many resolved (settled + rolled_back) adoptions on a surface,
# surface_reliability() returns None rather than a noisy ratio -- same "don't
# report a fabricated mid-range confidence during cold start" discipline as
# orion/proposals/scoring.py::dimension_confidence().
#
# CLAUDE.md 0A metric-quality gate, run before this was wired into
# ProposalFactory.plan_for_pressure() (2026-09-04):
#   1. Provenance: reads MutationAdoptionV1.status, set only by
#      record_settlement() ("settled") and record_rollback() ("rolled_back")
#      in this same file -- both real, already-exercised code paths, not
#      assumed.
#   2. Independence: NOT fully independent of the other signals already
#      feeding DecisionEngine.decide() (active_surface_exists,
#      has_replay_and_baseline) -- a surface that fails trials will tend to
#      also accumulate rollbacks, so these correlate in practice. They are
#      not the same measurement, though: active_surface_exists is a
#      real-time lock state, has_replay_and_baseline is about trial data
#      availability, and this is a retrospective outcome ratio over resolved
#      adoptions -- a different pipeline stage, not a relabeling of either
#      existing signal. Flagged as a real, only-partial independence, not
#      claimed clean.
#   3. Theory anchor: Laplace's rule of succession (the (settled+1)/
#      (settled+rolled_back+2) smoothing below) -- a track record of past
#      resolved outcomes is the standard prior for a future one, same
#      family of justification as this file's precision-weighted signals
#      elsewhere in the repo.
#   4. Live-data sanity, checked 2026-09-04 against real Postgres
#      (substrate_mutation_adoption): 6 real rows, all target_surface
#      "routing", all status "settled". Not degenerate -- above
#      SURFACE_RELIABILITY_MIN_SAMPLES and yields a real, non-trivial value
#      (7/8 = 0.875, comfortably above the 0.34 floor). Every OTHER surface
#      has zero resolved adoptions -- cold start (None) is the honest,
#      universal live state outside "routing" today. Caveat: "routing" is
#      currently parked (mutation_proposals.py's _PARKED_MUTATION_CLASSES),
#      so this reader is not actually exercised as a live gating decision
#      yet -- the number is real, but nothing is currently deciding off it.
#   5. Existing mechanism: searched -- no existing per-surface historical
#      reliability tracking anywhere in orion/substrate/ before this.
#   6. Reversibility: no new schema, no new persisted state -- both
#      rollback_cooldown_until() and surface_reliability() are pure
#      functions over already-persisted records. Deleting them and their 3
#      call sites (record_adoption(), mutation_worker.py's pre-apply check,
#      ProposalFactory.plan_for_pressure()) fully reverts this.
SURFACE_RELIABILITY_MIN_SAMPLES = 3


@dataclass
class SubstrateMutationStore:
    sql_db_path: str | None = None
    postgres_url: str | None = None
    _source_kind: str = field(default="memory", init=False)
    _last_error: str | None = field(default=None, init=False)
    _signals: list[MutationSignalV1] = field(default_factory=list, init=False)
    _pressures: dict[str, MutationPressureV1] = field(default_factory=dict, init=False)
    _proposals: dict[str, MutationProposalV1] = field(default_factory=dict, init=False)
    _queue: dict[str, MutationQueueItemV1] = field(default_factory=dict, init=False)
    _trials: dict[str, MutationTrialV1] = field(default_factory=dict, init=False)
    _decisions: dict[str, MutationDecisionV1] = field(default_factory=dict, init=False)
    _adoptions: dict[str, MutationAdoptionV1] = field(default_factory=dict, init=False)
    _rollbacks: dict[str, MutationRollbackV1] = field(default_factory=dict, init=False)
    _active_surface_by_target: dict[str, str] = field(default_factory=dict, init=False)
    _blocked_applies: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _cognitive_reviews: dict[str, CognitiveProposalReviewV1] = field(default_factory=dict, init=False)
    _cognitive_drafts: dict[str, CognitiveDraftRecommendationV1] = field(default_factory=dict, init=False)
    _cognitive_proposal_drafts: dict[str, CognitiveProposalDraftV1] = field(default_factory=dict, init=False)
    _cognitive_stance_notes: dict[str, CognitiveStanceNoteV1] = field(default_factory=dict, init=False)
    _recall_strategy_profiles: dict[str, RecallStrategyProfileV1] = field(default_factory=dict, init=False)
    _recall_shadow_eval_runs: dict[str, RecallShadowEvalRunV1] = field(default_factory=dict, init=False)
    _recall_production_candidate_reviews: dict[str, RecallProductionCandidateReviewV1] = field(default_factory=dict, init=False)
    _recall_canary_runs: dict[str, RecallCanaryRunV1] = field(default_factory=dict, init=False)
    #: Guards the in-memory dicts above against concurrent structural change.
    #: These are the live working set even when Postgres-backed -- Postgres is
    #: a mirror. Since 2026-09-03 the substrate mutation cycle runs on a worker
    #: thread (services/orion-hub/scripts/main.py) instead of inline on the
    #: event loop, so a writer here can now overlap a reader on the loop. An
    #: unlocked `sorted(self._proposals.values())` racing an insert raises
    #: RuntimeError: dictionary changed size during iteration -- which on the
    #: chat path (mutation_cognition_context) is a 500 on Orion's main chat
    #: endpoint. Reentrant because mutators call each other.
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _recall_canary_judgments: dict[str, RecallCanaryJudgmentRecordV1] = field(default_factory=dict, init=False)
    _recall_canary_review_artifacts: dict[str, RecallCanaryReviewArtifactV1] = field(default_factory=dict, init=False)
    _retention_max_blocked_applies: int = field(default=500, init=False)
    _retention_max_rollbacks: int = field(default=500, init=False)

    def __post_init__(self) -> None:
        self._retention_max_blocked_applies = self._env_int("SUBSTRATE_MUTATION_RETENTION_MAX_BLOCKED_APPLIES", 500, low=50, high=100000)
        self._retention_max_rollbacks = self._env_int("SUBSTRATE_MUTATION_RETENTION_MAX_ROLLBACKS", 500, low=50, high=100000)
        # Adoptions were self-limiting only by accident: the surface lock was
        # never released on success, so a surface could hold exactly one forever
        # and the live table had a single row. Now that a change settles and
        # hands the surface back, a surface can adopt roughly once per
        # rollback_window_sec -- and every _persist() re-upserts every adoption,
        # so unbounded growth would make each cycle progressively more expensive.
        self._retention_max_adoptions = self._env_int("SUBSTRATE_MUTATION_RETENTION_MAX_ADOPTIONS", 500, low=50, high=100000)
        if self.postgres_url:
            try:
                self._ensure_postgres_schema()
                self._load_from_postgres()
                self._source_kind = "postgres"
                return
            except Exception as exc:
                self._source_kind = "fallback"
                self._last_error = str(exc)
        if self.sql_db_path:
            self._ensure_sql_schema()
            self._load_from_sql()
            self._source_kind = "sqlite"

    def source_kind(self) -> str:
        return self._source_kind

    def degraded(self) -> bool:
        return self._source_kind == "fallback" or self._last_error is not None

    def last_error(self) -> str | None:
        return self._last_error

    def record_signal(self, signal: MutationSignalV1) -> None:
        self._signals.append(signal)
        if not self._persist_signal(signal):
            self._persist()

    def record_pressure(self, pressure: MutationPressureV1) -> None:
        # This is called once per signal in a mutation cycle (worker.run_cycle's
        # intake loop) -- 12+ times/cycle against a live instance -- to change
        # exactly one row in one table. `self._persist()` (full) instead
        # rewrites every row in all ~19 store-backed tables every time it's
        # called; with 17.8k accumulated signals that is the entire measured
        # 64-125s mutation-cycle cost (2026-09-03 diagnosis). Same
        # single-row-upsert-then-fallback shape as `record_signal`/
        # `_persist_signal` above, which already does this safely for the
        # store's biggest table. Deliberately narrow: this function and
        # `add_proposal` below only ever touch one or two single-purpose
        # tables each. Methods that fan out across multiple tables in one
        # call (record_trial, record_decision, record_adoption,
        # record_rollback, record_settlement, record_cognitive_review --
        # several of which also rewrite the active_surface lock table's
        # delete-then-reinsert semantics) are left on the full `_persist()`
        # path: they run far less often (147 rows total vs. 12+/cycle here),
        # and an incomplete incremental rewrite of one of those would be
        # invisible until the next restart reload, not now.
        key = self._pressure_key(pressure)
        self._pressures[key] = pressure
        if not self._persist_pressure(pressure):
            self._persist()

    def add_proposal(self, proposal: MutationProposalV1, *, priority: int = 50) -> MutationQueueItemV1:
        existing_queue_item = next((item for item in self._queue.values() if item.proposal_id == proposal.proposal_id), None)
        if existing_queue_item is not None:
            with self._lock:
                self._proposals[proposal.proposal_id] = proposal
            if not self._persist_proposal(proposal):
                self._persist()
            return existing_queue_item
        with self._lock:
            self._proposals[proposal.proposal_id] = proposal
        queue_item = MutationQueueItemV1(
            proposal_id=proposal.proposal_id,
            mutation_class=proposal.mutation_class,
            target_surface=proposal.target_surface,
            priority=priority,
        )
        self._queue[queue_item.queue_item_id] = queue_item
        if not self._persist_proposal_and_queue_item(proposal, queue_item):
            self._persist()
        return queue_item

    def list_due_queue(self, *, now: datetime | None = None, limit: int = 20) -> list[MutationQueueItemV1]:
        t = now or _utc_now()
        due = [item for item in self._queue.values() if item.due_at <= t and item.status == "queued"]
        due.sort(key=lambda item: (-item.priority, item.created_at))
        return due[:limit]

    def get_proposal(self, proposal_id: str) -> MutationProposalV1 | None:
        return self._proposals.get(proposal_id)

    def record_trial(self, trial: MutationTrialV1) -> None:
        with self._lock:
            self._trials[trial.trial_id] = trial
        proposal = self._proposals.get(trial.proposal_id)
        if proposal is not None:
            with self._lock:
                self._proposals[proposal.proposal_id] = proposal.model_copy(update={"rollout_state": "trialed"})
        self._set_queue_status_for_proposal(trial.proposal_id, "trialed")
        self._persist()

    def record_decision(self, decision: MutationDecisionV1) -> None:
        self._decisions[decision.decision_id] = decision
        proposal = self._proposals.get(decision.proposal_id)
        if proposal is not None:
            if decision.action == "auto_promote":
                next_state = "approved"
            elif decision.action == "require_review":
                next_state = "pending_review"
            elif decision.action == "hold":
                next_state = "trialed"
            else:
                next_state = "rejected"
            with self._lock:
                self._proposals[proposal.proposal_id] = proposal.model_copy(update={"rollout_state": next_state})
        if decision.action == "require_review":
            self._set_queue_status_for_proposal(decision.proposal_id, "pending_review")
        elif decision.action == "auto_promote":
            self._set_queue_status_for_proposal(decision.proposal_id, "approved")
        elif decision.action == "reject":
            self._set_queue_status_for_proposal(decision.proposal_id, "rejected")
        self._persist()

    def record_cognitive_review(self, review: CognitiveProposalReviewV1) -> CognitiveDraftRecommendationV1 | None:
        self._cognitive_reviews[review.review_id] = review
        proposal = self._proposals.get(review.proposal_id)
        if proposal is not None:
            with self._lock:
                self._proposals[proposal.proposal_id] = proposal.model_copy(update={"rollout_state": review.state})
            self._set_queue_status_for_proposal(proposal.proposal_id, review.state)
        created_draft: CognitiveDraftRecommendationV1 | None = None
        if review.state == "accepted_as_draft" and proposal is not None:
            created_draft = CognitiveDraftRecommendationV1(
                proposal_id=proposal.proposal_id,
                mutation_class=proposal.mutation_class,
                affected_surface=proposal.target_surface,
                pressure_kind=str(proposal.patch.patch.get("pressure_kind") or proposal.source_pressure_id),
                evidence_refs=list(proposal.evidence_refs),
                suggested_operator_action=str(proposal.patch.patch.get("suggested_operator_action") or "review"),
                blast_radius=str(proposal.patch.patch.get("blast_radius") or "bounded_cognitive_surface"),
                risk_tier=proposal.risk_tier,
                notes=[f"review_id:{review.review_id}", f"reviewer:{review.reviewer}"],
            )
            self._cognitive_drafts[created_draft.draft_id] = created_draft
            self._cognitive_proposal_drafts[created_draft.draft_id] = CognitiveProposalDraftV1(
                draft_id=created_draft.draft_id,
                proposal_id=proposal.proposal_id,
                proposal_class=proposal.mutation_class,
                title=str(proposal.patch.patch.get("title") or proposal.mutation_class),
                summary=str(proposal.patch.patch.get("summary") or proposal.rationale or "operator accepted cognitive draft"),
                draft_content=dict(proposal.patch.patch or {}),
                evidence_refs=list(proposal.evidence_refs),
                review_refs=[review.review_id],
                safety_scope={
                    "identity_kernel_rewrite_performed": False,
                    "production_self_model_rewrite_performed": False,
                    "policy_override_performed": False,
                    "freeform_prompt_self_rewrite_performed": False,
                    "live_apply_performed": False,
                },
                lineage={
                    "proposal_id": proposal.proposal_id,
                    "proposal_class": proposal.mutation_class,
                    "review_id": review.review_id,
                },
            )
        self._persist()
        return created_draft

    def record_adoption(self, adoption: MutationAdoptionV1) -> list[str]:
        # 2026-09-04 review finding: the duplicate/cooldown/active-surface
        # checks used to run unlocked, then write unlocked -- a real race
        # once the mutation cycle moved to a worker thread (2026-09-03, see
        # this class's own _lock comment above). Two concurrent adoptions for
        # the same target_surface could both observe "no lock held" and both
        # write, silently holding the surface twice. _lock is reentrant
        # (RLock), so the nested acquisition inside rollback_cooldown_until()
        # below is safe.
        with self._lock:
            existing_adoption = next((item for item in self._adoptions.values() if item.proposal_id == adoption.proposal_id), None)
            if existing_adoption is not None:
                if existing_adoption.adoption_id == adoption.adoption_id:
                    return []
                return ["duplicate_adoption_for_proposal"]
            target_surface = adoption.target_surface
            cooldown_until = self.rollback_cooldown_until(target_surface)
            if cooldown_until is not None and _utc_now() < cooldown_until:
                return ["target_surface_in_rollback_cooldown"]
            existing = self._active_surface_by_target.get(target_surface)
            if existing and existing != adoption.adoption_id:
                return ["active_mutation_exists_for_target_surface"]
            self._active_surface_by_target[target_surface] = adoption.adoption_id
            self._adoptions[adoption.adoption_id] = adoption
            proposal = self._proposals.get(adoption.proposal_id)
            if proposal is not None:
                self._proposals[proposal.proposal_id] = proposal.model_copy(update={"rollout_state": "applied"})
        self._set_queue_status_for_proposal(adoption.proposal_id, "applied")
        self._persist()
        return []

    def record_settlement(self, adoption_id: str) -> bool:
        """Release a surface because its change survived, not because it failed.

        ``record_adoption`` takes the one-live-mutation-per-surface lock and,
        until this existed, ``record_rollback`` was the only thing that gave it
        back. So a mutation that *succeeded* held its surface forever: on
        2026-09-02 a single adoption blocked 77 subsequent proposals for
        thirteen hours, every one of them decided ``hold /
        active_surface_mutation_exists``.

        Settling keeps the applied change and the adoption record. It only
        clears the lock, so the surface can be proposed against again.
        """
        adoption = self._adoptions.get(adoption_id)
        if adoption is None or adoption.status != "applied":
            return False
        self._adoptions[adoption_id] = adoption.model_copy(update={"status": "settled"})
        if self._active_surface_by_target.get(adoption.target_surface) == adoption_id:
            self._active_surface_by_target.pop(adoption.target_surface, None)
        self._persist()
        return True

    def record_rollback(self, rollback: MutationRollbackV1) -> bool:
        """Roll back an adoption. Returns False, and changes nothing, if the
        adoption is not currently "applied" -- already rolled back, already
        settled, or unknown.

        Before this guard, the store trusted every caller to only ever call
        this on a live "applied" adoption (true of the one real caller,
        mutation_worker.py's monitoring loop, which already skips non-applied
        adoptions) but enforced nothing itself. A settled adoption has no
        `undone` path by design (docs/superpowers/specs/2026-09-04-orion-
        emergent-choice-seams-brainstorm.md): once a change survives its
        rollback_window_sec, it is kept, not merely eligible to be kept.
        Defense in depth for any future caller (an operator endpoint, a bug)
        that doesn't share the worker loop's discipline.
        """
        adoption = self._adoptions.get(rollback.adoption_id)
        if adoption is None or adoption.status != "applied":
            return False
        self._rollbacks[rollback.rollback_id] = rollback
        self._adoptions[rollback.adoption_id] = adoption.model_copy(update={"status": "rolled_back"})
        self._active_surface_by_target.pop(adoption.target_surface, None)
        proposal = self._proposals.get(rollback.proposal_id)
        if proposal is not None:
            with self._lock:
                self._proposals[rollback.proposal_id] = proposal.model_copy(update={"rollout_state": "rolled_back"})
        self._set_queue_status_for_proposal(rollback.proposal_id, "rolled_back")
        self._persist()
        return True

    def rollback_cooldown_until(self, target_surface: str) -> datetime | None:
        """When `target_surface` next becomes eligible for a new adoption,
        given its most recent rollback -- or None if it has never been rolled
        back (or the relevant records can no longer be resolved; an
        unresolvable lookup fails open rather than becoming a permanent lock).

        Derived entirely from already-persisted rollback/adoption/proposal
        records -- no new schema, no new persistence path. Cheap to remove:
        deleting this method and its one call site in record_adoption()
        reverts to the prior free-retry behavior with no data loss elsewhere.
        """
        with self._lock:
            candidates = [
                r
                for r in self._rollbacks.values()
                if (a := self._adoptions.get(r.adoption_id)) is not None and a.target_surface == target_surface
            ]
            if not candidates:
                return None
            latest = max(candidates, key=lambda r: r.created_at)
            adoption = self._adoptions.get(latest.adoption_id)
            proposal = self._proposals.get(latest.proposal_id)
            if adoption is None or proposal is None:
                return None
            multiplier = _ROLLBACK_COOLDOWN_MULTIPLIER.get(proposal.risk_tier, 1.0)
            return latest.created_at + timedelta(seconds=adoption.rollback_window_sec * multiplier)

    def surface_reliability(self, target_surface: str) -> float | None:
        """Laplace-smoothed settled/(settled+rolled_back) ratio over this
        surface's resolved adoptions -- None below SURFACE_RELIABILITY_MIN_SAMPLES
        (cold start: too few outcomes to be a real read, not a report of
        perfect or zero reliability).

        Being wrong here is not erased by a rollback undoing the action --
        the rollback undoes the PATCH; it does not undo the fact that this
        surface's last proposal was wrong, which is what this method reports.
        """
        with self._lock:
            resolved = [
                a for a in self._adoptions.values() if a.target_surface == target_surface and a.status in ("settled", "rolled_back")
            ]
            if len(resolved) < SURFACE_RELIABILITY_MIN_SAMPLES:
                return None
            settled = sum(1 for a in resolved if a.status == "settled")
            rolled_back = len(resolved) - settled
            return (settled + 1) / (settled + rolled_back + 2)

    def record_apply_blocked(
        self,
        *,
        proposal_id: str,
        decision_id: str,
        target_surface: str,
        reason: str,
        notes: list[str] | None = None,
        queue_status: str | None = None,
    ) -> str:
        block_key = f"{proposal_id}|{decision_id}|{reason}"
        row = {
            "block_key": block_key,
            "proposal_id": proposal_id,
            "decision_id": decision_id,
            "target_surface": target_surface,
            "reason": reason,
            "queue_status": queue_status,
            "notes": list(notes or []),
            "created_at": _utc_now().isoformat(),
        }
        self._blocked_applies[block_key] = row
        self._compact_artifacts()
        self._persist()
        return block_key

    def active_surface(self, target_surface: str) -> str | None:
        return self._active_surface_by_target.get(target_surface)

    def queue_status_for_proposal(self, proposal_id: str) -> str | None:
        for item in self._queue.values():
            if item.proposal_id == proposal_id:
                return item.status
        return None

    def queue_item_id_for_proposal(self, proposal_id: str) -> str | None:
        for item in self._queue.values():
            if item.proposal_id == proposal_id:
                return item.queue_item_id
        return None

    def set_queue_status(self, queue_item_id: str, status: str) -> None:
        item = self._queue.get(queue_item_id)
        if item is None:
            return
        self._queue[queue_item_id] = item.model_copy(update={"status": status})
        self._persist()

    def latest_trials_by_proposal(self) -> dict[str, MutationTrialV1]:
        result: dict[str, MutationTrialV1] = {}
        for trial in self._trials.values():
            prev = result.get(trial.proposal_id)
            if prev is None or trial.created_at > prev.created_at:
                result[trial.proposal_id] = trial
        return result

    def active_surfaces_snapshot(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for target_surface, adoption_id in sorted(self._active_surface_by_target.items()):
            rows.append({"target_surface": target_surface, "adoption_id": adoption_id})
        return rows

    def lifecycle_for_proposal(self, proposal_id: str) -> dict[str, object] | None:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return None
        queue_item = next((item for item in self._queue.values() if item.proposal_id == proposal_id), None)
        trial_rows = sorted(
            (trial for trial in self._trials.values() if trial.proposal_id == proposal_id),
            key=lambda trial: trial.created_at,
        )
        decision_rows = sorted(
            (decision for decision in self._decisions.values() if decision.proposal_id == proposal_id),
            key=lambda decision: decision.created_at,
        )
        adoption = next((item for item in self._adoptions.values() if item.proposal_id == proposal_id), None)
        rollback = next((item for item in self._rollbacks.values() if item.proposal_id == proposal_id), None)
        review_rows = sorted(
            (review for review in self._cognitive_reviews.values() if review.proposal_id == proposal_id),
            key=lambda review: review.created_at,
        )
        draft_rows = sorted(
            (draft for draft in self._cognitive_drafts.values() if draft.proposal_id == proposal_id),
            key=lambda draft: draft.created_at,
        )
        proposal_draft_rows = sorted(
            (draft for draft in self._cognitive_proposal_drafts.values() if draft.proposal_id == proposal_id),
            key=lambda draft: draft.updated_at,
        )
        stance_note_rows = sorted(
            (note for note in self._cognitive_stance_notes.values() if note.source_proposal_id == proposal_id),
            key=lambda note: note.updated_at,
        )
        pressure = next((item for item in self._pressures.values() if item.pressure_id == proposal.source_pressure_id), None)
        signal_rows = [
            signal for signal in self._signals if signal.signal_id in set(proposal.source_signal_ids)
        ]
        signal_rows.sort(key=lambda signal: signal.detected_at)
        payload: dict[str, object] = {
            "proposal": proposal.model_dump(mode="json"),
            "pressure": pressure.model_dump(mode="json") if pressure else None,
            "signals": [signal.model_dump(mode="json") for signal in signal_rows],
            "queue_item": queue_item.model_dump(mode="json") if queue_item else None,
            "trials": [trial.model_dump(mode="json") for trial in trial_rows],
            "decisions": [decision.model_dump(mode="json") for decision in decision_rows],
            "adoption": adoption.model_dump(mode="json") if adoption else None,
            "rollback": rollback.model_dump(mode="json") if rollback else None,
            "cognitive_reviews": [review.model_dump(mode="json") for review in review_rows],
            "cognitive_drafts": [draft.model_dump(mode="json") for draft in draft_rows],
            "cognitive_proposal_drafts": [draft.model_dump(mode="json") for draft in proposal_draft_rows],
            "cognitive_stance_notes": [note.model_dump(mode="json") for note in stance_note_rows],
        }
        if pressure is not None and str(proposal.mutation_class).startswith("recall_") and str(proposal.mutation_class).endswith(
            "_candidate"
        ):
            payload["recall_pressure_evidence_lineage"] = {
                "recall_evidence_history": [dict(item) for item in pressure.recall_evidence_history],
                "recall_evidence_snapshot": dict(pressure.recall_evidence_snapshot),
                "recall_strategy_readiness": readiness_for_pressure(pressure).model_dump(mode="json"),
            }
        return payload

    def recent_lifecycles(self, *, limit: int = 20) -> list[dict[str, object]]:
        proposals = sorted(self._proposals.values(), key=lambda proposal: proposal.created_at, reverse=True)
        payload: list[dict[str, object]] = []
        for proposal in proposals[:limit]:
            lifecycle = self.lifecycle_for_proposal(proposal.proposal_id)
            if lifecycle is not None:
                payload.append(lifecycle)
        return payload

    def recent_blocked_applies(self, *, limit: int = 20) -> list[dict[str, object]]:
        if self._blocked_applies:
            rows = sorted(
                self._blocked_applies.values(),
                key=lambda item: str(item.get("created_at") or ""),
                reverse=True,
            )
            return rows[:limit]

        rows: list[dict[str, object]] = []
        decisions = sorted(self._decisions.values(), key=lambda item: item.created_at, reverse=True)
        for decision in decisions:
            if decision.action != "auto_promote":
                continue
            proposal = self._proposals.get(decision.proposal_id)
            if proposal is None:
                continue
            adoption = next((item for item in self._adoptions.values() if item.proposal_id == proposal.proposal_id), None)
            if adoption is not None:
                continue
            queue_status = self.queue_status_for_proposal(proposal.proposal_id)
            rows.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "decision_id": decision.decision_id,
                    "queue_status": queue_status,
                    "rollout_state": proposal.rollout_state,
                    "target_surface": proposal.target_surface,
                    "reason": decision.reason,
                    "notes": list(decision.notes),
                    "created_at": decision.created_at.isoformat(),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def recent_rollbacks(self, *, limit: int = 20) -> list[dict[str, object]]:
        rows = sorted(self._rollbacks.values(), key=lambda item: item.created_at, reverse=True)[:limit]
        return [row.model_dump(mode="json") for row in rows]

    def recent_signals(self, *, limit: int = 20, target_surface: str | None = None) -> list[dict[str, object]]:
        rows = sorted(self._signals, key=lambda item: item.detected_at, reverse=True)
        if target_surface:
            rows = [row for row in rows if row.target_surface == target_surface]
        return [row.model_dump(mode="json") for row in rows[:limit]]

    def recent_recall_pressures(self, *, limit: int = 20) -> list[dict[str, object]]:
        recall_surfaces = frozenset(
            {
                "recall",
                "recall_strategy_profile",
                "recall_anchor_policy",
                "recall_page_index_profile",
                "recall_graph_expansion_policy",
            }
        )
        rows = [item for item in self._pressures.values() if item.target_surface in recall_surfaces]
        rows.sort(key=lambda item: item.updated_at, reverse=True)
        return [row.model_dump(mode="json") for row in rows[:limit]]

    def recent_cognitive_reviews(self, *, limit: int = 20) -> list[dict[str, object]]:
        rows = sorted(self._cognitive_reviews.values(), key=lambda item: item.created_at, reverse=True)[:limit]
        return [row.model_dump(mode="json") for row in rows]

    def recent_cognitive_drafts(self, *, limit: int = 20) -> list[dict[str, object]]:
        rows = sorted(self._cognitive_drafts.values(), key=lambda item: item.created_at, reverse=True)[:limit]
        return [row.model_dump(mode="json") for row in rows]

    def list_cognitive_proposal_drafts(
        self,
        *,
        limit: int = 20,
        state: str | None = None,
        proposal_class: str | None = None,
    ) -> list[dict[str, object]]:
        rows = list(self._cognitive_proposal_drafts.values())
        if state:
            rows = [row for row in rows if row.state == state]
        if proposal_class:
            rows = [row for row in rows if str(row.proposal_class) == proposal_class]
        rows.sort(key=lambda item: item.updated_at, reverse=True)
        return [row.model_dump(mode="json") for row in rows[:limit]]

    def get_cognitive_proposal_draft(self, draft_id: str) -> CognitiveProposalDraftV1 | None:
        return self._cognitive_proposal_drafts.get(draft_id)

    def archive_cognitive_proposal_draft(self, draft_id: str) -> CognitiveProposalDraftV1 | None:
        row = self._cognitive_proposal_drafts.get(draft_id)
        if row is None:
            return None
        updated = row.model_copy(update={"state": "archived", "updated_at": _utc_now()})
        self._cognitive_proposal_drafts[draft_id] = updated
        self._persist()
        return updated

    def record_cognitive_stance_note(self, row: CognitiveStanceNoteV1) -> CognitiveStanceNoteV1:
        with self._lock:
            self._cognitive_stance_notes[row.stance_note_id] = row
        self._persist()
        return row

    def get_cognitive_stance_note(self, stance_note_id: str) -> CognitiveStanceNoteV1 | None:
        return self._cognitive_stance_notes.get(stance_note_id)

    def list_cognitive_stance_notes(
        self,
        *,
        limit: int = 20,
        status: str | None = None,
    ) -> list[dict[str, object]]:
        rows = list(self._cognitive_stance_notes.values())
        if status:
            rows = [row for row in rows if row.status == status]
        rows.sort(key=lambda item: item.updated_at, reverse=True)
        return [row.model_dump(mode="json") for row in rows[:limit]]

    def archive_cognitive_stance_note(self, stance_note_id: str) -> CognitiveStanceNoteV1 | None:
        row = self._cognitive_stance_notes.get(stance_note_id)
        if row is None:
            return None
        updated = row.model_copy(update={"status": "archived", "updated_at": _utc_now()})
        with self._lock:
            self._cognitive_stance_notes[stance_note_id] = updated
        self._persist()
        return updated

    def get_recall_strategy_profile(self, profile_id: str) -> RecallStrategyProfileV1 | None:
        return self._recall_strategy_profiles.get(profile_id)

    def list_recall_strategy_profiles(self, *, limit: int = 20) -> list[dict[str, object]]:
        rows = sorted(self._recall_strategy_profiles.values(), key=lambda item: item.updated_at, reverse=True)[:limit]
        return [row.model_dump(mode="json") for row in rows]

    def active_recall_shadow_profile(self) -> RecallStrategyProfileV1 | None:
        rows = [row for row in self._recall_strategy_profiles.values() if row.status == "shadow_active"]
        if not rows:
            return None
        return sorted(rows, key=lambda item: item.updated_at, reverse=True)[0]

    def stage_recall_profile(
        self,
        *,
        profile: RecallStrategyProfileV1,
    ) -> RecallStrategyProfileV1:
        staged = profile.model_copy(update={"status": "staged", "updated_at": _utc_now()})
        with self._lock:
            self._recall_strategy_profiles[staged.profile_id] = staged
        self._persist()
        return staged

    def activate_recall_shadow_profile(self, profile_id: str) -> RecallStrategyProfileV1 | None:
        profile = self._recall_strategy_profiles.get(profile_id)
        if profile is None:
            return None
        now = _utc_now()
        for pid, row in list(self._recall_strategy_profiles.items()):
            if pid == profile_id:
                continue
            if row.status == "shadow_active":
                with self._lock:
                    self._recall_strategy_profiles[pid] = row.model_copy(update={"status": "staged", "updated_at": now})
        activated = profile.model_copy(update={"status": "shadow_active", "updated_at": now})
        with self._lock:
            self._recall_strategy_profiles[profile_id] = activated
        self._persist()
        return activated

    def update_recall_strategy_profile(
        self,
        *,
        profile_id: str,
        readiness_snapshot: dict[str, Any] | None = None,
        eval_evidence_refs: list[str] | None = None,
        status: str | None = None,
    ) -> RecallStrategyProfileV1 | None:
        row = self._recall_strategy_profiles.get(profile_id)
        if row is None:
            return None
        patch: dict[str, Any] = {"updated_at": _utc_now()}
        if readiness_snapshot is not None:
            patch["readiness_snapshot"] = dict(readiness_snapshot)
        if eval_evidence_refs is not None:
            patch["eval_evidence_refs"] = list(eval_evidence_refs)[:128]
        if status is not None:
            patch["status"] = status
        updated = row.model_copy(update=patch)
        with self._lock:
            self._recall_strategy_profiles[profile_id] = updated
        self._persist()
        return updated

    def recall_strategy_profile_lineage(self, profile_id: str) -> dict[str, object] | None:
        profile = self._recall_strategy_profiles.get(profile_id)
        if profile is None:
            return None
        proposal = self._proposals.get(profile.source_proposal_id)
        pressure_rows = [row for row in self._pressures.values() if row.pressure_id in set(profile.source_pressure_ids)]
        eval_runs = sorted(
            [row for row in self._recall_shadow_eval_runs.values() if row.profile_id == profile_id],
            key=lambda item: item.completed_at,
            reverse=True,
        )
        reviews = sorted(
            [row for row in self._recall_production_candidate_reviews.values() if row.profile_id == profile_id],
            key=lambda item: item.updated_at,
            reverse=True,
        )
        canary_runs = sorted(
            [row for row in self._recall_canary_runs.values() if row.profile_id == profile_id],
            key=lambda item: item.updated_at,
            reverse=True,
        )
        return {
            "profile": profile.model_dump(mode="json"),
            "proposal": proposal.model_dump(mode="json") if proposal else None,
            "pressures": [row.model_dump(mode="json") for row in sorted(pressure_rows, key=lambda item: item.updated_at, reverse=True)],
            "recent_eval_runs": [row.model_dump(mode="json") for row in eval_runs[:20]],
            "recent_production_candidate_reviews": [row.model_dump(mode="json") for row in reviews[:20]],
            "recent_canary_runs": [row.model_dump(mode="json") for row in canary_runs[:20]],
            "proposal_lineage": self.lifecycle_for_proposal(profile.source_proposal_id),
        }

    def record_recall_shadow_eval_run(self, run: RecallShadowEvalRunV1) -> RecallShadowEvalRunV1:
        with self._lock:
            self._recall_shadow_eval_runs[run.run_id] = run
        self._persist()
        return run

    def get_recall_shadow_eval_run(self, run_id: str) -> RecallShadowEvalRunV1 | None:
        return self._recall_shadow_eval_runs.get(run_id)

    def list_recall_shadow_eval_runs(self, *, limit: int = 20, profile_id: str | None = None) -> list[dict[str, object]]:
        rows = list(self._recall_shadow_eval_runs.values())
        if profile_id:
            rows = [row for row in rows if row.profile_id == profile_id]
        rows.sort(key=lambda item: item.completed_at, reverse=True)
        return [row.model_dump(mode="json") for row in rows[:limit]]

    def record_recall_production_candidate_review(
        self,
        review: RecallProductionCandidateReviewV1,
    ) -> RecallProductionCandidateReviewV1:
        with self._lock:
            self._recall_production_candidate_reviews[review.review_id] = review
        self._persist()
        return review

    def get_recall_production_candidate_review(self, review_id: str) -> RecallProductionCandidateReviewV1 | None:
        return self._recall_production_candidate_reviews.get(review_id)

    def list_recall_production_candidate_reviews(
        self,
        *,
        limit: int = 20,
        profile_id: str | None = None,
    ) -> list[dict[str, object]]:
        rows = list(self._recall_production_candidate_reviews.values())
        if profile_id:
            rows = [row for row in rows if row.profile_id == profile_id]
        rows.sort(key=lambda item: item.updated_at, reverse=True)
        return [row.model_dump(mode="json") for row in rows[:limit]]

    def record_recall_canary_run(self, run: RecallCanaryRunV1) -> RecallCanaryRunV1:
        self._recall_canary_runs[run.canary_run_id] = run
        self._persist()
        return run

    def get_recall_canary_run(self, canary_run_id: str) -> RecallCanaryRunV1 | None:
        return self._recall_canary_runs.get(canary_run_id)

    def list_recall_canary_runs(self, *, limit: int = 20) -> list[dict[str, object]]:
        rows = sorted(self._recall_canary_runs.values(), key=lambda item: item.created_at, reverse=True)
        return [row.model_dump(mode="json") for row in rows[:limit]]

    def record_recall_canary_judgment(self, row: RecallCanaryJudgmentRecordV1) -> RecallCanaryJudgmentRecordV1:
        self._recall_canary_judgments[row.judgment_id] = row
        self._persist()
        return row

    def list_recall_canary_judgments(self, *, limit: int = 20, canary_run_id: str | None = None) -> list[dict[str, object]]:
        rows = list(self._recall_canary_judgments.values())
        if canary_run_id:
            rows = [item for item in rows if item.canary_run_id == canary_run_id]
        rows.sort(key=lambda item: item.created_at, reverse=True)
        return [row.model_dump(mode="json") for row in rows[:limit]]

    def latest_recall_canary_judgment_for_run(self, canary_run_id: str) -> RecallCanaryJudgmentRecordV1 | None:
        rows = [item for item in self._recall_canary_judgments.values() if item.canary_run_id == canary_run_id]
        if not rows:
            return None
        rows.sort(key=lambda item: item.created_at, reverse=True)
        return rows[0]

    def record_recall_canary_review_artifact(self, row: RecallCanaryReviewArtifactV1) -> RecallCanaryReviewArtifactV1:
        self._recall_canary_review_artifacts[row.review_artifact_id] = row
        self._persist()
        return row

    def list_recall_canary_review_artifacts(
        self,
        *,
        limit: int = 20,
        canary_run_id: str | None = None,
    ) -> list[dict[str, object]]:
        rows = list(self._recall_canary_review_artifacts.values())
        if canary_run_id:
            rows = [item for item in rows if item.canary_run_id == canary_run_id]
        rows.sort(key=lambda item: item.created_at, reverse=True)
        return [row.model_dump(mode="json") for row in rows[:limit]]

    def _persist(self) -> None:
        self._compact_artifacts()
        if self.postgres_url:
            try:
                self._persist_to_postgres()
                self._source_kind = "postgres"
                self._last_error = None
                return
            except Exception as exc:
                self._source_kind = "fallback"
                self._last_error = str(exc)
        if self.sql_db_path:
            self._persist_to_sql()

    def _ensure_sql_schema(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_signal (signal_id TEXT PRIMARY KEY, detected_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_pressure (pressure_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_proposal (proposal_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_queue (queue_item_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_trial (trial_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_decision (decision_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_adoption (adoption_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_rollback (rollback_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_cognitive_review (review_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_cognitive_draft (draft_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_cognitive_proposal_draft (draft_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_cognitive_stance_note (stance_note_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS substrate_mutation_recall_strategy_profile (profile_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS substrate_mutation_recall_shadow_eval_run (run_id TEXT PRIMARY KEY, completed_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS substrate_mutation_recall_production_candidate_review (review_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS substrate_mutation_recall_canary_run (canary_run_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS substrate_mutation_recall_canary_judgment (judgment_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS substrate_mutation_recall_canary_review_artifact (review_artifact_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
            )
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_active_surface (target_surface TEXT PRIMARY KEY, adoption_id TEXT NOT NULL, updated_at TEXT NOT NULL)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS substrate_mutation_apply_block (block_key TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
            )
            conn.commit()

    def _persist_to_sql(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            for item in self._signals:
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_signal(signal_id, detected_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(signal_id) DO UPDATE SET
                        detected_at=excluded.detected_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.signal_id, item.detected_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._pressures.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_pressure(pressure_id, updated_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(pressure_id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.pressure_id, item.updated_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._proposals.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_proposal(proposal_id, created_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(proposal_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.proposal_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._queue.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_queue(queue_item_id, created_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(queue_item_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.queue_item_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._trials.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_trial(trial_id, created_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(trial_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.trial_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._decisions.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_decision(decision_id, created_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(decision_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.decision_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._adoptions.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_adoption(adoption_id, created_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(adoption_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.adoption_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._rollbacks.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_rollback(rollback_id, created_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(rollback_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.rollback_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._cognitive_reviews.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_cognitive_review(review_id, created_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(review_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.review_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._cognitive_drafts.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_cognitive_draft(draft_id, created_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(draft_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.draft_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._cognitive_proposal_drafts.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_cognitive_proposal_draft(draft_id, updated_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(draft_id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.draft_id, item.updated_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._cognitive_stance_notes.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_cognitive_stance_note(stance_note_id, updated_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(stance_note_id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.stance_note_id, item.updated_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._recall_strategy_profiles.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_recall_strategy_profile(profile_id, updated_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(profile_id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.profile_id, item.updated_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._recall_shadow_eval_runs.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_recall_shadow_eval_run(run_id, completed_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        completed_at=excluded.completed_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.run_id, item.completed_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._recall_production_candidate_reviews.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_recall_production_candidate_review(review_id, updated_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(review_id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.review_id, item.updated_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._recall_canary_runs.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_recall_canary_run(canary_run_id, updated_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(canary_run_id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.canary_run_id, item.updated_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._recall_canary_judgments.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_recall_canary_judgment(judgment_id, created_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(judgment_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.judgment_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for item in self._recall_canary_review_artifacts.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_recall_canary_review_artifact(review_artifact_id, created_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(review_artifact_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        payload_json=excluded.payload_json
                    """,
                    (item.review_artifact_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
                )
            for surface, adoption_id in self._active_surface_by_target.items():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_active_surface(target_surface, adoption_id, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(target_surface) DO UPDATE SET
                        adoption_id=excluded.adoption_id,
                        updated_at=excluded.updated_at
                    """,
                    (surface, adoption_id, _utc_now().isoformat()),
                )
            if self._active_surface_by_target:
                placeholders = ",".join("?" for _ in self._active_surface_by_target)
                conn.execute(
                    f"DELETE FROM substrate_mutation_active_surface WHERE target_surface NOT IN ({placeholders})",
                    tuple(self._active_surface_by_target.keys()),
                )
            else:
                conn.execute("DELETE FROM substrate_mutation_active_surface")
            for item in self._blocked_applies.values():
                conn.execute(
                    """
                    INSERT INTO substrate_mutation_apply_block(block_key, created_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(block_key) DO UPDATE SET
                        created_at=excluded.created_at,
                        payload_json=excluded.payload_json
                    """,
                    (str(item.get("block_key")), str(item.get("created_at")), json.dumps(item, ensure_ascii=False, sort_keys=True)),
                )
            conn.commit()

    def _load_from_sql(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            self._signals = [MutationSignalV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_signal ORDER BY detected_at ASC").fetchall()]
            loaded_pressures = [MutationPressureV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_pressure ORDER BY updated_at ASC").fetchall()]
            self._pressures = {self._pressure_key(item): item for item in loaded_pressures}
            self._proposals = {item.proposal_id: item for item in [MutationProposalV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_proposal ORDER BY created_at ASC").fetchall()]}
            self._queue = {item.queue_item_id: item for item in [MutationQueueItemV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_queue ORDER BY created_at ASC").fetchall()]}
            self._trials = {item.trial_id: item for item in [MutationTrialV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_trial ORDER BY created_at ASC").fetchall()]}
            self._decisions = {item.decision_id: item for item in [MutationDecisionV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_decision ORDER BY created_at ASC").fetchall()]}
            self._adoptions = {item.adoption_id: item for item in [MutationAdoptionV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_adoption ORDER BY created_at ASC").fetchall()]}
            self._rollbacks = {item.rollback_id: item for item in [MutationRollbackV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_rollback ORDER BY created_at ASC").fetchall()]}
            self._cognitive_reviews = {item.review_id: item for item in [CognitiveProposalReviewV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_cognitive_review ORDER BY created_at ASC").fetchall()]}
            self._cognitive_drafts = {item.draft_id: item for item in [CognitiveDraftRecommendationV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_cognitive_draft ORDER BY created_at ASC").fetchall()]}
            self._cognitive_proposal_drafts = {item.draft_id: item for item in [CognitiveProposalDraftV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_cognitive_proposal_draft ORDER BY updated_at ASC").fetchall()]}
            self._cognitive_stance_notes = {item.stance_note_id: item for item in [CognitiveStanceNoteV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_cognitive_stance_note ORDER BY updated_at ASC").fetchall()]}
            self._recall_strategy_profiles = {
                item.profile_id: item
                for item in [RecallStrategyProfileV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_recall_strategy_profile ORDER BY updated_at ASC").fetchall()]
            }
            self._recall_shadow_eval_runs = {
                item.run_id: item
                for item in [RecallShadowEvalRunV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_recall_shadow_eval_run ORDER BY completed_at ASC").fetchall()]
            }
            self._recall_production_candidate_reviews = {
                item.review_id: item
                for item in [RecallProductionCandidateReviewV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_recall_production_candidate_review ORDER BY updated_at ASC").fetchall()]
            }
            self._recall_canary_runs = {
                item.canary_run_id: item
                for item in [RecallCanaryRunV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_recall_canary_run ORDER BY updated_at ASC").fetchall()]
            }
            self._recall_canary_judgments = {
                item.judgment_id: item
                for item in [RecallCanaryJudgmentRecordV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_recall_canary_judgment ORDER BY created_at ASC").fetchall()]
            }
            self._recall_canary_review_artifacts = {
                item.review_artifact_id: item
                for item in [RecallCanaryReviewArtifactV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_recall_canary_review_artifact ORDER BY created_at ASC").fetchall()]
            }
            self._active_surface_by_target = {surface: adoption_id for (surface, adoption_id) in conn.execute("SELECT target_surface, adoption_id FROM substrate_mutation_active_surface").fetchall()}
            self._blocked_applies = {
                str(item.get("block_key")): item
                for item in [json.loads(p) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_apply_block ORDER BY created_at ASC").fetchall()]
                if isinstance(item, dict) and item.get("block_key")
            }
        self._recover_active_surfaces()
        self._compact_artifacts()

    def _ensure_postgres_schema(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        ddl = [
            "CREATE TABLE IF NOT EXISTS substrate_mutation_signal (signal_id TEXT PRIMARY KEY, detected_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_pressure (pressure_id TEXT PRIMARY KEY, updated_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_proposal (proposal_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_queue (queue_item_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_trial (trial_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_decision (decision_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_adoption (adoption_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_rollback (rollback_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_cognitive_review (review_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_cognitive_draft (draft_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_cognitive_proposal_draft (draft_id TEXT PRIMARY KEY, updated_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_cognitive_stance_note (stance_note_id TEXT PRIMARY KEY, updated_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_recall_strategy_profile (profile_id TEXT PRIMARY KEY, updated_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_recall_shadow_eval_run (run_id TEXT PRIMARY KEY, completed_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_recall_production_candidate_review (review_id TEXT PRIMARY KEY, updated_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_recall_canary_run (canary_run_id TEXT PRIMARY KEY, updated_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_recall_canary_judgment (judgment_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_recall_canary_review_artifact (review_artifact_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_active_surface (target_surface TEXT PRIMARY KEY, adoption_id TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL)",
            "CREATE TABLE IF NOT EXISTS substrate_mutation_apply_block (block_key TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload_json JSONB NOT NULL)",
        ]
        with engine.begin() as conn:
            for statement in ddl:
                conn.execute(text(statement))

    def _persist_to_postgres(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            for item in self._signals:
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_signal(signal_id, detected_at, payload_json)
                        VALUES (:id, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (signal_id) DO UPDATE SET
                            detected_at = EXCLUDED.detected_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.signal_id, "created_at": item.detected_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._pressures.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_pressure(pressure_id, updated_at, payload_json)
                        VALUES (:id, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (pressure_id) DO UPDATE SET
                            updated_at = EXCLUDED.updated_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.pressure_id, "created_at": item.updated_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._proposals.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_proposal(proposal_id, created_at, payload_json)
                        VALUES (:id, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (proposal_id) DO UPDATE SET
                            created_at = EXCLUDED.created_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.proposal_id, "created_at": item.created_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._queue.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_queue(queue_item_id, created_at, payload_json)
                        VALUES (:id, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (queue_item_id) DO UPDATE SET
                            created_at = EXCLUDED.created_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.queue_item_id, "created_at": item.created_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._trials.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_trial(trial_id, created_at, payload_json)
                        VALUES (:id, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (trial_id) DO UPDATE SET
                            created_at = EXCLUDED.created_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.trial_id, "created_at": item.created_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._decisions.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_decision(decision_id, created_at, payload_json)
                        VALUES (:id, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (decision_id) DO UPDATE SET
                            created_at = EXCLUDED.created_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.decision_id, "created_at": item.created_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._adoptions.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_adoption(adoption_id, created_at, payload_json)
                        VALUES (:id, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (adoption_id) DO UPDATE SET
                            created_at = EXCLUDED.created_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.adoption_id, "created_at": item.created_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._rollbacks.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_rollback(rollback_id, created_at, payload_json)
                        VALUES (:id, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (rollback_id) DO UPDATE SET
                            created_at = EXCLUDED.created_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.rollback_id, "created_at": item.created_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._cognitive_reviews.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_cognitive_review(review_id, created_at, payload_json)
                        VALUES (:id, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (review_id) DO UPDATE SET
                            created_at = EXCLUDED.created_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.review_id, "created_at": item.created_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._cognitive_drafts.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_cognitive_draft(draft_id, created_at, payload_json)
                        VALUES (:id, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (draft_id) DO UPDATE SET
                            created_at = EXCLUDED.created_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.draft_id, "created_at": item.created_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._cognitive_proposal_drafts.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_cognitive_proposal_draft(draft_id, updated_at, payload_json)
                        VALUES (:id, :updated_at, CAST(:payload AS JSONB))
                        ON CONFLICT (draft_id) DO UPDATE SET
                            updated_at = EXCLUDED.updated_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.draft_id, "updated_at": item.updated_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._cognitive_stance_notes.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_cognitive_stance_note(stance_note_id, updated_at, payload_json)
                        VALUES (:id, :updated_at, CAST(:payload AS JSONB))
                        ON CONFLICT (stance_note_id) DO UPDATE SET
                            updated_at = EXCLUDED.updated_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.stance_note_id, "updated_at": item.updated_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._recall_strategy_profiles.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_recall_strategy_profile(profile_id, updated_at, payload_json)
                        VALUES (:id, :updated_at, CAST(:payload AS JSONB))
                        ON CONFLICT (profile_id) DO UPDATE SET
                            updated_at = EXCLUDED.updated_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.profile_id, "updated_at": item.updated_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._recall_shadow_eval_runs.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_recall_shadow_eval_run(run_id, completed_at, payload_json)
                        VALUES (:id, :completed_at, CAST(:payload AS JSONB))
                        ON CONFLICT (run_id) DO UPDATE SET
                            completed_at = EXCLUDED.completed_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.run_id, "completed_at": item.completed_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._recall_production_candidate_reviews.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_recall_production_candidate_review(review_id, updated_at, payload_json)
                        VALUES (:id, :updated_at, CAST(:payload AS JSONB))
                        ON CONFLICT (review_id) DO UPDATE SET
                            updated_at = EXCLUDED.updated_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.review_id, "updated_at": item.updated_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._recall_canary_runs.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_recall_canary_run(canary_run_id, updated_at, payload_json)
                        VALUES (:id, :updated_at, CAST(:payload AS JSONB))
                        ON CONFLICT (canary_run_id) DO UPDATE SET
                            updated_at = EXCLUDED.updated_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.canary_run_id, "updated_at": item.updated_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._recall_canary_judgments.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_recall_canary_judgment(judgment_id, created_at, payload_json)
                        VALUES (:id, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (judgment_id) DO UPDATE SET
                            created_at = EXCLUDED.created_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.judgment_id, "created_at": item.created_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            for item in self._recall_canary_review_artifacts.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_recall_canary_review_artifact(review_artifact_id, created_at, payload_json)
                        VALUES (:id, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (review_artifact_id) DO UPDATE SET
                            created_at = EXCLUDED.created_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {"id": item.review_artifact_id, "created_at": item.created_at, "payload": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)},
                )
            conn.execute(text("DELETE FROM substrate_mutation_active_surface"))
            for surface, adoption_id in self._active_surface_by_target.items():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_active_surface(target_surface, adoption_id, updated_at)
                        VALUES (:surface, :adoption_id, :updated_at)
                        ON CONFLICT (target_surface) DO UPDATE SET
                            adoption_id = EXCLUDED.adoption_id,
                            updated_at = EXCLUDED.updated_at
                        """
                    ),
                    {"surface": surface, "adoption_id": adoption_id, "updated_at": _utc_now()},
                )
            for item in self._blocked_applies.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_mutation_apply_block(block_key, created_at, payload_json)
                        VALUES (:block_key, :created_at, CAST(:payload AS JSONB))
                        ON CONFLICT (block_key) DO UPDATE SET
                            created_at = EXCLUDED.created_at,
                            payload_json = EXCLUDED.payload_json
                        """
                    ),
                    {
                        "block_key": str(item.get("block_key")),
                        "created_at": str(item.get("created_at")),
                        "payload": json.dumps(item, ensure_ascii=False, sort_keys=True),
                    },
                )

    def _load_from_postgres(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            self._signals = [MutationSignalV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_signal ORDER BY detected_at ASC")).fetchall()]
            loaded_pressures = [MutationPressureV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_pressure ORDER BY updated_at ASC")).fetchall()]
            self._pressures = {self._pressure_key(item): item for item in loaded_pressures}
            self._proposals = {item.proposal_id: item for item in [MutationProposalV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_proposal ORDER BY created_at ASC")).fetchall()]}
            self._queue = {item.queue_item_id: item for item in [MutationQueueItemV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_queue ORDER BY created_at ASC")).fetchall()]}
            self._trials = {item.trial_id: item for item in [MutationTrialV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_trial ORDER BY created_at ASC")).fetchall()]}
            self._decisions = {item.decision_id: item for item in [MutationDecisionV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_decision ORDER BY created_at ASC")).fetchall()]}
            self._adoptions = {item.adoption_id: item for item in [MutationAdoptionV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_adoption ORDER BY created_at ASC")).fetchall()]}
            self._rollbacks = {item.rollback_id: item for item in [MutationRollbackV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_rollback ORDER BY created_at ASC")).fetchall()]}
            self._cognitive_reviews = {item.review_id: item for item in [CognitiveProposalReviewV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_cognitive_review ORDER BY created_at ASC")).fetchall()]}
            self._cognitive_drafts = {item.draft_id: item for item in [CognitiveDraftRecommendationV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_cognitive_draft ORDER BY created_at ASC")).fetchall()]}
            self._cognitive_proposal_drafts = {item.draft_id: item for item in [CognitiveProposalDraftV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_cognitive_proposal_draft ORDER BY updated_at ASC")).fetchall()]}
            self._cognitive_stance_notes = {item.stance_note_id: item for item in [CognitiveStanceNoteV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_cognitive_stance_note ORDER BY updated_at ASC")).fetchall()]}
            self._recall_strategy_profiles = {
                item.profile_id: item
                for item in [RecallStrategyProfileV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_recall_strategy_profile ORDER BY updated_at ASC")).fetchall()]
            }
            self._recall_shadow_eval_runs = {
                item.run_id: item
                for item in [RecallShadowEvalRunV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_recall_shadow_eval_run ORDER BY completed_at ASC")).fetchall()]
            }
            self._recall_production_candidate_reviews = {
                item.review_id: item
                for item in [RecallProductionCandidateReviewV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_recall_production_candidate_review ORDER BY updated_at ASC")).fetchall()]
            }
            self._recall_canary_runs = {
                item.canary_run_id: item
                for item in [RecallCanaryRunV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_recall_canary_run ORDER BY updated_at ASC")).fetchall()]
            }
            self._recall_canary_judgments = {
                item.judgment_id: item
                for item in [RecallCanaryJudgmentRecordV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_recall_canary_judgment ORDER BY created_at ASC")).fetchall()]
            }
            self._recall_canary_review_artifacts = {
                item.review_artifact_id: item
                for item in [RecallCanaryReviewArtifactV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_recall_canary_review_artifact ORDER BY created_at ASC")).fetchall()]
            }
            self._active_surface_by_target = {surface: adoption_id for (surface, adoption_id) in conn.execute(text("SELECT target_surface, adoption_id FROM substrate_mutation_active_surface")).fetchall()}
            self._blocked_applies = {
                str(item.get("block_key")): item
                for item in [json.loads(p) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_apply_block ORDER BY created_at ASC")).fetchall()]
                if isinstance(item, dict) and item.get("block_key")
            }
        self._recover_active_surfaces()
        self._compact_artifacts()

    @staticmethod
    def _pressure_key(pressure: MutationPressureV1) -> str:
        return f"{pressure.anchor_scope}|{pressure.subject_ref}|{pressure.target_surface}"

    @staticmethod
    def pressure_key_for(*, anchor_scope: str, subject_ref: str, target_surface: str) -> str:
        return f"{anchor_scope}|{subject_ref}|{target_surface}"

    def _set_queue_status_for_proposal(self, proposal_id: str, status: str) -> None:
        for queue_item_id, item in self._queue.items():
            if item.proposal_id == proposal_id:
                self._queue[queue_item_id] = item.model_copy(update={"status": status})
                break

    def _recover_active_surfaces(self) -> None:
        recovered: dict[str, str] = {}
        for adoption in self._adoptions.values():
            if adoption.status == "applied":
                recovered[adoption.target_surface] = adoption.adoption_id
        self._active_surface_by_target = recovered

    def _compact_artifacts(self) -> None:
        if len(self._blocked_applies) > self._retention_max_blocked_applies:
            rows = sorted(self._blocked_applies.values(), key=lambda row: str(row.get("created_at") or ""))
            keep = rows[-self._retention_max_blocked_applies :]
            self._blocked_applies = {str(row["block_key"]): row for row in keep if row.get("block_key")}
        if len(self._rollbacks) > self._retention_max_rollbacks:
            rows = sorted(self._rollbacks.values(), key=lambda row: row.created_at)
            keep = rows[-self._retention_max_rollbacks :]
            self._rollbacks = {row.rollback_id: row for row in keep}
        if len(self._adoptions) > self._retention_max_adoptions:
            # Never evict an adoption that still holds a surface: dropping the
            # holder would strand the lock with nothing able to release it.
            # 2026-09-04 review finding: also never evict an adoption still
            # referenced by a RETAINED rollback record (rollbacks have their
            # own, separate retention above). rollback_cooldown_until() joins
            # a rollback back to its adoption to resolve target_surface and
            # rollback_window_sec -- dropping the adoption first would make
            # that lookup fail and the cooldown fail open (by design, an
            # unresolvable lookup does not become a permanent lock) on a
            # surface that should still be cooling down.
            held = set(self._active_surface_by_target.values()) | {r.adoption_id for r in self._rollbacks.values()}
            rows = sorted(self._adoptions.values(), key=lambda row: row.created_at)
            evictable = [row for row in rows if row.adoption_id not in held]
            surplus = len(self._adoptions) - self._retention_max_adoptions
            drop = {row.adoption_id for row in evictable[:surplus]}
            if drop:
                self._adoptions = {k: v for k, v in self._adoptions.items() if k not in drop}

    def _persist_signal(self, signal: MutationSignalV1) -> bool:
        if self.postgres_url:
            try:
                self._persist_signal_postgres(signal)
                return True
            except Exception:
                pass
        if self.sql_db_path:
            try:
                self._persist_signal_sqlite(signal)
                return True
            except Exception:
                pass
        return False

    def _persist_signal_sqlite(self, signal: MutationSignalV1) -> None:
        if not self.sql_db_path:
            raise RuntimeError("sqlite_disabled")
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute(
                """
                INSERT INTO substrate_mutation_signal(signal_id, detected_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(signal_id) DO UPDATE SET
                    detected_at=excluded.detected_at,
                    payload_json=excluded.payload_json
                """,
                (signal.signal_id, signal.detected_at.isoformat(), json.dumps(signal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
            )
            conn.commit()

    def _persist_signal_postgres(self, signal: MutationSignalV1) -> None:
        if not self.postgres_url:
            raise RuntimeError("postgres_disabled")
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_mutation_signal(signal_id, detected_at, payload_json)
                    VALUES (:id, :detected_at, CAST(:payload AS JSONB))
                    ON CONFLICT (signal_id) DO UPDATE SET
                        detected_at=EXCLUDED.detected_at,
                        payload_json=EXCLUDED.payload_json
                    """
                ),
                {
                    "id": signal.signal_id,
                    "detected_at": signal.detected_at,
                    "payload": json.dumps(signal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                },
            )

    def _persist_pressure(self, pressure: MutationPressureV1) -> bool:
        # Mirrors `_persist()`'s own postgres-success clearing (source_kind
        # set, last_error cleared) -- review caught that `_persist_signal`'s
        # existing fast path never did this, so a store that had drifted to
        # "fallback" from a past outage would stay flagged degraded forever
        # once the database recovered, since no incremental write ever un-set
        # it. Fixed here; `_persist_signal` has the identical pre-existing
        # gap, left as a follow-up rather than touched in this patch.
        if self.postgres_url:
            try:
                self._persist_pressure_postgres(pressure)
                self._source_kind = "postgres"
                self._last_error = None
                return True
            except Exception:
                pass
        if self.sql_db_path:
            try:
                self._persist_pressure_sqlite(pressure)
                return True
            except Exception:
                pass
        return False

    def _persist_pressure_sqlite(self, pressure: MutationPressureV1) -> None:
        if not self.sql_db_path:
            raise RuntimeError("sqlite_disabled")
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute(
                """
                INSERT INTO substrate_mutation_pressure(pressure_id, updated_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(pressure_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    payload_json=excluded.payload_json
                """,
                (pressure.pressure_id, pressure.updated_at.isoformat(), json.dumps(pressure.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
            )
            conn.commit()

    def _persist_pressure_postgres(self, pressure: MutationPressureV1) -> None:
        if not self.postgres_url:
            raise RuntimeError("postgres_disabled")
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_mutation_pressure(pressure_id, updated_at, payload_json)
                    VALUES (:id, :updated_at, CAST(:payload AS JSONB))
                    ON CONFLICT (pressure_id) DO UPDATE SET
                        updated_at=EXCLUDED.updated_at,
                        payload_json=EXCLUDED.payload_json
                    """
                ),
                {
                    "id": pressure.pressure_id,
                    "updated_at": pressure.updated_at,
                    "payload": json.dumps(pressure.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                },
            )

    def _persist_proposal(self, proposal: MutationProposalV1) -> bool:
        if self.postgres_url:
            try:
                self._persist_proposal_postgres(proposal)
                self._source_kind = "postgres"
                self._last_error = None
                return True
            except Exception:
                pass
        if self.sql_db_path:
            try:
                self._persist_proposal_sqlite(proposal)
                return True
            except Exception:
                pass
        return False

    def _persist_proposal_sqlite(self, proposal: MutationProposalV1) -> None:
        if not self.sql_db_path:
            raise RuntimeError("sqlite_disabled")
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute(
                """
                INSERT INTO substrate_mutation_proposal(proposal_id, created_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(proposal_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    payload_json=excluded.payload_json
                """,
                (proposal.proposal_id, proposal.created_at.isoformat(), json.dumps(proposal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
            )
            conn.commit()

    def _persist_proposal_postgres(self, proposal: MutationProposalV1) -> None:
        if not self.postgres_url:
            raise RuntimeError("postgres_disabled")
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_mutation_proposal(proposal_id, created_at, payload_json)
                    VALUES (:id, :created_at, CAST(:payload AS JSONB))
                    ON CONFLICT (proposal_id) DO UPDATE SET
                        created_at=EXCLUDED.created_at,
                        payload_json=EXCLUDED.payload_json
                    """
                ),
                {
                    "id": proposal.proposal_id,
                    "created_at": proposal.created_at,
                    "payload": json.dumps(proposal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                },
            )

    def _persist_proposal_and_queue_item(
        self, proposal: MutationProposalV1, queue_item: MutationQueueItemV1
    ) -> bool:
        """Write a brand-new proposal and its queue row in one transaction.

        Same fallback contract as `_persist_pressure`/`_persist_signal`: try
        the cheap incremental path, and if both backends fail, the caller
        falls back to the full `_persist()` sweep so `degraded()`/
        `last_error()` still reflect a real outage.
        """
        if self.postgres_url:
            try:
                self._persist_proposal_and_queue_item_postgres(proposal, queue_item)
                self._source_kind = "postgres"
                self._last_error = None
                return True
            except Exception:
                pass
        if self.sql_db_path:
            try:
                self._persist_proposal_and_queue_item_sqlite(proposal, queue_item)
                return True
            except Exception:
                pass
        return False

    def _persist_proposal_and_queue_item_sqlite(
        self, proposal: MutationProposalV1, queue_item: MutationQueueItemV1
    ) -> None:
        if not self.sql_db_path:
            raise RuntimeError("sqlite_disabled")
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute(
                """
                INSERT INTO substrate_mutation_proposal(proposal_id, created_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(proposal_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    payload_json=excluded.payload_json
                """,
                (proposal.proposal_id, proposal.created_at.isoformat(), json.dumps(proposal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
            )
            conn.execute(
                """
                INSERT INTO substrate_mutation_queue(queue_item_id, created_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(queue_item_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    payload_json=excluded.payload_json
                """,
                (queue_item.queue_item_id, queue_item.created_at.isoformat(), json.dumps(queue_item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)),
            )
            conn.commit()

    def _persist_proposal_and_queue_item_postgres(
        self, proposal: MutationProposalV1, queue_item: MutationQueueItemV1
    ) -> None:
        if not self.postgres_url:
            raise RuntimeError("postgres_disabled")
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_mutation_proposal(proposal_id, created_at, payload_json)
                    VALUES (:id, :created_at, CAST(:payload AS JSONB))
                    ON CONFLICT (proposal_id) DO UPDATE SET
                        created_at=EXCLUDED.created_at,
                        payload_json=EXCLUDED.payload_json
                    """
                ),
                {
                    "id": proposal.proposal_id,
                    "created_at": proposal.created_at,
                    "payload": json.dumps(proposal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                },
            )
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_mutation_queue(queue_item_id, created_at, payload_json)
                    VALUES (:id, :created_at, CAST(:payload AS JSONB))
                    ON CONFLICT (queue_item_id) DO UPDATE SET
                        created_at=EXCLUDED.created_at,
                        payload_json=EXCLUDED.payload_json
                    """
                ),
                {
                    "id": queue_item.queue_item_id,
                    "created_at": queue_item.created_at,
                    "payload": json.dumps(queue_item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                },
            )

    @staticmethod
    def _env_int(name: str, default: int, *, low: int, high: int) -> int:
        raw = str(os.getenv(name, str(default))).strip()
        try:
            value = int(raw)
        except ValueError:
            return default
        return max(low, min(high, value))


def cognition_view_snapshot(store: "SubstrateMutationStore") -> dict[str, list[Any]]:
    """Consistent copies of the dicts the chat-path cognition context reads.

    Two distinct guarantees, and they come from two different things -- worth
    separating, because conflating them produced a vacuous test on the first
    attempt:

      * The `list(...)` COPY is what prevents `RuntimeError: dictionary
        changed size during iteration`. The caller iterates these at Python
        level (`[row for row in ... if ...]`), which yields between items and
        so can be interrupted by a writer on the mutation-cycle thread.
        Copying first removes the live view entirely. (Note that `list(view)`
        is itself C-level and GIL-atomic, so the copy does not need the lock
        to be crash-safe -- verified by removing the lock and watching the
        race test still pass.)
      * The LOCK is what makes all six copies come from one instant. Before
        the mutation cycle moved off the event loop, a loop-side reader could
        only ever observe a cycle boundary; without the lock the reader could
        now see a trial whose proposal is not yet visible. That is a
        consistency change nobody asked for, so it is held rather than
        silently dropped.

    Values are not deep-copied: the records are pydantic models the store
    replaces wholesale (`model_copy(update=...)`) rather than mutating, so a
    shallow copy of the value list is a real point-in-time view.
    """
    with store._lock:
        return {
            "trials": list(store._trials.values()),
            "proposals": list(store._proposals.values()),
            "recall_strategy_profiles": list(store._recall_strategy_profiles.values()),
            "recall_shadow_eval_runs": list(store._recall_shadow_eval_runs.values()),
            "recall_production_candidate_reviews": list(
                store._recall_production_candidate_reviews.values()
            ),
            "cognitive_stance_notes": list(store._cognitive_stance_notes.values()),
        }
