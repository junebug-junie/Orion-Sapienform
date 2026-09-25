from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
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


# 2026-09-25 live incident: `_persist_to_postgres` used to re-UPSERT every
# in-memory row of every table (550k+ signals) inside ONE transaction, one
# statement at a time, on every `_persist()` call. That transaction sat "idle
# in transaction" for 7+ minutes between statements; `CREATE INDEX
# CONCURRENTLY` anywhere in the database (LangGraph saver.setup() at
# orion-gpu-pool boot) waited on it, and the LLM lane was down ~10 minutes
# until the sessions were terminated by hand. Persisting now writes only rows
# whose serialized payload differs from what this store last wrote/loaded, in
# transactions of at most this many rows each.
_PERSIST_BATCH_ROWS = 500

# Row-per-record tables, in the order the pre-2026-09-25 full sweep wrote
# them. (table, id column, timestamp column, store attribute, model class).
# The id/timestamp column names are also the model attribute names.
_ROW_TABLES: tuple[tuple[str, str, str, str, type], ...] = (
    ("substrate_mutation_pressure", "pressure_id", "updated_at", "_pressures", MutationPressureV1),
    ("substrate_mutation_proposal", "proposal_id", "created_at", "_proposals", MutationProposalV1),
    ("substrate_mutation_queue", "queue_item_id", "created_at", "_queue", MutationQueueItemV1),
    ("substrate_mutation_trial", "trial_id", "created_at", "_trials", MutationTrialV1),
    ("substrate_mutation_decision", "decision_id", "created_at", "_decisions", MutationDecisionV1),
    ("substrate_mutation_adoption", "adoption_id", "created_at", "_adoptions", MutationAdoptionV1),
    ("substrate_mutation_rollback", "rollback_id", "created_at", "_rollbacks", MutationRollbackV1),
    ("substrate_mutation_cognitive_review", "review_id", "created_at", "_cognitive_reviews", CognitiveProposalReviewV1),
    ("substrate_mutation_cognitive_draft", "draft_id", "created_at", "_cognitive_drafts", CognitiveDraftRecommendationV1),
    ("substrate_mutation_cognitive_proposal_draft", "draft_id", "updated_at", "_cognitive_proposal_drafts", CognitiveProposalDraftV1),
    ("substrate_mutation_cognitive_stance_note", "stance_note_id", "updated_at", "_cognitive_stance_notes", CognitiveStanceNoteV1),
    ("substrate_mutation_recall_strategy_profile", "profile_id", "updated_at", "_recall_strategy_profiles", RecallStrategyProfileV1),
    ("substrate_mutation_recall_shadow_eval_run", "run_id", "completed_at", "_recall_shadow_eval_runs", RecallShadowEvalRunV1),
    (
        "substrate_mutation_recall_production_candidate_review",
        "review_id",
        "updated_at",
        "_recall_production_candidate_reviews",
        RecallProductionCandidateReviewV1,
    ),
    ("substrate_mutation_recall_canary_run", "canary_run_id", "updated_at", "_recall_canary_runs", RecallCanaryRunV1),
    ("substrate_mutation_recall_canary_judgment", "judgment_id", "created_at", "_recall_canary_judgments", RecallCanaryJudgmentRecordV1),
    (
        "substrate_mutation_recall_canary_review_artifact",
        "review_artifact_id",
        "created_at",
        "_recall_canary_review_artifacts",
        RecallCanaryReviewArtifactV1,
    ),
)
_SIGNAL_TABLE = ("substrate_mutation_signal", "signal_id", "detected_at")
_APPLY_BLOCK_TABLE = ("substrate_mutation_apply_block", "block_key", "created_at")
_ACTIVE_SURFACE_TABLE = "substrate_mutation_active_surface"


def _payload_json(item: Any) -> str:
    """The exact payload string every writer in this module persists."""
    data = item.model_dump(mode="json") if hasattr(item, "model_dump") else item
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _digest(payload: str) -> bytes:
    """What the persisted-state maps keep per row: a 16-byte digest of the
    payload string, not the (KB-sized) string itself."""
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()


def _normalize_payload_text(raw: Any) -> str:
    """A loaded payload re-serialized the same way `_payload_json` writes one,
    so an unchanged row compares equal to what was loaded."""
    data = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _upsert_sql(backend: str, table: str, id_col: str, ts_col: str) -> str:
    if backend == "postgres":
        return (
            f"INSERT INTO {table}({id_col}, {ts_col}, payload_json) "
            f"VALUES (:id, :ts, CAST(:payload AS JSONB)) "
            f"ON CONFLICT ({id_col}) DO UPDATE SET {ts_col} = EXCLUDED.{ts_col}, payload_json = EXCLUDED.payload_json"
        )
    return (
        f"INSERT INTO {table}({id_col}, {ts_col}, payload_json) VALUES (?, ?, ?) "
        f"ON CONFLICT({id_col}) DO UPDATE SET {ts_col}=excluded.{ts_col}, payload_json=excluded.payload_json"
    )


def _sqlite_ts(value: Any) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


@dataclass
class _WriteRow:
    table: str
    row_id: str
    ts: Any
    payload: str
    #: Position in `_signals` for signal rows (advances the signal mark on
    #: commit); None for every other table.
    signal_index: int | None = None


@dataclass
class _WriteOp:
    table: str
    sql: str
    rows: list[_WriteRow]
    #: Active-surface rewrite: DELETE-all then INSERT, never split across
    #: transactions (a split would expose an empty lock table to a reload).
    active_surface: bool = False


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
    #: What each backend is known to hold, per table: row id -> digest of the
    #: exact payload string last written or loaded. `_persist()` only writes rows
    #: whose current payload differs. Keyed by backend because the Postgres
    #: and sqlite mirrors are written independently (sqlite only on fallback).
    _persisted_payloads: dict[str, dict[str, dict[str, bytes]]] = field(default_factory=dict, init=False, repr=False)
    #: `_signals` is append-only; per backend, (the list object, count of its
    #: leading entries already persisted). Keeping the list object detects a
    #: wholesale reassignment (reload), which resets the mark to 0.
    _signal_marks: dict[str, tuple[list[MutationSignalV1], int]] = field(default_factory=dict, init=False, repr=False)
    #: One cached SQLAlchemy engine per store (was: create_engine per call).
    _engine: Any = field(default=None, init=False, repr=False)
    _engine_url: str | None = field(default=None, init=False, repr=False)

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

    def recent_lifecycles(self, *, limit: int = 20, mutation_class: str | None = None) -> list[dict[str, object]]:
        proposals = sorted(self._proposals.values(), key=lambda proposal: proposal.created_at, reverse=True)
        if mutation_class is not None:
            # Filtered BEFORE the limit slice, not after: this store has no
            # eviction and is shared by every mutation class the scheduler
            # runs (12+, at last count). A class filter applied to the
            # already-sliced top `limit` proposals would go silently empty
            # the moment `limit` other-class proposals land more recently
            # than the last one in the class you actually asked about --
            # exactly the "loop is doing real work but the view shows
            # nothing" failure a caller filtering post-hoc is trying to
            # avoid. Confirmed live: routing_threshold_patch alone produced
            # ~190 proposals in 36 hours before its own retirement.
            proposals = [proposal for proposal in proposals if proposal.mutation_class == mutation_class]
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

    # ------------------------------------------------------------------
    # Dirty-only, batched persistence (2026-09-25; see _PERSIST_BATCH_ROWS)
    # ------------------------------------------------------------------

    def _pg_engine(self) -> Any:
        """The store's one cached engine. Rebuilt only if postgres_url changes."""
        from sqlalchemy import create_engine

        with self._lock:
            if self._engine is None or self._engine_url != self.postgres_url:
                old = self._engine
                self._engine = create_engine(self.postgres_url, pool_pre_ping=True, pool_size=2, max_overflow=3)
                self._engine_url = self.postgres_url
                if old is not None:
                    try:
                        old.dispose()
                    except Exception:
                        pass
            return self._engine

    def _known_payloads(self, backend: str, table: str) -> dict[str, bytes]:
        return self._persisted_payloads.setdefault(backend, {}).setdefault(table, {})

    def _note_persisted(self, backend: str, table: str, row_id: str, payload: str) -> None:
        with self._lock:
            self._known_payloads(backend, table)[row_id] = _digest(payload)

    def _note_signal_persisted(self, backend: str, signal: MutationSignalV1) -> None:
        """Advance the signal mark past `signal` if it is the next unpersisted one."""
        with self._lock:
            lst = self._signals
            mark = self._signal_marks.get(backend)
            if mark is None or mark[0] is not lst:
                return
            idx = mark[1]
            if idx < len(lst) and lst[idx] is signal:
                self._signal_marks[backend] = (lst, idx + 1)

    def _dirty_write_ops(self, backend: str) -> list[_WriteOp]:
        """Every row whose current payload differs from what `backend` holds.

        Signals are append-only facts: only entries past the backend's mark
        are written (O(new), not O(550k) -- no per-signal serialization of
        history). Every other table is small (thousands of rows) and mutated
        via model_copy or in place, so it is diffed by serialized payload.
        The active-surface lock table is always rewritten, exactly as before
        (DELETE-all + INSERT, in one transaction), since its updated_at is
        stamped per write and it holds a handful of rows.
        """
        ops: list[_WriteOp] = []
        with self._lock:
            signals = self._signals
            end = len(signals)
            mark = self._signal_marks.get(backend)
            start = mark[1] if mark is not None and mark[0] is signals and mark[1] <= end else 0
            new_signals = signals[start:end]
            tables = [(table, id_col, ts_col, list(getattr(self, attr).values())) for table, id_col, ts_col, attr, _model in _ROW_TABLES]
            active_surfaces = list(self._active_surface_by_target.items())
            blocked = list(self._blocked_applies.values())
        table, id_col, ts_col = _SIGNAL_TABLE
        if new_signals:
            ops.append(
                _WriteOp(
                    table,
                    _upsert_sql(backend, table, id_col, ts_col),
                    [_WriteRow(table, s.signal_id, s.detected_at, _payload_json(s), start + i) for i, s in enumerate(new_signals)],
                )
            )
        for table, id_col, ts_col, items in tables:
            known = self._known_payloads(backend, table)
            rows = []
            for item in items:
                payload = _payload_json(item)
                row_id = getattr(item, id_col)
                if known.get(row_id) != _digest(payload):
                    rows.append(_WriteRow(table, row_id, getattr(item, ts_col), payload))
            if rows:
                ops.append(_WriteOp(table, _upsert_sql(backend, table, id_col, ts_col), rows))
        now = _utc_now()
        ops.append(
            _WriteOp(
                _ACTIVE_SURFACE_TABLE,
                "",
                [_WriteRow(_ACTIVE_SURFACE_TABLE, surface, now, adoption_id) for surface, adoption_id in active_surfaces],
                active_surface=True,
            )
        )
        table, id_col, ts_col = _APPLY_BLOCK_TABLE
        known = self._known_payloads(backend, table)
        rows = []
        for item in blocked:
            payload = _payload_json(item)
            row_id = str(item.get("block_key"))
            if known.get(row_id) != _digest(payload):
                rows.append(_WriteRow(table, row_id, str(item.get("created_at")), payload))
        if rows:
            ops.append(_WriteOp(table, _upsert_sql(backend, table, id_col, ts_col), rows))
        return ops

    @staticmethod
    def _chunk_ops(ops: list[_WriteOp], limit: int = _PERSIST_BATCH_ROWS) -> list[list[_WriteOp]]:
        """Pack ops into transactions of at most `limit` rows, in order."""
        txns: list[list[_WriteOp]] = []
        current: list[_WriteOp] = []
        count = 0
        for op in ops:
            if op.active_surface:
                size = max(1, len(op.rows))
                if current and count + size > limit:
                    txns.append(current)
                    current, count = [], 0
                current.append(op)
                count += size
                continue
            i = 0
            while i < len(op.rows):
                room = limit - count
                if room <= 0:
                    txns.append(current)
                    current, count = [], 0
                    room = limit
                part = op.rows[i : i + room]
                current.append(_WriteOp(op.table, op.sql, part))
                count += len(part)
                i += len(part)
        if current:
            txns.append(current)
        return txns

    def _mark_committed(self, backend: str, txn: list[_WriteOp]) -> None:
        with self._lock:
            for op in txn:
                if op.active_surface:
                    continue
                for row in op.rows:
                    if row.signal_index is not None:
                        mark = self._signal_marks.get(backend)
                        base = mark[1] if mark is not None and mark[0] is self._signals else 0
                        if row.signal_index + 1 > base:
                            self._signal_marks[backend] = (self._signals, row.signal_index + 1)
                    else:
                        self._known_payloads(backend, row.table)[row.row_id] = _digest(row.payload)

    def _seed_persisted_state(self, backend: str, raw: dict[str, list[tuple[str, Any]]]) -> None:
        """After a load, record what `backend` holds so the next persist is a no-op."""
        with self._lock:
            self._persisted_payloads[backend] = {
                table: {str(row_id): _digest(_normalize_payload_text(payload)) for row_id, payload in rows}
                for table, rows in raw.items()
                if table != _SIGNAL_TABLE[0]
            }
            self._signal_marks[backend] = (self._signals, len(self._signals))

    def _apply_loaded_rows(self, raw: dict[str, list[tuple[str, Any]]]) -> None:
        """Rebuild in-memory state from raw (row_id, payload) rows per table.

        Parsing happens here, outside any database transaction.
        """
        def _parse(payload: Any) -> Any:
            return json.loads(payload) if isinstance(payload, (str, bytes, bytearray)) else payload

        self._signals = [MutationSignalV1.model_validate(_parse(p)) for _row_id, p in raw.get(_SIGNAL_TABLE[0], [])]
        for table, id_col, _ts_col, attr, model in _ROW_TABLES:
            items = [model.model_validate(_parse(p)) for _row_id, p in raw.get(table, [])]
            if attr == "_pressures":
                setattr(self, attr, {self._pressure_key(item): item for item in items})
            else:
                setattr(self, attr, {getattr(item, id_col): item for item in items})
        self._active_surface_by_target = {str(surface): str(adoption_id) for surface, adoption_id in raw.get(_ACTIVE_SURFACE_TABLE, [])}
        self._blocked_applies = {
            str(item.get("block_key")): item
            for item in [_parse(p) for _row_id, p in raw.get(_APPLY_BLOCK_TABLE[0], [])]
            if isinstance(item, dict) and item.get("block_key")
        }

    @staticmethod
    def _load_queries(backend: str) -> list[tuple[str, str]]:
        cast = "::text" if backend == "postgres" else ""
        queries = [(_SIGNAL_TABLE[0], f"SELECT {_SIGNAL_TABLE[1]}, payload_json{cast} FROM {_SIGNAL_TABLE[0]} ORDER BY {_SIGNAL_TABLE[2]} ASC")]
        for table, id_col, ts_col, _attr, _model in _ROW_TABLES:
            queries.append((table, f"SELECT {id_col}, payload_json{cast} FROM {table} ORDER BY {ts_col} ASC"))
        queries.append((_ACTIVE_SURFACE_TABLE, f"SELECT target_surface, adoption_id FROM {_ACTIVE_SURFACE_TABLE}"))
        queries.append(
            (_APPLY_BLOCK_TABLE[0], f"SELECT {_APPLY_BLOCK_TABLE[1]}, payload_json{cast} FROM {_APPLY_BLOCK_TABLE[0]} ORDER BY {_APPLY_BLOCK_TABLE[2]} ASC")
        )
        return queries

    def _persist_to_sql(self) -> None:
        if not self.sql_db_path:
            return
        conn = sqlite3.connect(self.sql_db_path)
        try:
            for txn in self._chunk_ops(self._dirty_write_ops("sqlite")):
                with conn:  # one short transaction per batch; commits on exit
                    for op in txn:
                        if op.active_surface:
                            conn.execute(f"DELETE FROM {_ACTIVE_SURFACE_TABLE}")
                            if op.rows:
                                conn.executemany(
                                    f"INSERT INTO {_ACTIVE_SURFACE_TABLE}(target_surface, adoption_id, updated_at) VALUES (?, ?, ?)",
                                    [(row.row_id, row.payload, _sqlite_ts(row.ts)) for row in op.rows],
                                )
                        else:
                            conn.executemany(op.sql, [(row.row_id, _sqlite_ts(row.ts), row.payload) for row in op.rows])
                self._mark_committed("sqlite", txn)
        finally:
            conn.close()

    def _load_from_sql(self) -> None:
        if not self.sql_db_path:
            return
        conn = sqlite3.connect(self.sql_db_path)
        try:
            raw = {table: conn.execute(query).fetchall() for table, query in self._load_queries("sqlite")}
        finally:
            conn.close()
        self._apply_loaded_rows(raw)
        self._seed_persisted_state("sqlite", {k: v for k, v in raw.items() if k != _ACTIVE_SURFACE_TABLE})
        self._recover_active_surfaces()
        self._compact_artifacts()
    def _ensure_postgres_schema(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import text

        engine = self._pg_engine()
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
        from sqlalchemy import text

        engine = self._pg_engine()
        for txn in self._chunk_ops(self._dirty_write_ops("postgres")):
            # One short transaction per <= _PERSIST_BATCH_ROWS rows, one
            # executemany per table slice: no transaction stays open across
            # the whole sweep, and none is ever idle between statements for
            # longer than it takes to send the next batch.
            with engine.begin() as conn:
                for op in txn:
                    if op.active_surface:
                        conn.execute(text(f"DELETE FROM {_ACTIVE_SURFACE_TABLE}"))
                        if op.rows:
                            conn.execute(
                                text(
                                    f"INSERT INTO {_ACTIVE_SURFACE_TABLE}(target_surface, adoption_id, updated_at) "
                                    "VALUES (:surface, :adoption_id, :updated_at) "
                                    "ON CONFLICT (target_surface) DO UPDATE SET adoption_id = EXCLUDED.adoption_id, updated_at = EXCLUDED.updated_at"
                                ),
                                [{"surface": row.row_id, "adoption_id": row.payload, "updated_at": row.ts} for row in op.rows],
                            )
                    else:
                        conn.execute(text(op.sql), [{"id": row.row_id, "ts": row.ts, "payload": row.payload} for row in op.rows])
            self._mark_committed("postgres", txn)

    def _load_from_postgres(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import text

        engine = self._pg_engine()
        # Fetch raw rows only inside the transaction; the (slow, 550k-row)
        # pydantic parse happens after it has closed, so the load never sits
        # "idle in transaction" while Python works.
        with engine.begin() as conn:
            raw = {table: [tuple(r) for r in conn.execute(text(query)).fetchall()] for table, query in self._load_queries("postgres")}
        self._apply_loaded_rows(raw)
        self._seed_persisted_state("postgres", {k: v for k, v in raw.items() if k != _ACTIVE_SURFACE_TABLE})
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
                self._note_signal_persisted("postgres", signal)
                return True
            except Exception:
                pass
        if self.sql_db_path:
            try:
                self._persist_signal_sqlite(signal)
                self._note_signal_persisted("sqlite", signal)
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
        from sqlalchemy import text

        engine = self._pg_engine()
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
                self._note_persisted("postgres", "substrate_mutation_pressure", pressure.pressure_id, _payload_json(pressure))
                self._source_kind = "postgres"
                self._last_error = None
                return True
            except Exception:
                pass
        if self.sql_db_path:
            try:
                self._persist_pressure_sqlite(pressure)
                self._note_persisted("sqlite", "substrate_mutation_pressure", pressure.pressure_id, _payload_json(pressure))
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
        from sqlalchemy import text

        engine = self._pg_engine()
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
                self._note_persisted("postgres", "substrate_mutation_proposal", proposal.proposal_id, _payload_json(proposal))
                self._source_kind = "postgres"
                self._last_error = None
                return True
            except Exception:
                pass
        if self.sql_db_path:
            try:
                self._persist_proposal_sqlite(proposal)
                self._note_persisted("sqlite", "substrate_mutation_proposal", proposal.proposal_id, _payload_json(proposal))
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
        from sqlalchemy import text

        engine = self._pg_engine()
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
                self._note_persisted("postgres", "substrate_mutation_proposal", proposal.proposal_id, _payload_json(proposal))
                self._note_persisted("postgres", "substrate_mutation_queue", queue_item.queue_item_id, _payload_json(queue_item))
                self._source_kind = "postgres"
                self._last_error = None
                return True
            except Exception:
                pass
        if self.sql_db_path:
            try:
                self._persist_proposal_and_queue_item_sqlite(proposal, queue_item)
                self._note_persisted("sqlite", "substrate_mutation_proposal", proposal.proposal_id, _payload_json(proposal))
                self._note_persisted("sqlite", "substrate_mutation_queue", queue_item.queue_item_id, _payload_json(queue_item))
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
        from sqlalchemy import text

        engine = self._pg_engine()
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
