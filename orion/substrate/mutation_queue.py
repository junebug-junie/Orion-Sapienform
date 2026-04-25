from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import sqlite3

from orion.core.schemas.substrate_mutation import (
    MutationAdoptionV1,
    MutationDecisionV1,
    MutationPressureV1,
    MutationProposalV1,
    MutationQueueItemV1,
    MutationRollbackV1,
    MutationSignalV1,
    MutationTrialV1,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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

    def __post_init__(self) -> None:
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
        self._persist()

    def record_pressure(self, pressure: MutationPressureV1) -> None:
        key = f"{pressure.anchor_scope}|{pressure.subject_ref}|{pressure.target_surface}"
        self._pressures[key] = pressure
        self._persist()

    def add_proposal(self, proposal: MutationProposalV1, *, priority: int = 50) -> MutationQueueItemV1:
        self._proposals[proposal.proposal_id] = proposal
        queue_item = MutationQueueItemV1(
            proposal_id=proposal.proposal_id,
            mutation_class=proposal.mutation_class,
            target_surface=proposal.target_surface,
            priority=priority,
        )
        self._queue[queue_item.queue_item_id] = queue_item
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
        self._trials[trial.trial_id] = trial
        proposal = self._proposals.get(trial.proposal_id)
        if proposal is not None:
            self._proposals[proposal.proposal_id] = proposal.model_copy(update={"rollout_state": "trialed"})
        self._persist()

    def record_decision(self, decision: MutationDecisionV1) -> None:
        self._decisions[decision.decision_id] = decision
        proposal = self._proposals.get(decision.proposal_id)
        if proposal is not None:
            next_state = "approved" if decision.action in {"auto_promote", "require_review"} else "rejected"
            self._proposals[proposal.proposal_id] = proposal.model_copy(update={"rollout_state": next_state})
        self._persist()

    def record_adoption(self, adoption: MutationAdoptionV1) -> list[str]:
        target_surface = adoption.target_surface
        existing = self._active_surface_by_target.get(target_surface)
        if existing and existing != adoption.adoption_id:
            return ["active_mutation_exists_for_target_surface"]
        self._active_surface_by_target[target_surface] = adoption.adoption_id
        self._adoptions[adoption.adoption_id] = adoption
        proposal = self._proposals.get(adoption.proposal_id)
        if proposal is not None:
            self._proposals[proposal.proposal_id] = proposal.model_copy(update={"rollout_state": "applied"})
        self._persist()
        return []

    def record_rollback(self, rollback: MutationRollbackV1) -> None:
        self._rollbacks[rollback.rollback_id] = rollback
        adoption = self._adoptions.get(rollback.adoption_id)
        if adoption is not None:
            self._adoptions[rollback.adoption_id] = adoption.model_copy(update={"status": "rolled_back"})
            self._active_surface_by_target.pop(adoption.target_surface, None)
        proposal = self._proposals.get(rollback.proposal_id)
        if proposal is not None:
            self._proposals[rollback.proposal_id] = proposal.model_copy(update={"rollout_state": "rolled_back"})
        self._persist()

    def active_surface(self, target_surface: str) -> str | None:
        return self._active_surface_by_target.get(target_surface)

    def latest_trials_by_proposal(self) -> dict[str, MutationTrialV1]:
        result: dict[str, MutationTrialV1] = {}
        for trial in self._trials.values():
            prev = result.get(trial.proposal_id)
            if prev is None or trial.created_at > prev.created_at:
                result[trial.proposal_id] = trial
        return result

    def _persist(self) -> None:
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
            conn.execute("CREATE TABLE IF NOT EXISTS substrate_mutation_active_surface (target_surface TEXT PRIMARY KEY, adoption_id TEXT NOT NULL, updated_at TEXT NOT NULL)")
            conn.commit()

    def _persist_to_sql(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute("DELETE FROM substrate_mutation_signal")
            for item in self._signals:
                conn.execute("INSERT INTO substrate_mutation_signal(signal_id, detected_at, payload_json) VALUES (?, ?, ?)", (item.signal_id, item.detected_at.isoformat(), json.dumps(item.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)))
            conn.execute("DELETE FROM substrate_mutation_pressure")
            for item in self._pressures.values():
                conn.execute("INSERT INTO substrate_mutation_pressure(pressure_id, updated_at, payload_json) VALUES (?, ?, ?)", (item.pressure_id, item.updated_at.isoformat(), json.dumps(item.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)))
            conn.execute("DELETE FROM substrate_mutation_proposal")
            for item in self._proposals.values():
                conn.execute("INSERT INTO substrate_mutation_proposal(proposal_id, created_at, payload_json) VALUES (?, ?, ?)", (item.proposal_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)))
            conn.execute("DELETE FROM substrate_mutation_queue")
            for item in self._queue.values():
                conn.execute("INSERT INTO substrate_mutation_queue(queue_item_id, created_at, payload_json) VALUES (?, ?, ?)", (item.queue_item_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)))
            conn.execute("DELETE FROM substrate_mutation_trial")
            for item in self._trials.values():
                conn.execute("INSERT INTO substrate_mutation_trial(trial_id, created_at, payload_json) VALUES (?, ?, ?)", (item.trial_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)))
            conn.execute("DELETE FROM substrate_mutation_decision")
            for item in self._decisions.values():
                conn.execute("INSERT INTO substrate_mutation_decision(decision_id, created_at, payload_json) VALUES (?, ?, ?)", (item.decision_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)))
            conn.execute("DELETE FROM substrate_mutation_adoption")
            for item in self._adoptions.values():
                conn.execute("INSERT INTO substrate_mutation_adoption(adoption_id, created_at, payload_json) VALUES (?, ?, ?)", (item.adoption_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)))
            conn.execute("DELETE FROM substrate_mutation_rollback")
            for item in self._rollbacks.values():
                conn.execute("INSERT INTO substrate_mutation_rollback(rollback_id, created_at, payload_json) VALUES (?, ?, ?)", (item.rollback_id, item.created_at.isoformat(), json.dumps(item.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)))
            conn.execute("DELETE FROM substrate_mutation_active_surface")
            for surface, adoption_id in self._active_surface_by_target.items():
                conn.execute("INSERT INTO substrate_mutation_active_surface(target_surface, adoption_id, updated_at) VALUES (?, ?, ?)", (surface, adoption_id, _utc_now().isoformat()))
            conn.commit()

    def _load_from_sql(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            self._signals = [MutationSignalV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_signal ORDER BY detected_at ASC").fetchall()]
            self._pressures = {item.pressure_id: item for item in [MutationPressureV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_pressure ORDER BY updated_at ASC").fetchall()]}
            self._proposals = {item.proposal_id: item for item in [MutationProposalV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_proposal ORDER BY created_at ASC").fetchall()]}
            self._queue = {item.queue_item_id: item for item in [MutationQueueItemV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_queue ORDER BY created_at ASC").fetchall()]}
            self._trials = {item.trial_id: item for item in [MutationTrialV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_trial ORDER BY created_at ASC").fetchall()]}
            self._decisions = {item.decision_id: item for item in [MutationDecisionV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_decision ORDER BY created_at ASC").fetchall()]}
            self._adoptions = {item.adoption_id: item for item in [MutationAdoptionV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_adoption ORDER BY created_at ASC").fetchall()]}
            self._rollbacks = {item.rollback_id: item for item in [MutationRollbackV1.model_validate(json.loads(p)) for (p,) in conn.execute("SELECT payload_json FROM substrate_mutation_rollback ORDER BY created_at ASC").fetchall()]}
            self._active_surface_by_target = {surface: adoption_id for (surface, adoption_id) in conn.execute("SELECT target_surface, adoption_id FROM substrate_mutation_active_surface").fetchall()}

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
            "CREATE TABLE IF NOT EXISTS substrate_mutation_active_surface (target_surface TEXT PRIMARY KEY, adoption_id TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL)",
        ]
        with engine.begin() as conn:
            for statement in ddl:
                conn.execute(text(statement))

    def _persist_to_postgres(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        def _replace(table: str, rows: list[tuple[str, datetime, str]]) -> None:
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {table}"))
                for row_id, created_at, payload in rows:
                    conn.execute(text(f"INSERT INTO {table} VALUES (:id, :created_at, CAST(:payload AS JSONB))"), {"id": row_id, "created_at": created_at, "payload": payload})

        _replace("substrate_mutation_signal", [(x.signal_id, x.detected_at, json.dumps(x.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)) for x in self._signals])
        _replace("substrate_mutation_pressure", [(x.pressure_id, x.updated_at, json.dumps(x.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)) for x in self._pressures.values()])
        _replace("substrate_mutation_proposal", [(x.proposal_id, x.created_at, json.dumps(x.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)) for x in self._proposals.values()])
        _replace("substrate_mutation_queue", [(x.queue_item_id, x.created_at, json.dumps(x.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)) for x in self._queue.values()])
        _replace("substrate_mutation_trial", [(x.trial_id, x.created_at, json.dumps(x.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)) for x in self._trials.values()])
        _replace("substrate_mutation_decision", [(x.decision_id, x.created_at, json.dumps(x.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)) for x in self._decisions.values()])
        _replace("substrate_mutation_adoption", [(x.adoption_id, x.created_at, json.dumps(x.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)) for x in self._adoptions.values()])
        _replace("substrate_mutation_rollback", [(x.rollback_id, x.created_at, json.dumps(x.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)) for x in self._rollbacks.values()])
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM substrate_mutation_active_surface"))
            for surface, adoption_id in self._active_surface_by_target.items():
                conn.execute(
                    text("INSERT INTO substrate_mutation_active_surface(target_surface, adoption_id, updated_at) VALUES (:surface, :adoption_id, :updated_at)"),
                    {"surface": surface, "adoption_id": adoption_id, "updated_at": _utc_now()},
                )

    def _load_from_postgres(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            self._signals = [MutationSignalV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_signal ORDER BY detected_at ASC")).fetchall()]
            self._pressures = {item.pressure_id: item for item in [MutationPressureV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_pressure ORDER BY updated_at ASC")).fetchall()]}
            self._proposals = {item.proposal_id: item for item in [MutationProposalV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_proposal ORDER BY created_at ASC")).fetchall()]}
            self._queue = {item.queue_item_id: item for item in [MutationQueueItemV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_queue ORDER BY created_at ASC")).fetchall()]}
            self._trials = {item.trial_id: item for item in [MutationTrialV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_trial ORDER BY created_at ASC")).fetchall()]}
            self._decisions = {item.decision_id: item for item in [MutationDecisionV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_decision ORDER BY created_at ASC")).fetchall()]}
            self._adoptions = {item.adoption_id: item for item in [MutationAdoptionV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_adoption ORDER BY created_at ASC")).fetchall()]}
            self._rollbacks = {item.rollback_id: item for item in [MutationRollbackV1.model_validate(json.loads(p)) for (p,) in conn.execute(text("SELECT payload_json::text FROM substrate_mutation_rollback ORDER BY created_at ASC")).fetchall()]}
            self._active_surface_by_target = {surface: adoption_id for (surface, adoption_id) in conn.execute(text("SELECT target_surface, adoption_id FROM substrate_mutation_active_surface")).fetchall()}
