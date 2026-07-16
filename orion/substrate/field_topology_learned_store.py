"""Causal Geometry v1, Rung 2B: HITL queue + adopted-overlay store for field-topology
weight-patch proposals (`field_topology_weight_patch`, `orion/substrate/mutation_contracts.py`).

Mirrors the shape of `orion/substrate/mutation_queue.py`'s `SubstrateMutationStore`
(in-memory by default, optional sqlite-backed persistence via `sql_db_path`) but is a
separate, much smaller store: this class is the *only* way a learned edge-weight delta
becomes active. `adopt()` is an explicit human/operator action -- nothing in this rung,
and nothing this rung wires up, may reach the overlay any other way.

Hard constraint (see PR description / task spec): this module must never import
`orion.substrate.mutation_apply.PatchApplier`. That class is the generic
auto-promote path (`if decision.action != "auto_promote": return None`), and
`field_topology_weight_patch` has `auto_promote_default=False` specifically so it
never travels that path. Adoption here is a distinct action a later rung's hub UI
will trigger directly against this store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import sqlite3
from typing import Literal

from orion.core.schemas.substrate_mutation import MutationProposalV1

# Half-life used for `decay_half_life_hours` and the default when
# FIELD_PLASTICITY_DECAY_HALF_LIFE_HOURS is unset in the digester's .env_example.
# 168h = 7 days: an adopted delta should meaningfully fade back toward the designed
# YAML weight within about a week absent reinforcement (repeated re-adoption), per the
# design spec's "decay-to-designed" containment for plasticity degeneracy.
DEFAULT_DECAY_HALF_LIFE_HOURS = 168.0

# Below this magnitude, the decayed effective delta is indistinguishable from "no
# override" -- current_overlay() drops entries under this threshold instead of
# returning noise-level non-zero deltas forever.
EFFECTIVE_DELTA_EPSILON = 1e-4

ProposalStatus = Literal["pending_review", "adopted", "rejected"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AdoptedWeightEntry:
    edge_id: str
    proposal_id: str
    adopted_delta: float
    adopted_at: datetime
    operator_id: str


@dataclass
class FieldTopologyLearnedWeightsStore:
    # Decay half-life in hours: current_overlay()'s effective delta is
    # adopted_delta * 0.5 ** (hours_elapsed / half_life_hours), i.e. it actually
    # halves at t=half_life_hours (a true half-life, not an e-folding time constant).
    half_life_hours: float = field(default_factory=lambda: _env_float("FIELD_PLASTICITY_DECAY_HALF_LIFE_HOURS", DEFAULT_DECAY_HALF_LIFE_HOURS))
    sql_db_path: str | None = None
    _source_kind: str = field(default="memory", init=False)
    _last_error: str | None = field(default=None, init=False)
    _proposals: dict[str, MutationProposalV1] = field(default_factory=dict, init=False)
    _status: dict[str, ProposalStatus] = field(default_factory=dict, init=False)
    _rejections: dict[str, dict[str, str]] = field(default_factory=dict, init=False)
    _adopted: dict[str, AdoptedWeightEntry] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.sql_db_path:
            try:
                self._ensure_sql_schema()
                self._load_from_sql()
                self._source_kind = "sqlite"
            except Exception as exc:  # pragma: no cover - defensive, mirrors mutation_queue.py's fallback pattern
                self._source_kind = "fallback"
                self._last_error = str(exc)

    # -- SUBSTRATE_REVIEW_QUEUE_STORE-shaped introspection methods (hub rung reads these) --

    def source_kind(self) -> str:
        return self._source_kind

    def degraded(self) -> bool:
        return self._source_kind == "fallback" or self._last_error is not None

    def last_error(self) -> str | None:
        return self._last_error

    # -- HITL queue --

    def propose(self, proposal: MutationProposalV1) -> None:
        self._proposals[proposal.proposal_id] = proposal
        self._status[proposal.proposal_id] = "pending_review"
        self._persist()

    def list_pending(self, limit: int = 50) -> list[MutationProposalV1]:
        rows = [proposal for proposal in self._proposals.values() if self._status.get(proposal.proposal_id) == "pending_review"]
        rows.sort(key=lambda item: item.created_at)
        return rows[:limit]

    def get_proposal(self, proposal_id: str) -> MutationProposalV1 | None:
        return self._proposals.get(proposal_id)

    def status_for(self, proposal_id: str) -> ProposalStatus | None:
        return self._status.get(proposal_id)

    def adopt(self, proposal_id: str, *, operator_id: str, now: datetime | None = None) -> dict[str, object]:
        """The ONLY entry point by which a proposed delta becomes active.

        Explicit HITL action -- operator_id is required and recorded. Returns a
        result dict with `ok: bool` rather than raising, so a hub caller can surface
        `reason` directly without a try/except.
        """
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return {"ok": False, "reason": "proposal_not_found"}
        status = self._status.get(proposal_id)
        if status != "pending_review":
            return {"ok": False, "reason": f"proposal_not_pending:{status}"}
        edge_id = proposal.patch.target_ref
        try:
            adopted_delta = float(proposal.patch.patch.get("edge_weight_delta", 0.0))
        except (TypeError, ValueError):
            return {"ok": False, "reason": "invalid_edge_weight_delta"}
        moment = now or _utc_now()
        entry = AdoptedWeightEntry(
            edge_id=edge_id,
            proposal_id=proposal_id,
            adopted_delta=adopted_delta,
            adopted_at=moment,
            operator_id=operator_id,
        )
        self._adopted[edge_id] = entry
        self._status[proposal_id] = "adopted"
        self._persist()
        return {
            "ok": True,
            "edge_id": edge_id,
            "adopted_delta": adopted_delta,
            "adopted_at": moment.isoformat(),
            "learned_at": moment.isoformat(),
            "operator_id": operator_id,
        }

    def reject(self, proposal_id: str, *, operator_id: str, reason: str = "") -> None:
        """Reject a pending proposal. A no-op on an already-adopted proposal --

        rejecting must never leave a live overlay entry (`self._adopted`) attached
        to a proposal whose visible status says "rejected". Undoing an adoption is a
        distinct action (not implemented here); this method only ever moves a
        proposal out of `pending_review`.
        """
        if proposal_id not in self._proposals:
            return
        if self._status.get(proposal_id) != "pending_review":
            return
        self._status[proposal_id] = "rejected"
        self._rejections[proposal_id] = {
            "operator_id": operator_id,
            "reason": reason,
            "rejected_at": _utc_now().isoformat(),
        }
        self._persist()

    # -- Adopted overlay (Rung 3A's diffusion read-path calls this) --

    def current_overlay(self, *, now: datetime | None = None) -> dict[str, float]:
        moment = now or _utc_now()
        half_life = max(self.half_life_hours, 1e-6)
        overlay: dict[str, float] = {}
        for edge_id, entry in self._adopted.items():
            hours_elapsed = max(0.0, (moment - entry.adopted_at).total_seconds() / 3600.0)
            decay_factor = 0.5 ** (hours_elapsed / half_life)
            effective_delta = entry.adopted_delta * decay_factor
            if abs(effective_delta) < EFFECTIVE_DELTA_EPSILON:
                continue
            overlay[edge_id] = effective_delta
        return overlay

    def adopted_entry(self, edge_id: str) -> AdoptedWeightEntry | None:
        return self._adopted.get(edge_id)

    # -- persistence (sqlite, optional; matches mutation_queue.py's optional durability pattern) --

    def _persist(self) -> None:
        if not self.sql_db_path:
            return
        try:
            self._persist_to_sql()
            self._source_kind = "sqlite"
            self._last_error = None
        except Exception as exc:
            self._source_kind = "fallback"
            self._last_error = str(exc)

    def _ensure_sql_schema(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS field_topology_learned_proposal (
                    proposal_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS field_topology_learned_adopted (
                    edge_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    adopted_delta REAL NOT NULL,
                    adopted_at TEXT NOT NULL,
                    operator_id TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS field_topology_learned_rejection (
                    proposal_id TEXT PRIMARY KEY,
                    operator_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    rejected_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _persist_to_sql(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            for proposal_id, proposal in self._proposals.items():
                conn.execute(
                    """
                    INSERT INTO field_topology_learned_proposal(proposal_id, status, created_at, payload_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(proposal_id) DO UPDATE SET
                        status=excluded.status,
                        payload_json=excluded.payload_json
                    """,
                    (
                        proposal_id,
                        self._status.get(proposal_id, "pending_review"),
                        proposal.created_at.isoformat(),
                        json.dumps(proposal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                    ),
                )
            for edge_id, entry in self._adopted.items():
                conn.execute(
                    """
                    INSERT INTO field_topology_learned_adopted(edge_id, proposal_id, adopted_delta, adopted_at, operator_id)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(edge_id) DO UPDATE SET
                        proposal_id=excluded.proposal_id,
                        adopted_delta=excluded.adopted_delta,
                        adopted_at=excluded.adopted_at,
                        operator_id=excluded.operator_id
                    """,
                    (edge_id, entry.proposal_id, entry.adopted_delta, entry.adopted_at.isoformat(), entry.operator_id),
                )
            for proposal_id, row in self._rejections.items():
                conn.execute(
                    """
                    INSERT INTO field_topology_learned_rejection(proposal_id, operator_id, reason, rejected_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(proposal_id) DO UPDATE SET
                        operator_id=excluded.operator_id,
                        reason=excluded.reason,
                        rejected_at=excluded.rejected_at
                    """,
                    (proposal_id, row.get("operator_id", ""), row.get("reason", ""), row.get("rejected_at", "")),
                )
            conn.commit()

    def _load_from_sql(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            for proposal_id, status, payload_json in conn.execute(
                "SELECT proposal_id, status, payload_json FROM field_topology_learned_proposal ORDER BY created_at ASC"
            ).fetchall():
                self._proposals[proposal_id] = MutationProposalV1.model_validate(json.loads(payload_json))
                self._status[proposal_id] = status
            for edge_id, proposal_id, adopted_delta, adopted_at, operator_id in conn.execute(
                "SELECT edge_id, proposal_id, adopted_delta, adopted_at, operator_id FROM field_topology_learned_adopted"
            ).fetchall():
                self._adopted[edge_id] = AdoptedWeightEntry(
                    edge_id=edge_id,
                    proposal_id=proposal_id,
                    adopted_delta=float(adopted_delta),
                    adopted_at=datetime.fromisoformat(adopted_at),
                    operator_id=operator_id,
                )
            for proposal_id, operator_id, reason, rejected_at in conn.execute(
                "SELECT proposal_id, operator_id, reason, rejected_at FROM field_topology_learned_rejection"
            ).fetchall():
                self._rejections[proposal_id] = {"operator_id": operator_id, "reason": reason, "rejected_at": rejected_at}
