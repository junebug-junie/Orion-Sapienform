"""Phase G — compaction applier (memory mutation, hard-gated).

THIS IS THE ONE RUNG THAT WRITES MEMORY. Everything here is built to be inert
until three independent conditions all hold:

  1. `ORION_DREAM_COMPACTION_APPLY_ENABLED` is true (the hot gate — default off);
  2. the delta's proposal was **policy-approved for execution** — and reverie
     proposals carry `operator_review`, so a human must have approved it;
  3. an injected `CompactionMemoryStore` is provided (there is no default
     canonical-memory binding in this module — the real adapter is wired only in
     a dedicated, signed-off change, so importing this module touches nothing).

Safety invariants (§14 backfill protocol + §0A):
  - **snapshot precedes every apply**: the before-state of every target is
    captured to `DREAM_COMPACTION_SNAPSHOT_DIR/<delta_id>/` before any write;
  - **rollback on any error**: a failure mid-apply restores from the snapshot;
  - **downscale-renormalize first, prune last**: with `DOWNSCALE_ONLY` (default
    true) prune is skipped entirely — never prune before downscale is trusted;
  - **consolidate is dream-provenance**: gist cards are written with
    `source_kind="dream"` and never promoted to fact;
  - **caps**: the applier honors the schema caps already on the delta.

Deterministic (§4): given (delta, approval, store) the applier is a pure control
flow — no LLM. It degrades to a typed no-op result and never raises.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from orion.schemas.compaction import MemoryCompactionDeltaV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1

from app.settings import settings

logger = logging.getLogger("orion-dream.compaction_applier")

ApplyStatus = Literal[
    "disabled",  # hot gate off
    "not_approved",  # no execution-approved policy decision for this delta
    "empty",  # nothing to apply
    "applied",  # applied cleanly (snapshot taken first)
    "rolled_back",  # a failure mid-apply → restored from snapshot
]


class CompactionMemoryStore(Protocol):
    """The narrow, injected memory surface the applier is allowed to touch.

    A real (canonical) implementation is provided ONLY in a dedicated, signed-off
    change — never imported here — so this module is inert by construction.
    """

    def snapshot(self, delta: MemoryCompactionDeltaV1) -> dict[str, Any]:
        """Capture the before-state of every target the delta touches (§14)."""
        ...

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore targets to their snapshotted before-state (rollback)."""
        ...

    def downscale(self, target_id: str, old_w: float, new_w: float) -> None:
        """Renormalize one edge/weight downward. Never deletes."""
        ...

    def prune(self, episodic_id: str) -> None:
        """Remove one low-salience episodic (only after downscale is trusted)."""
        ...

    def write_gist(self, gist_card: str, evidence_refs: list[str], supersedes: list[str]) -> None:
        """Crystallize a gist card with source_kind='dream' (never promoted to fact)."""
        ...


class CompactionApplyReceiptV1(BaseModel):
    """Inspectable trace of an apply attempt — proof the live path did (or didn't)
    move, and exactly what it touched. Not bus-published; returned + logged."""

    delta_id: str
    status: ApplyStatus
    applied: bool = False
    downscaled: int = 0
    pruned: int = 0
    cards_written: int = 0
    prune_skipped_downscale_only: int = 0
    snapshot_path: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def policy_approves_execution(
    delta: MemoryCompactionDeltaV1,
    policy_frame: PolicyDecisionFrameV1 | None,
) -> bool:
    """True only if this delta's proposal was approved *for execution*.

    Reverie proposals carry `operator_review`, so absent an explicit human
    approval this returns False and the applier stays inert. A missing frame,
    a frame that doesn't reference this delta, or any non-execution decision all
    fail closed. Any malformed frame also fails closed (never raises).
    """
    try:
        if policy_frame is None or not policy_frame.execution_allowed:
            return False
        for decision in policy_frame.approved_decisions:
            if decision.decision != "approved_for_execution":
                continue
            # Memory mutation requires the human-review execution gate. The
            # autonomy engine's self-authorized `autonomy_policy` gate is NOT
            # accepted here — a human must approve this rung (§0A).
            if decision.policy_gate != "execution_policy":
                continue
            # The approval must reference THIS exact delta. A shared
            # `source_request_id` is deliberately NOT enough: two deltas from one
            # night can answer the same request with different ops, so a human who
            # reviewed delta A must not thereby authorize un-reviewed delta B.
            refs = {decision.proposal_id, *decision.evidence_refs}
            if delta.delta_id in refs:
                return True
        return False
    except Exception:
        # A malformed/duck-typed frame must never crash the gate — fail closed.
        return False


def _write_snapshot_file(delta_id: str, snapshot: dict[str, Any]) -> str | None:
    """Persist the before-state to DREAM_COMPACTION_SNAPSHOT_DIR/<delta_id>/ (§14).

    Best-effort: a snapshot-write failure aborts the apply (fail closed), handled
    by the caller. Returns the path on success.
    """
    root = settings.DREAM_COMPACTION_SNAPSHOT_DIR
    job_dir = os.path.join(root, delta_id)
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "snapshot_before.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, default=str)
    return path


def apply_compaction_delta(
    delta: MemoryCompactionDeltaV1,
    policy_frame: PolicyDecisionFrameV1 | None,
    *,
    store: CompactionMemoryStore,
    enabled: bool | None = None,
    downscale_only: bool | None = None,
) -> CompactionApplyReceiptV1:
    """Apply a policy-approved compaction delta to memory. Snapshot-first, with
    rollback on any error. Returns a typed receipt; never raises.

    Order is fixed: snapshot → downscale-renormalize → (prune unless downscale_only)
    → consolidate gist cards. Any exception after the snapshot triggers a restore.
    """
    enabled = settings.ORION_DREAM_COMPACTION_APPLY_ENABLED if enabled is None else enabled
    downscale_only = (
        settings.ORION_DREAM_COMPACTION_DOWNSCALE_ONLY if downscale_only is None else downscale_only
    )

    if not enabled:
        logger.info("compaction apply disabled; delta %s not applied", delta.delta_id)
        return CompactionApplyReceiptV1(delta_id=delta.delta_id, status="disabled")

    if not policy_approves_execution(delta, policy_frame):
        logger.info("compaction apply blocked: no execution approval for %s", delta.delta_id)
        return CompactionApplyReceiptV1(delta_id=delta.delta_id, status="not_approved")

    if delta.is_empty():
        return CompactionApplyReceiptV1(delta_id=delta.delta_id, status="empty")

    # §14: snapshot the before-state BEFORE any write. A snapshot failure fails
    # closed — we do not apply what we cannot roll back.
    try:
        snapshot = store.snapshot(delta)
        snapshot_path = _write_snapshot_file(delta.delta_id, snapshot)
    except Exception as exc:
        logger.warning("compaction apply aborted: snapshot failed for %s: %s", delta.delta_id, exc)
        return CompactionApplyReceiptV1(
            delta_id=delta.delta_id, status="rolled_back", error=f"snapshot_failed: {exc}"
        )

    downscaled = pruned = cards = prune_skipped = 0
    try:
        # 1) downscale-renormalize (the safer subset — always first).
        for entry in delta.downscale:
            store.downscale(entry.target_id, entry.old_w, entry.new_w)
            downscaled += 1

        # 2) prune — only once downscale is trusted (downscale_only=false).
        if downscale_only:
            prune_skipped = len(delta.prune)
        else:
            for entry in delta.prune:
                store.prune(entry.episodic_id)
                pruned += 1

        # 3) consolidate — dream-provenance gist cards, never promoted to fact.
        for entry in delta.consolidate:
            store.write_gist(entry.gist_card, list(entry.evidence_refs), list(entry.supersedes))
            cards += 1
    except Exception as exc:
        logger.warning("compaction apply failed mid-flight for %s: %s — rolling back", delta.delta_id, exc)
        try:
            store.restore(snapshot)
        except Exception as rexc:
            logger.error("ROLLBACK FAILED for %s: %s (snapshot at %s)", delta.delta_id, rexc, snapshot_path)
            return CompactionApplyReceiptV1(
                delta_id=delta.delta_id,
                status="rolled_back",
                snapshot_path=snapshot_path,
                error=f"apply_failed_and_rollback_failed: {exc} / {rexc}",
            )
        return CompactionApplyReceiptV1(
            delta_id=delta.delta_id,
            status="rolled_back",
            snapshot_path=snapshot_path,
            error=f"apply_failed_rolled_back: {exc}",
        )

    logger.info(
        "compaction applied delta=%s downscaled=%d pruned=%d cards=%d prune_skipped=%d",
        delta.delta_id, downscaled, pruned, cards, prune_skipped,
    )
    return CompactionApplyReceiptV1(
        delta_id=delta.delta_id,
        status="applied",
        applied=True,
        downscaled=downscaled,
        pruned=pruned,
        cards_written=cards,
        prune_skipped_downscale_only=prune_skipped,
        snapshot_path=snapshot_path,
    )
