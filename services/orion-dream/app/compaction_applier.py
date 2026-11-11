"""Phase G — compaction applier (memory mutation, hard-gated).

THIS IS THE ONE RUNG THAT WRITES MEMORY. Everything here is built to be inert
until three independent conditions all hold, asked in this order:

  1. an injected **consent** callable says yes for THIS delta and THIS plan (downscale only, or
     prune too). Since attention patch 3d (2026-11-11) that is the want market's consent gate:
     `app.compaction_consent.market_consent` bids for family `dream.compaction` with a consent
     request, and the market answers from config/physiology/consent.yaml (downscale: needs
     Juniper's standing consent; prune: irreversible, always refused). It is asked FIRST, even
     while apply is off, so every staged delta leaves a verdict in want_ledger and a
     `compaction_apply not_approved ... rule=...` log line: the live proof the gate answers.
     No consent (no callable, no answer, refuse, needs_consent) -> `not_approved`. (Until
     2026-11-07 this was a policy-rung decision frame; the policy rung retired, ladder PR 6.);
  2. `ORION_DREAM_COMPACTION_APPLY_ENABLED` is true (the hot gate — default off) -> else `disabled`;
  3. an injected `CompactionMemoryStore` is provided (there is no default
     canonical-memory binding in this module — the real adapter is wired only in
     a dedicated, signed-off change, so importing this module touches nothing) -> else `no_store`.

Safety invariants (§14 backfill protocol + §0A):
  - **snapshot precedes every apply**: the before-state of every target is
    captured to `DREAM_COMPACTION_SNAPSHOT_DIR/<delta_id>/` before any write;
  - **rollback on any error**: a failure mid-apply restores from the snapshot;
  - **downscale-renormalize first, prune last**: with `DOWNSCALE_ONLY` (default
    true) prune is skipped entirely — never prune before downscale is trusted;
  - **consolidate is dream-provenance**: gist cards are written with
    `source_kind="dream"` and never promoted to fact;
  - **caps**: the applier honors the schema caps already on the delta.

Deterministic (§4): given (delta, consent answer, store) the applier is a pure control
flow — no LLM. It degrades to a typed no-op result and never raises.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field

from orion.schemas.compaction import MemoryCompactionDeltaV1

from app.settings import settings

logger = logging.getLogger("orion-dream.compaction_applier")

ApplyStatus = Literal[
    "not_approved",  # no consent for this delta and plan (asked first; fails closed)
    "disabled",  # consented, but the hot gate is off
    "no_store",  # consented and on, but no memory store is wired (nothing can be written)
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
    # The consent gate's answer (attention patch 3d): allow | refuse | needs_consent, the consent.yaml
    # rule, and why. None: no gate answered (no callable, no market answer).
    consent_verdict: str | None = None
    consent_rule: str | None = None
    consent_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ConsentAnswer:
    """What a consent authority said about one delta and plan. ``approved`` is the only field the applier
    obeys; the rest is the trace (the market's verdict, its rule, its reason, the want_ledger decision)."""

    approved: bool
    verdict: Optional[str] = None
    rule_id: Optional[str] = None
    reason: str = ""
    decision_id: Optional[str] = None


#: Asked once per apply, with the exact delta and whether this apply is downscale only (prune skipped).
#: Approves only THIS delta (not a sibling that shares a source request) and THIS plan (a consent to
#: downscale is not a consent to prune). Returns a ``ConsentAnswer`` or a bare bool. Never raises into
#: the applier: an exception is no.
CompactionConsent = Callable[[MemoryCompactionDeltaV1, bool], Union[bool, ConsentAnswer]]


def ask_consent(delta: MemoryCompactionDeltaV1, consent: CompactionConsent | None, downscale_only: bool) -> ConsentAnswer:
    """The consent answer for this exact delta and plan. Fails closed: no consent callable, an answer
    that is not ``True`` / an approved ``ConsentAnswer``, or any exception is a no."""
    if consent is None:
        return ConsentAnswer(False, reason="no consent authority")
    try:
        ans = consent(delta, downscale_only)
    except Exception as exc:  # noqa: BLE001
        return ConsentAnswer(False, reason=f"consent raised {type(exc).__name__}")
    if isinstance(ans, ConsentAnswer):
        return ans if ans.approved is True else ConsentAnswer(False, ans.verdict, ans.rule_id, ans.reason, ans.decision_id)
    return ConsentAnswer(ans is True, reason="" if ans is True else "consent said no")


def consent_approves(delta: MemoryCompactionDeltaV1, consent: CompactionConsent | None,
                     downscale_only: bool = True) -> bool:
    """True only if ``consent`` exists and says yes for this exact delta and plan."""
    return ask_consent(delta, consent, downscale_only).approved


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
    consent: CompactionConsent | None = None,
    *,
    store: CompactionMemoryStore | None,
    enabled: bool | None = None,
    downscale_only: bool | None = None,
) -> CompactionApplyReceiptV1:
    """Apply a consented compaction delta to memory. Snapshot-first, with
    rollback on any error. Returns a typed receipt; never raises.

    Gates, in order: consent (asked even while apply is off, so the gate leaves a live trace) →
    the hot gate → a store. Then the order is fixed: snapshot → downscale-renormalize → (prune
    unless downscale_only) → consolidate gist cards. Any exception after the snapshot triggers a
    restore.
    """
    enabled = settings.ORION_DREAM_COMPACTION_APPLY_ENABLED if enabled is None else enabled
    downscale_only = (
        settings.ORION_DREAM_COMPACTION_DOWNSCALE_ONLY if downscale_only is None else downscale_only
    )

    answer = ask_consent(delta, consent, bool(downscale_only))
    trace = {"consent_verdict": answer.verdict, "consent_rule": answer.rule_id,
             "consent_reason": answer.reason or None}
    if not answer.approved:
        logger.info("compaction_apply not_approved delta=%s verdict=%s rule=%s decision=%s reason=%s apply_enabled=%s",
                    delta.delta_id, answer.verdict, answer.rule_id, answer.decision_id, answer.reason, bool(enabled))
        return CompactionApplyReceiptV1(delta_id=delta.delta_id, status="not_approved", **trace)

    if not enabled:
        logger.info("compaction_apply disabled delta=%s verdict=%s rule=%s (consented; the hot gate is off)",
                    delta.delta_id, answer.verdict, answer.rule_id)
        return CompactionApplyReceiptV1(delta_id=delta.delta_id, status="disabled", **trace)

    if store is None:
        logger.warning("compaction_apply no_store delta=%s rule=%s (consented and on, but no memory store is wired)",
                       delta.delta_id, answer.rule_id)
        return CompactionApplyReceiptV1(delta_id=delta.delta_id, status="no_store", **trace)

    if delta.is_empty():
        return CompactionApplyReceiptV1(delta_id=delta.delta_id, status="empty", **trace)

    # §14: snapshot the before-state BEFORE any write. A snapshot failure fails
    # closed — we do not apply what we cannot roll back.
    try:
        snapshot = store.snapshot(delta)
        snapshot_path = _write_snapshot_file(delta.delta_id, snapshot)
    except Exception as exc:
        logger.warning("compaction apply aborted: snapshot failed for %s: %s", delta.delta_id, exc)
        return CompactionApplyReceiptV1(
            delta_id=delta.delta_id, status="rolled_back", error=f"snapshot_failed: {exc}", **trace
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
                **trace,
            )
        return CompactionApplyReceiptV1(
            delta_id=delta.delta_id,
            status="rolled_back",
            snapshot_path=snapshot_path,
            error=f"apply_failed_rolled_back: {exc}",
            **trace,
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
        **trace,
    )
