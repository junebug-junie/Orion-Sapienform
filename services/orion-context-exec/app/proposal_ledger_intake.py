"""Opt-in proposal ledger intake for context-exec ProposalEnvelopeV1 outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from orion.schemas.context_exec import (
    MemoryCorrectionProposalV1,
    PatchProposalV1,
    ProposalEnvelopeV1,
)
from orion.schemas.proposal_ledger import (
    JsonFileProposalLedgerRepository,
    ProposalLedgerRecordV1,
    ProposalLedgerRepository,
    ProposalLedgerStoreError,
    ProposalTriageDecisionV1,
)

if TYPE_CHECKING:
    from .settings import ContextExecSettings

# Deterministic auto-triage thresholds — conservative, no LLM.
_LOW_CONFIDENCE_THRESHOLD = 0.25
_IDENTITY_BELIEF_MARKERS = (
    "denver",
    "identity",
    "user is from",
    "i am from",
    "core belief",
    "who i am",
)
_SAFETY_REVIEW_MARKERS = (
    "human review",
    "requires review",
    "safety concern",
    "safety note",
    "review before",
)


@dataclass(frozen=True)
class ProposalLedgerIntakeResult:
    persisted: bool
    proposal_id: str | None
    ledger_status: str | None
    attention_required: bool
    triage_action: str | None
    error: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _envelope_is_obviously_empty(envelope: ProposalEnvelopeV1) -> bool:
    return not (envelope.title or "").strip() and not (envelope.summary or "").strip()


def _text_indicates_insufficient_grounding(*texts: str | None) -> bool:
    for text in texts:
        if not text:
            continue
        lowered = text.lower()
        if "insufficient" in lowered or "not enough evidence" in lowered:
            return True
    return False


def _memory_is_identity_correction(correction: MemoryCorrectionProposalV1) -> bool:
    belief = correction.current_belief.lower()
    return any(marker in belief for marker in _IDENTITY_BELIEF_MARKERS)


def _safety_notes_warrant_review(envelope: ProposalEnvelopeV1) -> str | None:
    for note in envelope.safety_notes:
        lowered = note.lower()
        if any(marker in lowered for marker in _SAFETY_REVIEW_MARKERS):
            return note
    return None


def triage_proposal_envelope(envelope: ProposalEnvelopeV1) -> ProposalTriageDecisionV1 | None:
    """Deterministic auto-triage policy — store, block, or promote; never approve or execute."""
    memory: MemoryCorrectionProposalV1 | None = None
    patch: PatchProposalV1 | None = None
    if envelope.artifact_type == "MemoryCorrectionProposalV1":
        memory = MemoryCorrectionProposalV1.model_validate(envelope.artifact)
    elif envelope.artifact_type == "PatchProposalV1":
        patch = PatchProposalV1.model_validate(envelope.artifact)

    # Block for evidence — checked before promotion so garbage never reaches Juniper.
    if memory is not None:
        if memory.confidence < _LOW_CONFIDENCE_THRESHOLD:
            return ProposalTriageDecisionV1(
                proposal_id=envelope.proposal_id,
                action="block_for_evidence",
                rationale="inner artifact confidence below threshold",
                attention_required=True,
                attention_reason="low confidence",
            )
        if memory.missing_evidence and not memory.supporting_evidence:
            return ProposalTriageDecisionV1(
                proposal_id=envelope.proposal_id,
                action="block_for_evidence",
                rationale="missing evidence without supporting evidence",
                attention_required=True,
                attention_reason="missing evidence without support",
            )

    if _text_indicates_insufficient_grounding(
        envelope.summary,
        memory.rationale if memory else None,
        patch.proposed_change_summary if patch else None,
    ):
        return ProposalTriageDecisionV1(
            proposal_id=envelope.proposal_id,
            action="block_for_evidence",
            rationale="proposal indicates insufficient grounding",
            attention_required=True,
            attention_reason="insufficient grounding",
        )

    safety_reason = _safety_notes_warrant_review(envelope)
    if safety_reason:
        return ProposalTriageDecisionV1(
            proposal_id=envelope.proposal_id,
            action="promote_to_review",
            rationale=f"safety note warrants human review: {safety_reason}",
            attention_required=True,
            attention_reason=safety_reason,
        )

    if envelope.risk in {"high", "unknown"} and not _envelope_is_obviously_empty(envelope):
        return ProposalTriageDecisionV1(
            proposal_id=envelope.proposal_id,
            action="promote_to_review",
            rationale=f"{envelope.risk} risk envelope with grounded content",
            attention_required=True,
            attention_reason=f"{envelope.risk} risk",
        )

    if memory is not None and _memory_is_identity_correction(memory):
        if memory.confidence >= _LOW_CONFIDENCE_THRESHOLD:
            return ProposalTriageDecisionV1(
                proposal_id=envelope.proposal_id,
                action="promote_to_review",
                rationale="identity or core user belief memory correction",
                attention_required=True,
                attention_reason="memory correction involving identity",
            )

    return None


def maybe_persist_proposal_envelope(
    envelope: ProposalEnvelopeV1,
    settings: ContextExecSettings,
    *,
    repository: ProposalLedgerRepository | None = None,
    source_run_id: str | None = None,
) -> ProposalLedgerIntakeResult:
    """Persist a proposal envelope when ledger intake is explicitly enabled."""
    if not settings.context_exec_proposal_ledger_enabled:
        return ProposalLedgerIntakeResult(
            persisted=False,
            proposal_id=None,
            ledger_status=None,
            attention_required=False,
            triage_action=None,
            error=None,
        )

    store_path = (settings.context_exec_proposal_ledger_store_path or "").strip()
    if not store_path:
        return ProposalLedgerIntakeResult(
            persisted=False,
            proposal_id=envelope.proposal_id,
            ledger_status=None,
            attention_required=False,
            triage_action=None,
            error="store_path_required",
        )

    try:
        repo = repository or JsonFileProposalLedgerRepository(Path(store_path))
        record = ProposalLedgerRecordV1(
            proposal_id=envelope.proposal_id,
            envelope=envelope,
            status="stored",
            triage_action="store_only",
            attention_required=False,
            source_run_id=source_run_id or envelope.source_run_id,
            source_mode=envelope.source_mode,
            created_at=_utc_now(),
        )
        repo.store(record)

        if settings.context_exec_proposal_ledger_auto_triage:
            triage = triage_proposal_envelope(envelope)
            if triage is not None:
                record = repo.apply_triage(triage)

        return ProposalLedgerIntakeResult(
            persisted=True,
            proposal_id=record.proposal_id,
            ledger_status=record.status,
            attention_required=record.attention_required,
            triage_action=record.triage_action,
            error=None,
        )
    except ProposalLedgerStoreError as exc:
        return ProposalLedgerIntakeResult(
            persisted=False,
            proposal_id=envelope.proposal_id,
            ledger_status=None,
            attention_required=False,
            triage_action=None,
            error=str(exc),
        )


def intake_runtime_debug(result: ProposalLedgerIntakeResult, *, enabled: bool) -> dict[str, object]:
    debug: dict[str, object] = {
        "proposal_ledger_enabled": enabled,
        "proposal_ledger_persisted": result.persisted,
    }
    if result.proposal_id:
        debug["proposal_id"] = result.proposal_id
    if result.ledger_status:
        debug["ledger_status"] = result.ledger_status
    debug["attention_required"] = result.attention_required
    if result.triage_action:
        debug["triage_action"] = result.triage_action
    if result.error:
        debug["proposal_ledger_error"] = result.error
    return debug


def intake_final_text_line(result: ProposalLedgerIntakeResult) -> str | None:
    if not result.persisted or not result.proposal_id:
        return None
    return (
        f"Proposal stored in ledger: {result.proposal_id} "
        f"status={result.ledger_status} attention_required={str(result.attention_required).lower()}"
    )
