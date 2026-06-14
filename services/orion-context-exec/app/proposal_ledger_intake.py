"""Opt-in proposal ledger intake for context-exec ProposalEnvelopeV1 outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from orion.schemas.context_exec import (
    MemoryCorrectionProposalV1,
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


def _deterministic_auto_triage(
    envelope: ProposalEnvelopeV1,
) -> ProposalTriageDecisionV1 | None:
    """Conservative deterministic triage — no LLM, no auto-approve."""
    if envelope.risk in {"high", "unknown"}:
        return ProposalTriageDecisionV1(
            proposal_id=envelope.proposal_id,
            action="promote_to_review",
            rationale="high or unknown risk envelope",
            attention_required=True,
            attention_reason="high or unknown risk",
        )

    if envelope.artifact_type == "MemoryCorrectionProposalV1":
        correction = MemoryCorrectionProposalV1.model_validate(envelope.artifact)
        domains = {d.lower() for d in correction.target_memory_domains}
        if domains & {"identity", "core"}:
            return ProposalTriageDecisionV1(
                proposal_id=envelope.proposal_id,
                action="promote_to_review",
                rationale="identity or core memory correction",
                attention_required=True,
                attention_reason="identity or core memory correction",
            )
        if correction.confidence < 0.3:
            return ProposalTriageDecisionV1(
                proposal_id=envelope.proposal_id,
                action="block_for_evidence",
                rationale="low confidence memory correction",
                attention_required=True,
                attention_reason="low confidence",
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
            triage = _deterministic_auto_triage(envelope)
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
