#!/usr/bin/env python3
"""Operator proposal ledger CLI — list, show, triage, review, eligibility."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if not p.endswith("/orion/schemas")]
sys.path.insert(0, str(ROOT))
# orion/schemas can shadow stdlib `platform`; restore before uuid imports it.
stdlib_platform = Path(sysconfig.get_paths()["stdlib"]) / "platform.py"
spec = importlib.util.spec_from_file_location("platform", stdlib_platform)
if spec and spec.loader:
    platform_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(platform_mod)
    sys.modules["platform"] = platform_mod

import uuid

from orion.schemas.context_exec import (  # noqa: E402
    MemoryCorrectionProposalV1,
    PatchProposalV1,
    ProposalEnvelopeV1,
    build_memory_correction_proposal_envelope,
    build_patch_proposal_envelope,
)
from orion.schemas.proposal_ledger import (  # noqa: E402
    JsonFileProposalLedgerRepository,
    ProposalLedgerRecordV1,
    ProposalLedgerStoreError,
    ProposalReviewDecisionV1,
    ProposalTriageDecisionV1,
)
from orion.schemas.proposal_lifecycle import derive_execution_eligibility  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_store(path: str | None) -> Path:
    if not path:
        raise SystemExit("--store is required (explicit JSON path; no repo writes)")
    return Path(path)


def _open_repo(store_path: Path) -> JsonFileProposalLedgerRepository:
    try:
        return JsonFileProposalLedgerRepository(store_path)
    except ProposalLedgerStoreError as exc:
        raise SystemExit(str(exc)) from exc


def _parse_reviewer(reviewer: str) -> tuple[str, str]:
    if reviewer.startswith("human:"):
        return "human", reviewer.split(":", 1)[1]
    if reviewer.startswith("cortex_policy:"):
        return "cortex_policy", reviewer.split(":", 1)[1]
    if reviewer.startswith("system:"):
        return "system", reviewer.split(":", 1)[1]
    if reviewer == "context-exec":
        raise SystemExit("context-exec cannot act as proposal reviewer")
    return "human", reviewer


def _compact_attention_reason(reason: str | None) -> str:
    if not reason:
        return ""
    compact = " ".join(reason.split())
    return compact if len(compact) <= 80 else compact[:77] + "..."


def _record_row(record: ProposalLedgerRecordV1) -> dict[str, Any]:
    return {
        "proposal_id": record.proposal_id,
        "proposal_type": record.envelope.proposal_type,
        "status": record.status,
        "risk": record.envelope.risk,
        "triage_action": record.triage_action,
        "attention_required": record.attention_required,
        "attention_reason": _compact_attention_reason(record.attention_reason),
        "title": record.envelope.title,
    }


def _inner_artifact_summary(envelope: ProposalEnvelopeV1) -> dict[str, Any]:
    artifact = envelope.artifact
    if envelope.artifact_type == "PatchProposalV1":
        patch = PatchProposalV1.model_validate(artifact)
        return {
            "artifact_type": envelope.artifact_type,
            "problem": patch.problem,
            "proposed_change_summary": patch.proposed_change_summary,
            "files_to_change": patch.files_to_change,
            "rollback_plan": patch.rollback_plan,
        }
    if envelope.artifact_type == "MemoryCorrectionProposalV1":
        correction = MemoryCorrectionProposalV1.model_validate(artifact)
        return {
            "artifact_type": envelope.artifact_type,
            "current_belief": correction.current_belief,
            "proposed_belief": correction.proposed_belief,
            "correction_type": correction.correction_type,
            "rationale": correction.rationale,
            "target_memory_domains": correction.target_memory_domains,
            "rollback_plan": correction.rollback_plan,
        }
    return {
        "artifact_type": envelope.artifact_type,
        "artifact_keys": sorted(artifact.keys()) if isinstance(artifact, dict) else [],
    }


def cmd_seed_demo(args: argparse.Namespace) -> int:
    store_path = _require_store(args.store)
    repo = _open_repo(store_path)

    patch = PatchProposalV1(
        problem="README typo in operator section",
        proposed_change_summary="Fix spelling in operator docs",
        rollback_plan="Revert README change",
        evidence=["typo spotted in review"],
        risk="low",
    )
    patch_envelope = build_patch_proposal_envelope(patch, source_mode="patch_proposal")
    stored_record = ProposalLedgerRecordV1(
        proposal_id=patch_envelope.proposal_id,
        envelope=patch_envelope,
        status="stored",
        source_mode="patch_proposal",
        created_at=_utc_now(),
    )
    repo.store(stored_record)

    correction = MemoryCorrectionProposalV1(
        current_belief="User is from Denver",
        proposed_belief="User location is unknown",
        correction_type="mark_uncertain",
        rationale="Insufficient evidence for Denver claim",
        rollback_plan="No mutation proposed; no rollback required.",
        supporting_evidence=["user mentioned Colorado once"],
        risk="medium",
    )
    memory_envelope = build_memory_correction_proposal_envelope(
        correction,
        source_mode="memory_correction_proposal",
    )
    memory_record = ProposalLedgerRecordV1(
        proposal_id=memory_envelope.proposal_id,
        envelope=memory_envelope,
        status="stored",
        source_mode="memory_correction_proposal",
        created_at=_utc_now(),
    )
    repo.store(memory_record)
    pending = repo.apply_triage(
        ProposalTriageDecisionV1(
            proposal_id=memory_record.proposal_id,
            action="promote_to_review",
            rationale="identity memory correction",
            attention_required=True,
            attention_reason="identity memory correction",
        )
    )

    blocked_correction = MemoryCorrectionProposalV1(
        current_belief="User prefers dark mode",
        rationale="Low confidence preference claim",
        rollback_plan="No mutation proposed; no rollback required.",
        confidence=0.1,
        risk="low",
    )
    blocked_envelope = build_memory_correction_proposal_envelope(
        blocked_correction,
        source_mode="memory_correction_proposal",
    )
    blocked_record = ProposalLedgerRecordV1(
        proposal_id=blocked_envelope.proposal_id,
        envelope=blocked_envelope,
        status="stored",
        source_mode="memory_correction_proposal",
        created_at=_utc_now(),
    )
    repo.store(blocked_record)
    blocked = repo.apply_triage(
        ProposalTriageDecisionV1(
            proposal_id=blocked_record.proposal_id,
            action="block_for_evidence",
            rationale="needs stronger grounding",
            attention_required=True,
            attention_reason="low confidence",
        )
    )

    payload = {
        "seeded": True,
        "store": str(store_path),
        "records": {
            "stored_patch": stored_record.proposal_id,
            "pending_review_memory": pending.proposal_id,
            "blocked_low_confidence": blocked.proposal_id,
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    store_path = _require_store(args.store)
    repo = _open_repo(store_path)
    records = repo.list_by_status(args.status) if args.status else repo.list_all()
    rows = [_record_row(record) for record in records]
    if args.json:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    else:
        for row in rows:
            print(
                "\t".join(
                    [
                        row["proposal_id"],
                        row["proposal_type"],
                        row["status"],
                        row["risk"],
                        row["triage_action"],
                        str(row["attention_required"]).lower(),
                        row["attention_reason"],
                        row["title"],
                    ]
                )
            )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    store_path = _require_store(args.store)
    repo = _open_repo(store_path)
    record = repo.get(args.proposal_id)
    if record is None:
        raise SystemExit(f"proposal not found: {args.proposal_id}")

    review_history = [
        item.model_dump(mode="json") for item in repo.review_history(args.proposal_id)
    ]
    latest_review = repo.latest_review_decision(args.proposal_id)
    eligibility = derive_execution_eligibility(record, latest_review)

    payload = {
        "proposal_id": record.proposal_id,
        "status": record.status,
        "triage_action": record.triage_action,
        "attention_required": record.attention_required,
        "attention_reason": record.attention_reason,
        "review_status": record.envelope.review_status,
        "envelope": record.envelope.model_dump(mode="json"),
        "inner_artifact_summary": _inner_artifact_summary(record.envelope),
        "evidence": record.envelope.evidence,
        "risk": record.envelope.risk,
        "review_history": review_history,
        "execution_eligibility": eligibility.model_dump(mode="json"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_triage(args: argparse.Namespace) -> int:
    store_path = _require_store(args.store)
    repo = _open_repo(store_path)
    decision = ProposalTriageDecisionV1(
        proposal_id=args.proposal_id,
        action=args.action,
        rationale=args.reason,
        reviewer_type="cortex_policy",
        created_at=_utc_now(),
    )
    try:
        updated = repo.apply_triage(decision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    payload = {
        "proposal_id": updated.proposal_id,
        "status": updated.status,
        "attention_required": updated.attention_required,
        "attention_reason": updated.attention_reason,
        "triage_action": updated.triage_action,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    store_path = _require_store(args.store)
    repo = _open_repo(store_path)
    reviewer_type, reviewer_id = _parse_reviewer(args.reviewer)
    if reviewer_type == "human" and reviewer_id == "context-exec":
        raise SystemExit("context-exec cannot act as proposal reviewer")

    decision = ProposalReviewDecisionV1(
        decision_id=f"dec_{uuid.uuid4().hex[:12]}",
        proposal_id=args.proposal_id,
        decision=args.decision,
        reviewer_type=reviewer_type,
        reviewer_id=reviewer_id,
        rationale=args.reason,
        created_at=_utc_now(),
    )
    try:
        updated = repo.apply_review(decision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    eligibility = derive_execution_eligibility(updated, decision)
    payload = {
        "proposal_id": updated.proposal_id,
        "status": updated.status,
        "review_decision": decision.model_dump(mode="json"),
        "execution_eligibility": eligibility.model_dump(mode="json"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_eligibility(args: argparse.Namespace) -> int:
    store_path = _require_store(args.store)
    repo = _open_repo(store_path)
    record = repo.get(args.proposal_id)
    if record is None:
        raise SystemExit(f"proposal not found: {args.proposal_id}")

    latest_review = repo.latest_review_decision(args.proposal_id)
    eligibility = derive_execution_eligibility(record, latest_review)
    print(json.dumps(eligibility.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orion proposal ledger operator CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed-demo", help="Seed demo proposal records (no mutation)")
    seed.add_argument("--store", required=True, help="JSON ledger file path")
    seed.set_defaults(func=cmd_seed_demo)

    list_cmd = sub.add_parser("list", help="List proposal ledger records")
    list_cmd.add_argument("--store", required=True, help="JSON ledger file path")
    list_cmd.add_argument("--status", default=None, help="Filter by proposal status")
    list_cmd.add_argument("--json", action="store_true", help="Emit JSON lines")
    list_cmd.set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="Show proposal details")
    show.add_argument("proposal_id")
    show.add_argument("--store", required=True, help="JSON ledger file path")
    show.set_defaults(func=cmd_show)

    triage = sub.add_parser("triage", help="Apply triage decision")
    triage.add_argument("proposal_id")
    triage.add_argument(
        "--action",
        required=True,
        choices=[
            "store_only",
            "promote_to_review",
            "block_for_evidence",
            "discard",
            "supersede",
            "expire",
        ],
    )
    triage.add_argument("--reason", required=True)
    triage.add_argument("--store", required=True, help="JSON ledger file path")
    triage.set_defaults(func=cmd_triage)

    review = sub.add_parser("review", help="Apply review decision")
    review.add_argument("proposal_id")
    review.add_argument("--decision", required=True, choices=["approve", "reject", "request_changes"])
    review.add_argument("--reason", required=True)
    review.add_argument("--reviewer", default="human:operator")
    review.add_argument("--store", required=True, help="JSON ledger file path")
    review.set_defaults(func=cmd_review)

    eligibility = sub.add_parser("eligibility", help="Show execution eligibility")
    eligibility.add_argument("proposal_id")
    eligibility.add_argument("--store", required=True, help="JSON ledger file path")
    eligibility.set_defaults(func=cmd_eligibility)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
