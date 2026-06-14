#!/usr/bin/env python3
"""Compact CLI runner for context-exec RLM eval fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _script_dir:
    sys.path.pop(0)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SERVICE_DIR = os.path.join(REPO_ROOT, "services", "orion-context-exec")
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

from orion.schemas.context_exec import ContextExecPermissionV1  # noqa: E402

from app.rlm_eval_harness import (  # noqa: E402
    ALL_EVAL_CASES,
    DENVER_CLAIM_FORBIDDEN,
    REPO_ROOT,
    assert_claim_clean,
    run_eval_case,
    seed_case_organs,
)


def _inner_proposal_artifact(artifact: dict) -> dict:
    proposal_type = artifact.get("proposal_type")
    if proposal_type in {"patch_proposal", "memory_correction_proposal"}:
        inner = artifact.get("artifact") or {}
        return inner if isinstance(inner, dict) else artifact
    if artifact.get("artifact_type") in {"PatchProposalV1", "MemoryCorrectionProposalV1"} and "proposal_id" not in artifact:
        return artifact
    return artifact


def _quality_pass(case_name: str, engine: str, run) -> bool:
    if run.status != "ok":
        return False
    if case_name == "belief_provenance_denver" and engine == "alexzhang":
        claim = str((run.artifact or {}).get("claim") or "")
        try:
            assert_claim_clean(claim, DENVER_CLAIM_FORBIDDEN)
        except AssertionError:
            return False
    if case_name == "trace_autopsy_known_corr" and engine == "alexzhang":
        root_cause = str((run.artifact or {}).get("root_cause") or "").lower()
        if any(term in root_cause for term in ("why did", "orion")):
            return False
    if case_name == "patch_proposal_trace_autopsy_quality":
        artifact = run.artifact or {}
        if run.artifact_type != "ProposalEnvelopeV1":
            return False
        if artifact.get("proposal_type") != "patch_proposal":
            return False
        if artifact.get("artifact_type") != "PatchProposalV1":
            return False
        if artifact.get("mutation_allowed") is not False:
            return False
        if artifact.get("requires_human_approval") is not True:
            return False
        if artifact.get("review_status") not in {"draft", "pending_review"}:
            return False
        inner = _inner_proposal_artifact(artifact)
        risk = str(inner.get("risk") or artifact.get("risk") or "unknown")
        if risk not in {"low", "medium", "unknown"}:
            return False
        if engine == "alexzhang" and inner.get("files_to_change"):
            blob = " ".join(str(p) for p in inner.get("files_to_change") or []).lower()
            if not any(name.lower() in blob for name in (
                "alexzhang_rlm_engine.py",
                "test_alexzhang_rlm_engine.py",
                "test_rlm_eval_fixtures.py",
                "rlm_eval_harness.py",
            )):
                return False
        if engine == "alexzhang" and not inner.get("files_to_change"):
            open_q = " ".join(str(q) for q in inner.get("open_questions") or []).lower()
            if "insufficient repo grounding" not in open_q:
                return False
    if case_name == "memory_correction_denver_uncertain":
        artifact = run.artifact or {}
        if run.artifact_type != "ProposalEnvelopeV1":
            return False
        if artifact.get("proposal_type") != "memory_correction_proposal":
            return False
        if artifact.get("artifact_type") != "MemoryCorrectionProposalV1":
            return False
        if artifact.get("mutation_allowed") is not False:
            return False
        if artifact.get("requires_human_approval") is not True:
            return False
        if artifact.get("review_status") not in {"draft", "pending_review"}:
            return False
        inner = _inner_proposal_artifact(artifact)
        current_belief = str(inner.get("current_belief") or "").lower()
        if "denver" not in current_belief:
            return False
        correction_type = str(inner.get("correction_type") or "")
        if correction_type not in {"mark_uncertain", "mark_contradicted", "replace_belief"}:
            return False
        confidence = float(inner.get("confidence") or 0.0)
        if confidence > 0.7:
            ogden_evidence = any(
                "ogden" in str(e).lower()
                for e in (inner.get("contradicting_evidence") or [])
            )
            if not ogden_evidence:
                return False
        blob = json.dumps(inner, default=str).lower()
        if any(term in blob for term in ("applied=true", "memory_written=true", "graph_written=true")):
            return False
    return True


async def _run(engine: str) -> list[tuple]:
    rows: list[tuple] = []
    repo_settings = {
        "context_exec_repo_root": REPO_ROOT,
        "orion_repo_root": REPO_ROOT,
        "context_exec_real_repo_enabled": True,
    }
    for case in ALL_EVAL_CASES:
        seed_case_organs(case.name)
        notes = ""
        perms = (
            ContextExecPermissionV1(read_repo=True)
            if case.mode in {"repo_impact_analysis", "patch_proposal"}
            else None
        )
        overrides = repo_settings if case.mode in {"repo_impact_analysis", "patch_proposal"} else None
        try:
            run = await run_eval_case(
                engine,
                case,
                permissions=perms,
                settings_overrides=overrides,
            )
            artifact = run.artifact or {}
            inner = _inner_proposal_artifact(artifact)
            artifact_status = str(inner.get("status", artifact.get("status", run.status)))
            quality = _quality_pass(case.name, engine, run)
            proposal_type = str(artifact.get("proposal_type") or "")
            inner_artifact_type = str(artifact.get("artifact_type") or "")
            review_status = str(artifact.get("review_status") or "")
            mutation_allowed = artifact.get("mutation_allowed")
            requires_human_approval = artifact.get("requires_human_approval")
        except Exception as exc:
            artifact_status = "error"
            quality = False
            notes = str(exc)
            run = None
            proposal_type = ""
            inner_artifact_type = ""
            review_status = ""
            mutation_allowed = ""
            requires_human_approval = ""
        if run is not None:
            if case.mode in {"patch_proposal", "memory_correction_proposal"}:
                rows.append(
                    (
                        case.name,
                        engine,
                        case.mode,
                        quality,
                        str(run.artifact_type or ""),
                        proposal_type,
                        inner_artifact_type,
                        review_status,
                        mutation_allowed,
                        requires_human_approval,
                        notes,
                    )
                )
            else:
                rows.append(
                    (
                        case.name,
                        engine,
                        case.mode,
                        artifact_status,
                        str(run.artifact_type or ""),
                        quality,
                        notes,
                    )
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run context-exec RLM eval fixtures")
    parser.add_argument("--engine", choices=["fake", "alexzhang"], default="fake")
    args = parser.parse_args()

    rows = asyncio.run(_run(args.engine))
    has_proposal = any(c.mode in {"patch_proposal", "memory_correction_proposal"} for c in ALL_EVAL_CASES)
    if has_proposal:
        header = (
            "case",
            "engine",
            "mode",
            "quality_pass",
            "artifact_type",
            "proposal_type",
            "inner_artifact_type",
            "review_status",
            "mutation_allowed",
            "requires_human_approval",
            "notes",
        )
    else:
        header = ("case", "engine", "mode", "status", "artifact_type", "quality_pass", "notes")
    print(" | ".join(header))
    for row in rows:
        print(" | ".join(str(c) for c in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
