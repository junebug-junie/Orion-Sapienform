"""Memory correction proposal mode scaffold tests (read-only, no mutation)."""

from __future__ import annotations

import json

import pytest

from orion.schemas.context_exec import (
    ContextExecPermissionV1,
    ContextExecRequestV1,
    MemoryCorrectionProposalV1,
    PatchProposalV1,
    ProposalEnvelopeV1,
    build_memory_correction_proposal_envelope,
)
from orion.schemas.registry import _REGISTRY

from app.alexzhang_rlm_engine import AlexZhangRLMEngine
from app.artifact_builder import validate_artifact
from app.callable_namespace import ContextNamespace
from app.rlm_engine import FakeRLMEngine
from app.runner import ContextExecRunner

MEMORY_CORRECTION_PROMPT = (
    "Propose a memory correction for the unsupported claim that I am from Denver."
)

EXECUTION_FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "git_commit",
        "git_push",
        "apply_patch",
        "write_enabled",
        "shell_command",
        "pr_create",
        "applied",
        "memory_written",
        "graph_written",
    }
)

MUTATION_FORBIDDEN_REVIEW_STATUSES = frozenset({"approved", "executed"})


def _assert_memory_correction_envelope_contract(envelope: ProposalEnvelopeV1) -> None:
    assert envelope.proposal_type == "memory_correction_proposal"
    assert envelope.artifact_type == "MemoryCorrectionProposalV1"
    assert envelope.mutation_allowed is False
    assert envelope.requires_human_approval is True
    assert envelope.review_status in {"draft", "pending_review"}
    inner = MemoryCorrectionProposalV1.model_validate(envelope.artifact)
    assert inner.mutation_allowed is False


def test_memory_correction_proposal_schema_defaults_to_no_mutation() -> None:
    model = MemoryCorrectionProposalV1(
        current_belief="User is from Denver",
        rationale="Insufficient evidence for Denver claim",
        rollback_plan="No mutation proposed; no rollback required.",
    )
    assert model.mutation_allowed is False
    assert model.correction_type == "mark_uncertain"
    assert model.confidence == 0.0
    assert model.supporting_evidence == []
    assert model.contradicting_evidence == []
    assert model.missing_evidence == []
    assert model.target_memory_domains == []
    assert model.affected_ids == []
    assert model.tests_to_run == []
    assert model.open_questions == []
    assert model.risk == "unknown"


def test_memory_correction_proposal_registered_with_context_exec_schemas() -> None:
    assert _REGISTRY["MemoryCorrectionProposalV1"] is MemoryCorrectionProposalV1
    assert _REGISTRY["ProposalEnvelopeV1"] is ProposalEnvelopeV1
    assert _REGISTRY["PatchProposalV1"] is PatchProposalV1


@pytest.mark.asyncio
async def test_fake_engine_memory_correction_proposal_returns_safe_envelope() -> None:
    engine = FakeRLMEngine()
    ns = ContextNamespace(permissions=ContextExecPermissionV1())
    req = ContextExecRequestV1(text=MEMORY_CORRECTION_PROMPT, mode="memory_correction_proposal")
    raw = await engine.run(req, ns)
    artifact, artifact_type, valid = validate_artifact("memory_correction_proposal", raw)
    assert valid is True
    assert artifact_type == "ProposalEnvelopeV1"
    envelope = ProposalEnvelopeV1.model_validate(artifact)
    _assert_memory_correction_envelope_contract(envelope)
    inner = MemoryCorrectionProposalV1.model_validate(envelope.artifact)
    assert inner.correction_type == "mark_uncertain"
    assert inner.risk == "unknown"
    assert inner.confidence == 0.0
    assert "fake engine has no grounded memory evidence" in inner.missing_evidence


@pytest.mark.asyncio
async def test_alexzhang_memory_correction_proposal_returns_read_only_envelope() -> None:
    memory_hits = [
        {
            "claim": "User is from Denver",
            "source_ref": "m:denver:1",
            "verified": True,
            "confidence": 0.9,
        }
    ]
    engine = AlexZhangRLMEngine()
    ns = ContextNamespace(
        permissions=ContextExecPermissionV1(),
        memory_fn={
            "search_claims": lambda q, limit=20: memory_hits[:limit],
            "read": lambda h: {},
        },
    )
    req = ContextExecRequestV1(text=MEMORY_CORRECTION_PROMPT, mode="memory_correction_proposal")
    raw = await engine.run(req, ns)
    artifact, artifact_type, valid = validate_artifact("memory_correction_proposal", raw)
    assert valid is True
    assert artifact_type == "ProposalEnvelopeV1"
    envelope = ProposalEnvelopeV1.model_validate(artifact)
    _assert_memory_correction_envelope_contract(envelope)
    inner = MemoryCorrectionProposalV1.model_validate(envelope.artifact)
    assert "denver" in inner.current_belief.lower()
    assert inner.correction_type in {"mark_uncertain", "mark_contradicted", "replace_belief"}


@pytest.mark.asyncio
async def test_memory_correction_proposal_does_not_mutate_memory(monkeypatch) -> None:
    monkeypatch.setattr("app.runner.settings.rlm_engine", "fake")
    runner = ContextExecRunner()
    req = ContextExecRequestV1(
        text=MEMORY_CORRECTION_PROMPT,
        mode="memory_correction_proposal",
        permissions=ContextExecPermissionV1(
            write_memory=False,
            write_graph=False,
            write_repo=False,
            mutate_runtime=False,
            network_enabled=False,
            shell_enabled=False,
        ),
    )
    run = await runner.run(req)
    assert run.status == "ok"
    assert run.artifact_type == "ProposalEnvelopeV1"
    envelope = ProposalEnvelopeV1.model_validate(run.artifact)
    _assert_memory_correction_envelope_contract(envelope)
    rd = run.runtime_debug
    assert rd.get("schema_valid") is True
    assert rd.get("proposal_enveloped") is True
    assert rd.get("proposal_type") == "memory_correction_proposal"
    assert rd.get("write_enabled") is False
    assert rd.get("network_enabled") is False
    assert rd.get("mutation_allowed") is False
    assert rd.get("requires_human_approval") is True
    assert rd.get("rlm_depth", 99) <= 1

    for key in EXECUTION_FORBIDDEN_ARTIFACT_KEYS:
        assert key not in run.artifact
    artifact_blob = json.dumps(run.artifact, default=str).lower()
    assert '"write_enabled": true' not in artifact_blob
    assert '"mutation_allowed": true' not in artifact_blob
    assert '"applied": true' not in artifact_blob
    assert '"memory_written": true' not in artifact_blob
    assert '"graph_written": true' not in artifact_blob
    assert envelope.review_status not in MUTATION_FORBIDDEN_REVIEW_STATUSES


@pytest.mark.asyncio
async def test_memory_correction_denver_proposal_mentions_current_belief(monkeypatch) -> None:
    monkeypatch.setattr("app.runner.settings.rlm_engine", "fake")
    runner = ContextExecRunner()
    req = ContextExecRequestV1(text=MEMORY_CORRECTION_PROMPT, mode="memory_correction_proposal")
    run = await runner.run(req)
    envelope = ProposalEnvelopeV1.model_validate(run.artifact)
    inner = MemoryCorrectionProposalV1.model_validate(envelope.artifact)
    assert "denver" in inner.current_belief.lower()
    final_blob = f"{run.final_text} {json.dumps(run.artifact, default=str)}".lower()
    assert "applied" not in final_blob or "not allowed" in run.final_text.lower()
    assert "mutation is not allowed" in run.final_text.lower()


def test_build_memory_correction_proposal_envelope_rejects_self_approval() -> None:
    correction = MemoryCorrectionProposalV1(
        current_belief="User is from Denver",
        rationale="Unsupported Denver claim",
        rollback_plan="No mutation proposed; no rollback required.",
    )
    envelope = build_memory_correction_proposal_envelope(
        correction, source_mode="memory_correction_proposal"
    )
    assert envelope.review_status == "draft"
    assert envelope.proposal_type == "memory_correction_proposal"
