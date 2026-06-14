"""Patch proposal mode scaffold tests (read-only blueprint, no mutation)."""

from __future__ import annotations

import json

import pytest

from orion.schemas.context_exec import (
    ContextExecPermissionV1,
    ContextExecRequestV1,
    PatchProposalV1,
    ProposalEnvelopeV1,
    build_patch_proposal_envelope,
)

from app.alexzhang_rlm_engine import AlexZhangRLMEngine
from app.artifact_builder import validate_artifact
from app.callable_namespace import ContextNamespace
from app.repo_tools import RepoHit
from app.rlm_engine import FakeRLMEngine
from app.runner import ContextExecRunner

PATCH_PROMPT = (
    "Propose a patch for weak trace-autopsy root cause synthesis in context-exec."
)

EXECUTION_FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "git_commit",
        "git_push",
        "apply_patch",
        "write_enabled",
        "shell_command",
        "pr_create",
    }
)


def _assert_proposal_envelope_contract(envelope: ProposalEnvelopeV1) -> None:
    assert envelope.proposal_type == "patch_proposal"
    assert envelope.artifact_type == "PatchProposalV1"
    assert envelope.mutation_allowed is False
    assert envelope.requires_human_approval is True
    assert envelope.review_status in {"draft", "pending_review"}
    inner = PatchProposalV1.model_validate(envelope.artifact)
    assert inner.mutation_allowed is False


def test_patch_proposal_schema_defaults_to_no_mutation() -> None:
    model = PatchProposalV1(
        problem="weak synthesis",
        proposed_change_summary="Improve root cause heuristics",
        rollback_plan="Revert file edits",
    )
    assert model.mutation_allowed is False
    assert model.risk == "unknown"
    assert model.evidence == []
    assert model.files_to_change == []
    assert model.tests_to_run == []
    assert model.open_questions == []


@pytest.mark.asyncio
async def test_fake_engine_patch_proposal_returns_proposal_envelope() -> None:
    engine = FakeRLMEngine()
    ns = ContextNamespace(permissions=ContextExecPermissionV1())
    req = ContextExecRequestV1(text=PATCH_PROMPT, mode="patch_proposal")
    raw = await engine.run(req, ns)
    artifact, artifact_type, valid = validate_artifact("patch_proposal", raw)
    assert valid is True
    assert artifact_type == "ProposalEnvelopeV1"
    envelope = ProposalEnvelopeV1.model_validate(artifact)
    _assert_proposal_envelope_contract(envelope)
    inner = PatchProposalV1.model_validate(envelope.artifact)
    assert inner.risk == "unknown"
    assert inner.files_to_change == []
    assert inner.rollback_plan == "No changes proposed; no rollback required."


@pytest.mark.asyncio
async def test_alexzhang_patch_proposal_returns_proposal_envelope(monkeypatch) -> None:
    fake_hits = [
        {
            "path": "services/orion-context-exec/app/alexzhang_rlm_engine.py",
            "line_start": 162,
            "snippet": "def _synthesize_trace_root_cause",
            "source_ref": "repo:alexzhang:1",
        },
        {
            "path": "services/orion-context-exec/tests/test_alexzhang_rlm_engine.py",
            "line_start": 109,
            "snippet": "mode=\"trace_autopsy\"",
            "source_ref": "repo:test:1",
        },
    ]

    def _stub_grep(pattern: str, path: str | None = None, limit: int = 50) -> list[RepoHit]:
        return [RepoHit.model_validate(h) for h in fake_hits]

    monkeypatch.setattr("app.repo_tools.repo_grep", _stub_grep)

    engine = AlexZhangRLMEngine()
    ns = ContextNamespace(permissions=ContextExecPermissionV1(read_repo=True))
    req = ContextExecRequestV1(
        text=PATCH_PROMPT,
        mode="patch_proposal",
        permissions=ContextExecPermissionV1(read_repo=True),
    )
    raw = await engine.run(req, ns)
    artifact, artifact_type, valid = validate_artifact("patch_proposal", raw)
    assert valid is True
    assert artifact_type == "ProposalEnvelopeV1"
    envelope = ProposalEnvelopeV1.model_validate(artifact)
    _assert_proposal_envelope_contract(envelope)
    inner = PatchProposalV1.model_validate(envelope.artifact)
    grounded_refs = {str(h.get("path") or "") for h in fake_hits}
    for path in inner.files_to_change:
        assert path in grounded_refs
    assert inner.tests_to_run or any(
        "tests_to_run" in q.lower() or "ground" in q.lower() for q in inner.open_questions
    )


@pytest.mark.asyncio
async def test_patch_proposal_does_not_execute_or_request_mutation(monkeypatch) -> None:
    monkeypatch.setattr("app.runner.settings.rlm_engine", "fake")
    runner = ContextExecRunner()
    req = ContextExecRequestV1(
        text=PATCH_PROMPT,
        mode="patch_proposal",
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
    _assert_proposal_envelope_contract(envelope)
    rd = run.runtime_debug
    assert rd.get("schema_valid") is True
    assert rd.get("proposal_enveloped") is True
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

    for step in run.verb_trace:
        summary = f"{step.verb} {step.callable or ''} {step.output_summary or ''}".lower()
        assert "git commit" not in summary
        assert "git push" not in summary
        assert "gh pr create" not in summary
        assert "apply patch now" not in summary


@pytest.mark.asyncio
async def test_runner_patch_proposal_final_text_boundary(monkeypatch) -> None:
    monkeypatch.setattr("app.runner.settings.rlm_engine", "fake")
    runner = ContextExecRunner()
    req = ContextExecRequestV1(text=PATCH_PROMPT, mode="patch_proposal")
    run = await runner.run(req)
    assert "Mutation is not allowed by context-exec" in run.final_text
    envelope = ProposalEnvelopeV1.model_validate(run.artifact)
    assert envelope.mutation_allowed is False


def test_build_patch_proposal_envelope_rejects_self_approval() -> None:
    patch = PatchProposalV1(
        problem="p",
        proposed_change_summary="s",
        rollback_plan="r",
    )
    envelope = build_patch_proposal_envelope(patch, source_mode="patch_proposal")
    assert envelope.review_status == "draft"
