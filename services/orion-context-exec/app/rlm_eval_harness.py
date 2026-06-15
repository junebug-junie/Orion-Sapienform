"""Shared RLM eval case definitions and runner helpers (pytest-free)."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from orion.schemas.context_exec import (
    ContextExecPermissionV1,
    ContextExecRequestV1,
    ContextExecRunV1,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

KNOWN_CORR = "5506f854-2606-42b5-a2ee-89775b3a8ed5"

BELIEF_STATUSES = frozenset(
    {"supported", "unsupported", "contradicted", "stale", "inferred", "unknown"}
)
TRACE_STATUSES = frozenset({"explained", "partial", "unknown"})
REPO_STATUSES = frozenset({"analyzed", "partial", "insufficient_grounding"})
REPO_RISKS = frozenset({"low", "medium", "high", "unknown"})
PATCH_PROPOSAL_RISKS = frozenset({"low", "medium", "high", "unknown"})

PATCH_PROPOSAL_LIKELY_FILES = (
    "alexzhang_rlm_engine.py",
    "test_alexzhang_rlm_engine.py",
    "test_rlm_eval_fixtures.py",
    "rlm_eval_harness.py",
)

TRACE_REQUIRED_TERMS = (
    "fail open",
    "fallback",
    "ContextExecService",
    "AgentChainService",
    "timeout",
)

REPO_ENGINE_FILES = (
    "rlm_engine.py",
    "alexzhang_rlm_engine.py",
    "runner.py",
    "settings.py",
)

ENGINES = ("fake", "alexzhang")

DENVER_CLAIM_FORBIDDEN = (
    "come from across your runtime",
    "where did",
    "Orion",
    "claim that",
)


@dataclass(frozen=True)
class RlmEvalCase:
    name: str
    mode: str
    text: str
    expected_artifact_type: str
    expected_statuses: set[str]
    required_final_text_terms: tuple[str, ...] = ()
    forbidden_final_text_terms: tuple[str, ...] = ()
    required_artifact_terms: tuple[str, ...] = ()
    forbidden_artifact_terms: tuple[str, ...] = ()


BELIEF_DENVER = RlmEvalCase(
    name="belief_provenance_denver",
    mode="belief_provenance",
    text="Orion, where did the claim that I am from Denver come from across your runtime?",
    expected_artifact_type="BeliefProvenanceReportV1",
    expected_statuses=set(BELIEF_STATUSES),
)

BELIEF_OGDEN = RlmEvalCase(
    name="belief_provenance_ogden",
    mode="belief_provenance",
    text="What evidence says my location is not Denver?",
    expected_artifact_type="BeliefProvenanceReportV1",
    expected_statuses={"unknown", "contradicted", "unsupported"},
)

TRACE_KNOWN_CORR = RlmEvalCase(
    name="trace_autopsy_known_corr",
    mode="trace_autopsy",
    text=f"Orion, why did corr {KNOWN_CORR} fail open?",
    expected_artifact_type="TraceAutopsyReportV1",
    expected_statuses=set(TRACE_STATUSES),
)

TRACE_MISSING_CORR = RlmEvalCase(
    name="trace_autopsy_missing_corr",
    mode="trace_autopsy",
    text="Why did corr does-not-exist fail open?",
    expected_artifact_type="TraceAutopsyReportV1",
    expected_statuses={"unknown"},
)

REPO_AGENT_CHAIN = RlmEvalCase(
    name="repo_impact_agent_chain",
    mode="repo_impact_analysis",
    text="What breaks if I replace agent-chain-service with context-exec? Ground this in my repo.",
    expected_artifact_type="RepoImpactAnalysisReportV1",
    expected_statuses=set(REPO_STATUSES),
)

REPO_ENGINE_FILES_CASE = RlmEvalCase(
    name="repo_impact_engine_files",
    mode="repo_impact_analysis",
    text="What files are involved in context-exec RLM engine selection?",
    expected_artifact_type="RepoImpactAnalysisReportV1",
    expected_statuses=set(REPO_STATUSES),
)

PATCH_PROPOSAL_TRACE_AUTOPSY = RlmEvalCase(
    name="patch_proposal_trace_autopsy_quality",
    mode="patch_proposal",
    text="Propose a patch for weak trace-autopsy root cause synthesis in context-exec.",
    expected_artifact_type="ProposalEnvelopeV1",
    expected_statuses=set(PATCH_PROPOSAL_RISKS),
)

MEMORY_CORRECTION_DENVER = RlmEvalCase(
    name="memory_correction_denver_uncertain",
    mode="memory_correction_proposal",
    text="Propose a memory correction for the unsupported claim that I am from Denver.",
    expected_artifact_type="ProposalEnvelopeV1",
    expected_statuses=set(PATCH_PROPOSAL_RISKS),
)

ALL_EVAL_CASES = (
    BELIEF_DENVER,
    BELIEF_OGDEN,
    TRACE_KNOWN_CORR,
    TRACE_MISSING_CORR,
    REPO_AGENT_CHAIN,
    REPO_ENGINE_FILES_CASE,
    PATCH_PROPOSAL_TRACE_AUTOPSY,
    MEMORY_CORRECTION_DENVER,
)


def blob_text(run: ContextExecRunV1) -> str:
    artifact_blob = json.dumps(run.artifact or {}, default=str)
    return f"{run.final_text}\n{artifact_blob}".lower()


def assert_claim_clean(artifact_claim: str, forbidden: tuple[str, ...]) -> None:
    lowered = artifact_claim.lower()
    for term in forbidden:
        assert term.lower() not in lowered, f"claim contains forbidden term {term!r}: {artifact_claim!r}"


def assert_engine_runtime_debug(run: ContextExecRunV1, engine: str) -> None:
    rd = run.runtime_debug
    assert rd.get("engine_requested") == engine
    assert "engine_selected" in rd
    assert "engine" in rd
    selected = rd.get("engine_selected")
    assert rd.get("engine") == selected
    if selected == "fake" and engine == "alexzhang":
        assert rd.get("fallback_used") is True
        assert rd.get("fallback_engine") == "fake"
        assert rd.get("fallback_reason")
    else:
        assert selected == engine
        assert rd.get("fallback_used") is not True
        assert rd.get("fallback_engine") is None
    if run.status == "ok":
        assert rd.get("schema_valid") is True


def assert_safety_posture(run: ContextExecRunV1) -> None:
    rd = run.runtime_debug
    if "write_enabled" in rd:
        assert rd["write_enabled"] is False
    if "network_enabled" in rd:
        assert rd["network_enabled"] is False
    if "mutation_allowed" in rd:
        assert rd["mutation_allowed"] is False
    if run.artifact_type == "ProposalEnvelopeV1":
        assert run.artifact.get("mutation_allowed") is False
        assert run.artifact.get("requires_human_approval") is True
        assert run.artifact.get("review_status") in {"draft", "pending_review"}
        inner = run.artifact.get("artifact") or {}
        if run.artifact.get("artifact_type") == "PatchProposalV1":
            assert inner.get("mutation_allowed") is False
        if run.artifact.get("artifact_type") == "MemoryCorrectionProposalV1":
            assert inner.get("mutation_allowed") is False
    if run.artifact_type == "PatchProposalV1":
        assert run.artifact.get("mutation_allowed") is False
    if "sandbox_mode" in rd:
        assert rd["sandbox_mode"] == "docker"
    if "rlm_depth" in rd:
        assert rd["rlm_depth"] <= 1


@contextmanager
def patched_settings(**overrides: Any) -> Iterator[None]:
    from app import settings as settings_mod

    settings = settings_mod.settings
    originals: dict[str, Any] = {}
    for key, value in overrides.items():
        originals[key] = getattr(settings, key)
        setattr(settings, key, value)
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(settings, key, value)


async def run_eval_case(
    engine: str,
    case: RlmEvalCase,
    *,
    permissions: ContextExecPermissionV1 | None = None,
    settings_overrides: dict[str, Any] | None = None,
) -> ContextExecRunV1:
    base_overrides: dict[str, Any] = {
        "rlm_engine": engine,
        "context_exec_real_trace_enabled": False,
        "context_exec_real_recall_enabled": False,
    }
    if settings_overrides:
        base_overrides.update(settings_overrides)

    req = ContextExecRequestV1(
        text=case.text,
        mode=case.mode,
        permissions=permissions or ContextExecPermissionV1(),
        expected_artifact_type=case.expected_artifact_type,
    )
    with patched_settings(**base_overrides):
        from app.settings import settings as live_settings
        import app.runner as runner_mod
        from app.runner import ContextExecRunner

        runner_mod.settings = live_settings
        runner = ContextExecRunner()
        run = await runner.run(req)
    assert_engine_runtime_debug(run, engine)
    assert_safety_posture(run)
    if run.status == "ok":
        assert run.artifact_type == case.expected_artifact_type
    return run


def reset_fake_organs() -> None:
    from app.runner import FAKE_ORGANS

    FAKE_ORGANS.memory_hits = None
    FAKE_ORGANS.trace_hits = None


def seed_case_organs(case_name: str) -> None:
    from app.runner import FAKE_ORGANS

    reset_fake_organs()
    if case_name == "belief_provenance_denver":
        FAKE_ORGANS.memory_hits = [
            {
                "claim": "User is from Denver",
                "source_ref": "m:denver:1",
                "verified": True,
                "confidence": 0.9,
            }
        ]
    elif case_name == "belief_provenance_ogden":
        FAKE_ORGANS.memory_hits = [
            {
                "claim": "User location is Ogden, not Denver",
                "source_ref": "m:ogden:1",
                "verified": True,
                "confidence": 0.9,
            }
        ]
    elif case_name == "memory_correction_denver_uncertain":
        FAKE_ORGANS.memory_hits = [
            {
                "claim": "User is from Denver",
                "source_ref": "m:denver:1",
                "verified": True,
                "confidence": 0.9,
            }
        ]
    elif case_name == "trace_autopsy_known_corr":
        FAKE_ORGANS.trace_hits = [
            {
                "handle": "t:5506",
                "snippet": "fail open fallback to AgentChainService after ContextExecService timeout",
                "corr_id": KNOWN_CORR,
                "kind": "error",
                "source": "cortex",
            }
        ]
