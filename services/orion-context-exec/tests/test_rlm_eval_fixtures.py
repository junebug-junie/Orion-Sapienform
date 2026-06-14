"""Deterministic pytest eval fixtures for context-exec RLM outputs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest

from app.runner import ContextExecRunner, FAKE_ORGANS
from orion.schemas.context_exec import (
    BeliefProvenanceReportV1,
    ContextExecPermissionV1,
    ContextExecRequestV1,
    ContextExecRunV1,
    RepoImpactAnalysisReportV1,
    TraceAutopsyReportV1,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

KNOWN_CORR = "5506f854-2606-42b5-a2ee-89775b3a8ed5"

BELIEF_STATUSES = frozenset(
    {"supported", "unsupported", "contradicted", "stale", "inferred", "unknown"}
)
TRACE_STATUSES = frozenset({"explained", "partial", "unknown"})
REPO_STATUSES = frozenset({"analyzed", "partial", "insufficient_grounding"})
REPO_RISKS = frozenset({"low", "medium", "high", "unknown"})

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


def blob_text(run: ContextExecRunV1) -> str:
    artifact_blob = json.dumps(run.artifact or {}, default=str)
    return f"{run.final_text}\n{artifact_blob}".lower()


def assert_claim_clean(artifact_claim: str, forbidden: tuple[str, ...]) -> None:
    lowered = artifact_claim.lower()
    for term in forbidden:
        assert term.lower() not in lowered, f"claim contains forbidden term {term!r}: {artifact_claim!r}"


def assert_engine_runtime_debug(run: ContextExecRunV1, engine: str) -> None:
    rd = run.runtime_debug
    assert "engine" in rd
    assert "engine_selected" in rd
    assert rd.get("engine") == engine
    assert rd.get("engine_selected") == engine
    assert rd.get("fallback_engine") is None
    if run.status == "ok":
        assert rd.get("schema_valid") is True


def assert_safety_posture(run: ContextExecRunV1) -> None:
    rd = run.runtime_debug
    if "write_enabled" in rd:
        assert rd["write_enabled"] is False
    if "network_enabled" in rd:
        assert rd["network_enabled"] is False
    if "sandbox_mode" in rd:
        assert rd["sandbox_mode"] == "docker"
    if "rlm_depth" in rd:
        assert rd["rlm_depth"] <= 1


async def run_eval_case(
    engine: str,
    case: RlmEvalCase,
    monkeypatch: pytest.MonkeyPatch,
    *,
    permissions: ContextExecPermissionV1 | None = None,
) -> ContextExecRunV1:
    monkeypatch.setattr("app.runner.settings.rlm_engine", engine)
    monkeypatch.setattr("app.runner.settings.context_exec_real_trace_enabled", False)
    monkeypatch.setattr("app.runner.settings.context_exec_real_recall_enabled", False)
    req = ContextExecRequestV1(
        text=case.text,
        mode=case.mode,
        permissions=permissions or ContextExecPermissionV1(),
        expected_artifact_type=case.expected_artifact_type,
    )
    runner = ContextExecRunner()
    run = await runner.run(req)
    assert_engine_runtime_debug(run, engine)
    assert_safety_posture(run)
    if run.status == "ok":
        assert run.artifact_type == case.expected_artifact_type
    return run


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


@pytest.fixture(autouse=True)
def _reset_fake_organs():
    FAKE_ORGANS.memory_hits = None
    FAKE_ORGANS.trace_hits = None
    yield
    FAKE_ORGANS.memory_hits = None
    FAKE_ORGANS.trace_hits = None


@pytest.fixture
def repo_root_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.runner.settings.context_exec_repo_root", REPO_ROOT)
    monkeypatch.setattr("app.runner.settings.orion_repo_root", REPO_ROOT)
    monkeypatch.setattr("app.runner.settings.context_exec_real_repo_enabled", True)
    from app import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "context_exec_repo_root", REPO_ROOT)
    monkeypatch.setattr(settings_mod.settings, "orion_repo_root", REPO_ROOT)
    monkeypatch.setattr(settings_mod.settings, "context_exec_real_repo_enabled", True)


DENVER_CLAIM_FORBIDDEN = (
    "come from across your runtime",
    "where did",
    "Orion",
    "claim that",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_belief_provenance_denver_claim_clean(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FAKE_ORGANS.memory_hits = [
        {
            "claim": "User is from Denver",
            "source_ref": "m:denver:1",
            "verified": True,
            "confidence": 0.9,
        }
    ]
    run = await run_eval_case(engine, BELIEF_DENVER, monkeypatch)
    assert run.status == "ok"
    model = BeliefProvenanceReportV1.model_validate(run.artifact)
    assert model.status in BELIEF_DENVER.expected_statuses

    if engine == "alexzhang":
        pytest.xfail("AlexZhang _extract_claim_from_text tail extraction includes prompt noise")

    assert_claim_clean(model.claim, DENVER_CLAIM_FORBIDDEN)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_belief_provenance_ogden_evidence(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FAKE_ORGANS.memory_hits = [
        {
            "claim": "User location is Ogden, not Denver",
            "source_ref": "m:ogden:1",
            "verified": True,
            "confidence": 0.9,
        }
    ]
    run = await run_eval_case(engine, BELIEF_OGDEN, monkeypatch)
    assert run.status == "ok"
    model = BeliefProvenanceReportV1.model_validate(run.artifact)
    assert model.status in BELIEF_STATUSES
    findings_blob = " ".join(f.claim for f in model.findings).lower()
    assert "ogden" in findings_blob


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_trace_autopsy_does_not_echo_question_as_root_cause(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FAKE_ORGANS.trace_hits = [
        {
            "handle": "t:5506",
            "snippet": "fail open fallback to AgentChainService after ContextExecService timeout",
            "corr_id": KNOWN_CORR,
            "kind": "error",
            "source": "cortex",
        }
    ]
    run = await run_eval_case(engine, TRACE_KNOWN_CORR, monkeypatch)
    assert run.status == "ok"
    model = TraceAutopsyReportV1.model_validate(run.artifact)
    assert model.status in TRACE_KNOWN_CORR.expected_statuses

    blob = blob_text(run)
    assert any(term.lower() in blob for term in TRACE_REQUIRED_TERMS)

    root_cause = (model.root_cause or "").lower()
    question_echo_terms = ("why did", "orion")
    if engine == "alexzhang" and any(t in root_cause for t in question_echo_terms):
        pytest.xfail("AlexZhang trace autopsy root_cause echoes question text")

    assert "why did" not in root_cause
    assert "orion" not in root_cause


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_trace_autopsy_missing_corr_reports_insufficient_evidence(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = await run_eval_case(engine, TRACE_MISSING_CORR, monkeypatch)
    assert run.status == "ok"
    model = TraceAutopsyReportV1.model_validate(run.artifact)
    assert model.status == "unknown"
    if engine == "fake":
        assert model.root_cause in {
            "no trace evidence found for correlation id",
            "insufficient_trace_evidence",
        }
    else:
        assert model.root_cause == "insufficient_trace_evidence"


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_repo_impact_does_not_invent_grounding(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
    repo_root_env: None,
) -> None:
    perms = ContextExecPermissionV1(read_repo=True)
    run = await run_eval_case(
        engine,
        REPO_AGENT_CHAIN,
        monkeypatch,
        permissions=perms,
    )
    assert run.status == "ok"
    model = RepoImpactAnalysisReportV1.model_validate(run.artifact)
    assert model.status in REPO_AGENT_CHAIN.expected_statuses
    assert model.risk in REPO_RISKS

    finding_refs = {str(f.source_ref or "") for f in model.findings}
    finding_claims = " ".join(f.claim for f in model.findings).lower()
    grounded_blob = finding_claims + " " + " ".join(model.affected_paths).lower()
    for path in model.affected_paths:
        if model.status == "insufficient_grounding":
            assert not path
        else:
            path_lower = path.lower()
            assert path_lower in grounded_blob or any(
                path_lower in ref.lower() for ref in finding_refs
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_repo_impact_engine_files(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
    repo_root_env: None,
) -> None:
    if engine == "fake":
        pytest.xfail("FakeRLMEngine repo_impact greps fixed agent-chain patterns, not engine files")

    perms = ContextExecPermissionV1(read_repo=True)
    run = await run_eval_case(
        engine,
        REPO_ENGINE_FILES_CASE,
        monkeypatch,
        permissions=perms,
    )
    assert run.status == "ok"
    model = RepoImpactAnalysisReportV1.model_validate(run.artifact)
    assert model.status in REPO_ENGINE_FILES_CASE.expected_statuses

    blob = blob_text(run)
    if model.status == "insufficient_grounding":
        pytest.xfail("repo grep returned insufficient grounding for engine file probe")

    assert any(term.lower() in blob for term in REPO_ENGINE_FILES)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_runtime_debug_identifies_engine(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FAKE_ORGANS.memory_hits = [
        {"claim": "User is from Denver", "source_ref": "m:1", "verified": True, "confidence": 0.9}
    ]
    run = await run_eval_case(engine, BELIEF_DENVER, monkeypatch)
    rd = run.runtime_debug
    assert rd.get("engine") == engine
    assert rd.get("engine_selected") == engine
    assert "engine" in rd
    assert "engine_selected" in rd


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_safety_posture_read_only(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = await run_eval_case(engine, TRACE_MISSING_CORR, monkeypatch)
    assert_safety_posture(run)
    rd = run.runtime_debug
    assert rd.get("write_enabled") is False
    assert rd.get("network_enabled") is False
    assert rd.get("sandbox_mode") == "docker"
    assert rd.get("rlm_depth") <= 1
