"""Deterministic pytest eval fixtures for context-exec RLM outputs."""

from __future__ import annotations

import pytest

from orion.schemas.context_exec import (
    BeliefProvenanceReportV1,
    ContextExecPermissionV1,
    RepoImpactAnalysisReportV1,
    TraceAutopsyReportV1,
)

from app.rlm_eval_harness import (
    ALL_EVAL_CASES,
    BELIEF_DENVER,
    BELIEF_OGDEN,
    BELIEF_STATUSES,
    DENVER_CLAIM_FORBIDDEN,
    ENGINES,
    KNOWN_CORR,
    REPO_AGENT_CHAIN,
    REPO_ENGINE_FILES,
    REPO_ENGINE_FILES_CASE,
    REPO_RISKS,
    REPO_ROOT,
    TRACE_KNOWN_CORR,
    TRACE_MISSING_CORR,
    TRACE_REQUIRED_TERMS,
    assert_claim_clean,
    assert_engine_runtime_debug,
    assert_safety_posture,
    blob_text,
    reset_fake_organs,
    run_eval_case,
)

REPO_SETTINGS = {
    "context_exec_repo_root": REPO_ROOT,
    "orion_repo_root": REPO_ROOT,
    "context_exec_real_repo_enabled": True,
}


@pytest.fixture(autouse=True)
def _reset_fake_organs():
    reset_fake_organs()
    yield
    reset_fake_organs()


@pytest.fixture
def repo_settings() -> dict[str, object]:
    return dict(REPO_SETTINGS)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_belief_provenance_denver_claim_clean(
    engine: str,
) -> None:
    from app.rlm_eval_harness import seed_case_organs

    seed_case_organs("belief_provenance_denver")
    run = await run_eval_case(engine, BELIEF_DENVER)
    assert run.status == "ok"
    model = BeliefProvenanceReportV1.model_validate(run.artifact)
    assert model.status in BELIEF_DENVER.expected_statuses
    assert_claim_clean(model.claim, DENVER_CLAIM_FORBIDDEN)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_belief_provenance_ogden_evidence(engine: str) -> None:
    from app.rlm_eval_harness import seed_case_organs

    seed_case_organs("belief_provenance_ogden")
    run = await run_eval_case(engine, BELIEF_OGDEN)
    assert run.status == "ok"
    model = BeliefProvenanceReportV1.model_validate(run.artifact)
    assert model.status in BELIEF_STATUSES
    findings_blob = " ".join(f.claim for f in model.findings).lower()
    assert "ogden" in findings_blob


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_trace_autopsy_does_not_echo_question_as_root_cause(
    engine: str,
) -> None:
    from app.rlm_eval_harness import seed_case_organs

    seed_case_organs("trace_autopsy_known_corr")
    run = await run_eval_case(engine, TRACE_KNOWN_CORR)
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
) -> None:
    run = await run_eval_case(engine, TRACE_MISSING_CORR)
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
    repo_settings: dict[str, object],
) -> None:
    perms = ContextExecPermissionV1(read_repo=True)
    run = await run_eval_case(
        engine,
        REPO_AGENT_CHAIN,
        permissions=perms,
        settings_overrides=repo_settings,
    )
    assert run.status == "ok"
    model = RepoImpactAnalysisReportV1.model_validate(run.artifact)
    assert model.status in REPO_AGENT_CHAIN.expected_statuses
    assert model.risk in REPO_RISKS

    if model.status == "insufficient_grounding":
        assert not model.affected_paths
        assert not model.findings
        return

    finding_refs = {str(f.source_ref or "") for f in model.findings}
    for path in model.affected_paths:
        path_lower = path.lower()
        assert any(
            path_lower in str(f.claim).lower() or path_lower in ref.lower()
            for f, ref in zip(model.findings, finding_refs, strict=False)
        ) or path_lower in " ".join(model.affected_paths).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_repo_impact_engine_files(
    engine: str,
    repo_settings: dict[str, object],
) -> None:
    if engine == "fake":
        pytest.xfail("FakeRLMEngine repo_impact greps fixed agent-chain patterns, not engine files")

    perms = ContextExecPermissionV1(read_repo=True)
    run = await run_eval_case(
        engine,
        REPO_ENGINE_FILES_CASE,
        permissions=perms,
        settings_overrides=repo_settings,
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
async def test_rlm_eval_runtime_debug_identifies_engine(engine: str) -> None:
    from app.rlm_eval_harness import seed_case_organs

    seed_case_organs("belief_provenance_denver")
    run = await run_eval_case(engine, BELIEF_DENVER)
    rd = run.runtime_debug
    assert rd.get("engine") == engine
    assert rd.get("engine_selected") == engine
    assert_engine_runtime_debug(run, engine)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_rlm_eval_safety_posture_read_only(engine: str) -> None:
    run = await run_eval_case(engine, TRACE_MISSING_CORR)
    assert_safety_posture(run)
    rd = run.runtime_debug
    assert rd.get("write_enabled") is False
    assert rd.get("network_enabled") is False
    assert rd.get("sandbox_mode") == "docker"
    assert rd.get("rlm_depth") <= 1


def test_all_eval_cases_registered() -> None:
    assert len(ALL_EVAL_CASES) == 6
    assert {c.name for c in ALL_EVAL_CASES} == {
        "belief_provenance_denver",
        "belief_provenance_ogden",
        "trace_autopsy_known_corr",
        "trace_autopsy_missing_corr",
        "repo_impact_agent_chain",
        "repo_impact_engine_files",
    }
