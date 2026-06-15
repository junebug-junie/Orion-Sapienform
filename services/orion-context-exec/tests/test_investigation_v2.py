"""Tests for investigation_v2 skeleton handler (PR1)."""

from __future__ import annotations

import pytest

from orion.schemas.context_exec import ContextExecRequestV1, context_exec_permissions_for_llm_profile

from app.investigation_v2 import INVESTIGATION_V2_SKELETON_MESSAGE, build_investigation_v2_skeleton_artifact
from app.runner import ContextExecRunner


def test_investigation_v2_skeleton_artifact_shape() -> None:
    req = ContextExecRequestV1(
        text="what would happen if we changed the cortex-exec runtime?",
        mode="investigation_v2",
        permissions=context_exec_permissions_for_llm_profile("agent"),
    )
    artifact = build_investigation_v2_skeleton_artifact(req)
    assert artifact["mode"] == "investigation_v2"
    assert artifact["read_repo"] is True
    assert artifact["permissions_received"]["read_repo"] is True
    assert artifact["message"] == INVESTIGATION_V2_SKELETON_MESSAGE


@pytest.mark.asyncio
async def test_runner_investigation_v2_returns_skeleton_without_organs() -> None:
    runner = ContextExecRunner()
    req = ContextExecRequestV1(
        text="what would happen if we changed the cortex-exec runtime?",
        mode="investigation_v2",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        llm_profile="agent",
    )
    run = await runner.run(req)
    assert run.status == "ok"
    assert run.mode == "investigation_v2"
    assert run.artifact_type == "InvestigationV2SkeletonV1"
    assert run.artifact["read_repo"] is True
    assert run.final_text == INVESTIGATION_V2_SKELETON_MESSAGE
    assert run.runtime_debug.get("investigation_v2_skeleton") is True


@pytest.mark.asyncio
async def test_agent_compat_v2_skips_keyword_mode_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    from orion.schemas.agents.schemas import AgentChainRequest

    from app.agent_compat import agent_chain_request_to_context_exec
    from app.settings import settings

    monkeypatch.setattr(settings, "context_exec_investigation_v2_enabled", True)
    body = AgentChainRequest(
        text="what breaks if we replace agent-chain-service with context-exec?",
        mode="agent",
        response_profile="agent",
    )
    req = agent_chain_request_to_context_exec(body)
    assert req.mode == "investigation_v2"
    assert req.permissions.read_repo is True
