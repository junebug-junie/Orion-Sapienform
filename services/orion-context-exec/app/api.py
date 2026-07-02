from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from orion.schemas.agents.schemas import AgentChainRequest, AgentChainResult
from orion.schemas.context_exec import ContextExecRequestV1, ContextExecRunV1

from .agent_lane_health import agent_lane_health_block
from .agent_compat import agent_chain_request_to_context_exec, context_exec_run_to_agent_chain_result
from .runner import ContextExecRunner

logger = logging.getLogger("orion-context-exec.api")
router = APIRouter()
_runner: ContextExecRunner | None = None


def set_runner(runner: ContextExecRunner) -> None:
    global _runner
    _runner = runner


def _get_runner() -> ContextExecRunner:
    if _runner is None:
        return ContextExecRunner()
    return _runner


@router.get("/health")
async def health(request: Request) -> dict:
    from .bus_dependency_preflight import collect_bus_dependencies_health
    from .proposal_review_api import proposal_review_health_block
    from .settings import settings
    from .storage import storage_health_block

    lane = agent_lane_health_block()
    bus = getattr(request.app.state, "bus", None)
    bus_health = await collect_bus_dependencies_health(
        bus,
        timeout_sec=float(settings.context_exec_bus_readiness_timeout_sec),
    )
    return {
        "ok": True,
        "service": "orion-context-exec",
        "version": settings.service_version,
        "rlm_enabled": True,
        "sandbox_mode": settings.context_exec_sandbox_mode,
        "write_enabled": settings.context_exec_write_enabled,
        "max_depth": settings.context_exec_max_depth,
        "agent_repl_max_steps": settings.context_exec_agent_repl_max_steps,
        "max_seconds": settings.context_exec_max_seconds,
        "llm_timeout_sec": settings.context_exec_llm_timeout_sec,
        "storage": storage_health_block(),
        "proposal_review_api": proposal_review_health_block(),
        **lane,
        **bus_health,
    }


@router.post("/context-exec/run", response_model=ContextExecRunV1)
async def run_context_exec(body: ContextExecRequestV1) -> ContextExecRunV1:
    if not body.text.strip():
        raise HTTPException(400, "text required")
    return await _get_runner().run(body)


@router.post("/agent/chain/run", response_model=AgentChainResult)
async def run_agent_chain_compat(body: AgentChainRequest) -> AgentChainResult:
    if not body.text.strip():
        raise HTTPException(400, "text required")
    req = agent_chain_request_to_context_exec(body)
    run = await _get_runner().run(req)
    return context_exec_run_to_agent_chain_result(run, mode=body.mode or "agent")
