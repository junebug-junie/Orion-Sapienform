from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from orion.schemas.agents.schemas import AgentChainRequest, AgentChainResult
from orion.schemas.context_exec import ContextExecRequestV1, ContextExecRunV1

from .agent_compat import agent_chain_request_to_context_exec, context_exec_run_to_agent_chain_result
from .runner import ContextExecRunner

logger = logging.getLogger("orion-context-exec.api")
router = APIRouter()
_runner = ContextExecRunner()


@router.get("/health")
async def health() -> dict:
    from .settings import settings

    return {
        "ok": True,
        "service": "orion-context-exec",
        "version": settings.service_version,
        "rlm_enabled": True,
        "sandbox_mode": settings.context_exec_sandbox_mode,
        "write_enabled": settings.context_exec_write_enabled,
        "max_depth": settings.context_exec_max_depth,
    }


@router.post("/context-exec/run", response_model=ContextExecRunV1)
async def run_context_exec(body: ContextExecRequestV1) -> ContextExecRunV1:
    if not body.text.strip():
        raise HTTPException(400, "text required")
    return await _runner.run(body)


@router.post("/agent/chain/run", response_model=AgentChainResult)
async def run_agent_chain_compat(body: AgentChainRequest) -> AgentChainResult:
    if not body.text.strip():
        raise HTTPException(400, "text required")
    req = agent_chain_request_to_context_exec(body)
    run = await _runner.run(req)
    return context_exec_run_to_agent_chain_result(run, mode=body.mode or "agent")
