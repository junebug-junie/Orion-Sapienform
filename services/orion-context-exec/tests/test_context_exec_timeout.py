from __future__ import annotations

import pytest

from app.rlm_engine import TimeoutRLMEngine
from app.runner import ContextExecRunner
from orion.schemas.context_exec import ContextExecBudgetV1, ContextExecRequestV1


@pytest.mark.asyncio
async def test_context_exec_timeout() -> None:
    runner = ContextExecRunner(engine=TimeoutRLMEngine())
    req = ContextExecRequestV1(
        text="loop forever",
        mode="general_investigation",
        budget=ContextExecBudgetV1(max_seconds=0.05),
    )
    run = await runner.run(req)
    assert run.status == "timeout"
    assert "timeout" in run.failure_modes
    assert "insufficient grounding" in run.final_text.lower() or "could not complete" in run.final_text.lower()
