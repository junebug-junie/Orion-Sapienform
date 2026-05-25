from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.grammar_emit import CortexExecGrammarCollector, build_cortex_exec_grammar_events
from app.grammar_publish import publish_cortex_exec_grammar_trace
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionRequest


@pytest.mark.asyncio
async def test_publish_failure_is_non_fatal() -> None:
    bus = AsyncMock()
    bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))
    collector = CortexExecGrammarCollector(
        node_name="athena",
        correlation_id="c1",
        code_version="0.2.0",
        observed_at=datetime.now(timezone.utc),
    )
    req = PlanExecutionRequest(
        plan=ExecutionPlan(verb_name="chat_quick", steps=[]),
        args={"request_id": "r1"},
    )
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=0)
    collector.record_recall_gate_observed(run_recall=False, profile=None, reason="skip")
    collector.record_result_assembled(
        status="success",
        final_text_present=False,
        reasoning_present=False,
        thinking_source="none",
    )
    collector.record_result_emitted(reply_present=False, status="success")
    events = build_cortex_exec_grammar_events(collector)
    await publish_cortex_exec_grammar_trace(
        bus,
        events,
        correlation_id="c1",
        channel="orion:grammar:event",
        source_name="orion-cortex-exec",
    )
