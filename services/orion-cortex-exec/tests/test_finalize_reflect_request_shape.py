"""The finalize 5b judge request survives the Cortex Exec -> Gateway seam intact.

orion/harness/finalize.py puts the grade turn, the structured-output schema and thinking-off
at the top level of the harness_finalize_reflect context. Live 2026-11-03..11 the call had
none of them: one system message, no schema, thinking on, and 46 of 74 judge failures were
the model writing something other than a FinalizeReflectionV1.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.executor import call_step_services
from orion.cognition.plan_loader import build_plan_for_verb
from orion.core.bus.bus_schemas import ChatResponsePayload, ServiceRef
from orion.harness.finalize import (
    FINALIZE_REFLECT_USER_TURN,
    build_finalize_reflect_context,
    finalize_reflect_json_schema,
)
from orion.harness.tests.fixtures import make_appraisal, make_repair_overlay, make_thought

DRAFT = '{"what_i_learned": "Fetched the page once."}'
REQUEST = "Fetch the url below with WebFetch before answering.\nurl=https://example.org/paper"


def _run_judge_step():
    plan = build_plan_for_verb("harness_finalize_reflect", mode="brain")
    step = plan.steps[0]
    ctx = build_finalize_reflect_context(
        correlation_id="c-seam", draft_text=DRAFT, thought=make_thought(),
        substrate_appraisal=make_appraisal(), repair_overlay=make_repair_overlay(),
        user_message=REQUEST, fcc_model_label="agent")
    ctx["mode"] = "brain"
    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(
        return_value=ChatResponsePayload(content='{"alignment_verdict":"aligned"}')
    )) as chat:
        asyncio.run(call_step_services(bus=MagicMock(), source=ServiceRef(name="test"),
                    step=step, ctx=ctx, correlation_id=str(uuid4())))
    return chat.await_args.kwargs["req"]


def test_judge_request_ends_with_the_grade_turn_and_quotes_the_draft():
    req = _run_judge_step()
    roles = [m["role"] if isinstance(m, dict) else m.role for m in req.messages]
    contents = [m["content"] if isinstance(m, dict) else m.content for m in req.messages]
    assert roles == ["system", "user"]
    assert contents[-1] == FINALIZE_REFLECT_USER_TURN
    assert f"<draft_text>\n{DRAFT}\n</draft_text>" in contents[0]
    assert req.raw_user_text == REQUEST


def test_judge_request_carries_schema_and_thinking_off_to_the_gateway():
    req = _run_judge_step()
    assert req.options["structured_output_schema"] == finalize_reflect_json_schema()
    assert req.options["structured_output_method"] == "json_object_schema"
    assert req.options["structured_output_thinking_policy"] == "disabled_for_artifact"
    assert req.options["chat_template_kwargs"] == {"enable_thinking": False}


def test_judge_verb_runs_without_recall():
    plan = build_plan_for_verb("harness_finalize_reflect", mode="brain")
    assert plan.metadata["recall_enabled_default"] == "false"
