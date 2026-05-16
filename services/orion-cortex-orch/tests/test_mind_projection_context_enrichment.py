from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_guard = Path(__file__).resolve().parent / "_orch_import_guard.py"
_spec = importlib.util.spec_from_file_location("_orch_guard_boot", _guard)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _orch_prep() -> None:
    _guard_mod.ensure_orion_cortex_orch_app()


from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage, RecallDirective
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionRequest


def _plan_request() -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="chat_general",
            steps=[ExecutionStep(verb_name="chat_general", step_name="noop", order=0, services=[])],
            metadata={"personality_file": "orion/cognition/personality/orion_identity.yaml"},
        ),
        context={"metadata": {}},
    )


def _client_request() -> CortexClientRequest:
    return CortexClientRequest(
        verb="chat_general",
        mode="brain",
        recall=RecallDirective(enabled=True, profile="reflect.v1"),
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="Who are you?")],
            session_id="sess",
            trace_id="trace",
            user_message="Who are you?",
            metadata={"mind_enabled": True},
        ),
    )


@pytest.mark.asyncio
async def test_prepare_plan_context_injects_identity_and_recall() -> None:
    _orch_prep()
    from app import mind_runtime

    plan = _plan_request()
    client = _client_request()
    recall_merge = {
        "recall_bundle": {"fragments": [{"snippet": "past", "source": "journal"}], "citations": []},
        "memory_used": True,
    }
    with patch.object(
        mind_runtime,
        "prefetch_recall_bundle_for_projection",
        new=AsyncMock(return_value=recall_merge),
    ):
        await mind_runtime.prepare_plan_context_for_mind_projection(
            object(),
            source=ServiceRef(name="cortex-orch", version="0.2.0", node="test"),
            client_request=client,
            plan_request=plan,
            correlation_id="corr-enrich",
        )
    ctx = plan.context
    assert ctx["orion_identity_summary"]
    assert ctx["recall_bundle"]["fragments"]
    assert plan.context["metadata"]["orch_preflight_input_summary"]["recall_bundle_present"] is True
