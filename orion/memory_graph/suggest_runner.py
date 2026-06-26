from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.memory_graph.schema_contract import compact_suggest_draft_json_schema
from orion.memory_graph.suggest_token_budget import (
    SuggestTokenBudgetConfig,
    completion_budget_for_transcript,
    suggest_token_budget_config_from_env,
)
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage, RecallDirective

_SUGGEST_ROUTE_ATTEMPTS = ("quick", "metacog")


def build_memory_graph_suggest_options(
    *,
    llm_route: str = "quick",
    transcript: str = "",
    budget_config: SuggestTokenBudgetConfig | None = None,
) -> Dict[str, Any]:
    """LLM options aligned with Hub memory_graph_suggest structured output."""
    cfg = budget_config or suggest_token_budget_config_from_env()
    return {
        "llm_route": llm_route,
        "no_write": True,
        "skip_brain_reply_context": True,
        "skip_unified_beliefs": True,
        "skip_autonomy_context": True,
        "skip_chat_stance_inputs": True,
        "structured_output_schema_name": "SuggestDraftV1",
        "structured_output_schema": compact_suggest_draft_json_schema(),
        "structured_output_method": "json_object_schema",
        "structured_output_thinking_policy": "disabled_for_artifact",
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.1,
        "max_tokens": completion_budget_for_transcript(transcript, config=cfg),
    }


async def suggest_once(
    bus: OrionBusAsync,
    *,
    transcript: str,
    cortex_request_channel: str,
    cortex_result_prefix: str,
    source: ServiceRef,
    timeout_sec: float = 120.0,
    session_id: Optional[str] = None,
    llm_route: str = "quick",
    budget_config: SuggestTokenBudgetConfig | None = None,
) -> Dict[str, Any]:
    trace_id = str(uuid.uuid4())
    reply_channel = f"{cortex_result_prefix}:{trace_id}"
    ctx = CortexClientContext(
        messages=[LLMMessage(role="user", content=transcript)],
        session_id=session_id,
        trace_id=trace_id,
        metadata={"transcript": transcript, "consolidation_suggest": True},
    )
    cortex_req = CortexClientRequest(
        mode="brain",
        route_intent="none",
        verb="memory_graph_suggest",
        packs=[],
        options=build_memory_graph_suggest_options(
            llm_route=llm_route,
            transcript=transcript,
            budget_config=budget_config,
        ),
        recall=RecallDirective(enabled=False),
        context=ctx,
    )
    envelope = BaseEnvelope(
        kind="cortex.orch.request",
        source=source,
        correlation_id=trace_id,
        reply_to=reply_channel,
        payload=cortex_req.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(
        cortex_request_channel,
        envelope,
        reply_channel=reply_channel,
        timeout_sec=timeout_sec,
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(f"Cortex RPC decode failed: {decoded.error}")
    payload = decoded.envelope.payload
    if isinstance(payload, str):
        return json.loads(payload)
    return payload if isinstance(payload, dict) else payload.model_dump(mode="json")


async def suggest_with_escalation(
    bus: OrionBusAsync,
    *,
    transcript: str,
    cortex_request_channel: str,
    cortex_result_prefix: str,
    source: ServiceRef,
    timeout_sec: float = 120.0,
    session_id: Optional[str] = None,
    budget_config: SuggestTokenBudgetConfig | None = None,
) -> Dict[str, Any]:
    """Try quick lane first, then metacog if draft JSON cannot be extracted."""
    from orion.memory_graph.cortex_suggest_extract import extract_suggest_draft_dict_from_cortex_payload

    last_raw: Dict[str, Any] = {}
    last_err: Exception | None = None
    per_attempt_timeout = float(timeout_sec)
    cfg = budget_config or suggest_token_budget_config_from_env()
    for route in _SUGGEST_ROUTE_ATTEMPTS:
        try:
            raw = await suggest_once(
                bus,
                transcript=transcript,
                cortex_request_channel=cortex_request_channel,
                cortex_result_prefix=cortex_result_prefix,
                source=source,
                timeout_sec=per_attempt_timeout,
                session_id=session_id,
                llm_route=route,
                budget_config=cfg,
            )
            last_raw = raw if isinstance(raw, dict) else {}
            extract_suggest_draft_dict_from_cortex_payload(last_raw)
            return last_raw
        except Exception as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    return last_raw
