# services/orion-cortex-orch/app/clients.py
from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import uuid4

from pydantic import ValidationError
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.exec import CortexExecResultPayload
from orion.schemas.cortex.schemas import PlanExecutionRequest
from orion.schemas.state.contracts import StateGetLatestRequest, StateLatestReply

logger = logging.getLogger("orion.cortex.orch.clients")

class CortexExecClient:
    """
    Strict, typed client for sending plans to cortex-exec.
    """
    def __init__(self, bus: OrionBusAsync, *, request_channel: str, result_prefix: str):
        self.bus = bus
        self.request_channel = request_channel
        self.result_prefix = result_prefix

    async def execute_plan(
        self,
        source: ServiceRef,
        req: PlanExecutionRequest,
        correlation_id: str,
        timeout_sec: float,
        *,
        trace: dict | None = None,
    ) -> Dict[str, Any]:
        """
        Sends a typed PlanExecutionRequest, returns the raw result dict from Exec.
        """
        reply_channel = f"{self.result_prefix}:{uuid4()}"

        # 1. STRICT: Convert Pydantic -> JSON
        # This ensures we never send a malformed plan
        payload_json = req.model_dump(mode="json")

        env = BaseEnvelope(
            kind=req.kind,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            trace=dict(trace or {}),
            payload=payload_json,
        )

        logger.info(
            "RPC emit -> %s kind=%s corr=%s reply=%s steps=%s",
            self.request_channel,
            env.kind,
            correlation_id,
            reply_channel,
            len(req.plan.steps),
        )

        # 2. TRANSPORT
        msg = await self.bus.rpc_request(
            self.request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec
        )

        # 3. DECODE
        decoded = self.bus.codec.decode(msg.get("data"))

        if not decoded.ok:
            raise RuntimeError(f"Exec RPC failed: {decoded.error}")

        payload = decoded.envelope.payload
        if isinstance(payload, dict):
            incoming_result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
            incoming_reasoning = incoming_result.get("reasoning_content") if isinstance(incoming_result, dict) else None
            incoming_trace = incoming_result.get("reasoning_trace") if isinstance(incoming_result, dict) else None
            incoming_trace_content = incoming_trace.get("content") if isinstance(incoming_trace, dict) else None
            print(
                "===THINK_HOP=== hop=orch_or_gateway_in "
                f"corr={correlation_id} "
                f"payload_keys={sorted(incoming_result.keys()) if isinstance(incoming_result, dict) else []} "
                f"reasoning_len={len(incoming_reasoning) if isinstance(incoming_reasoning, str) else 0} "
                f"trace_len={len(incoming_trace_content) if isinstance(incoming_trace_content, str) else 0} "
                f"metacog_count={len(incoming_result.get('metacog_traces')) if isinstance(incoming_result, dict) and isinstance(incoming_result.get('metacog_traces'), list) else 0} "
                f"preview={repr(str((incoming_reasoning if isinstance(incoming_reasoning, str) else None) or incoming_trace_content or '')[:220])}",
                flush=True,
            )
            try:
                exec_payload = CortexExecResultPayload.model_validate(payload)
            except ValidationError:
                normalized = payload.get("result") or payload
                norm_reasoning = normalized.get("reasoning_content") if isinstance(normalized, dict) else None
                norm_trace = normalized.get("reasoning_trace") if isinstance(normalized, dict) else None
                norm_trace_content = norm_trace.get("content") if isinstance(norm_trace, dict) else None
                print(
                    "===THINK_HOP=== hop=orch_or_gateway_out "
                    f"corr={correlation_id} "
                    f"payload_keys={sorted(normalized.keys()) if isinstance(normalized, dict) else []} "
                    f"reasoning_len={len(norm_reasoning) if isinstance(norm_reasoning, str) else 0} "
                    f"trace_len={len(norm_trace_content) if isinstance(norm_trace_content, str) else 0} "
                    f"metacog_count={len(normalized.get('metacog_traces')) if isinstance(normalized, dict) and isinstance(normalized.get('metacog_traces'), list) else 0} "
                    f"preview={repr(str((norm_reasoning if isinstance(norm_reasoning, str) else None) or norm_trace_content or '')[:220])}",
                    flush=True,
                )
                return normalized
            if exec_payload.result is not None:
                normalized = exec_payload.result.model_dump(mode="json")
                norm_reasoning = normalized.get("reasoning_content") if isinstance(normalized, dict) else None
                norm_trace = normalized.get("reasoning_trace") if isinstance(normalized, dict) else None
                norm_trace_content = norm_trace.get("content") if isinstance(norm_trace, dict) else None
                print(
                    "===THINK_HOP=== hop=orch_or_gateway_out "
                    f"corr={correlation_id} "
                    f"payload_keys={sorted(normalized.keys()) if isinstance(normalized, dict) else []} "
                    f"reasoning_len={len(norm_reasoning) if isinstance(norm_reasoning, str) else 0} "
                    f"trace_len={len(norm_trace_content) if isinstance(norm_trace_content, str) else 0} "
                    f"metacog_count={len(normalized.get('metacog_traces')) if isinstance(normalized, dict) and isinstance(normalized.get('metacog_traces'), list) else 0} "
                    f"preview={repr(str((norm_reasoning if isinstance(norm_reasoning, str) else None) or norm_trace_content or '')[:220])}",
                    flush=True,
                )
                return normalized
            if exec_payload.error:
                normalized = {"ok": False, "error": exec_payload.error, "details": exec_payload.details}
                print(
                    "===THINK_HOP=== hop=orch_or_gateway_out "
                    f"corr={correlation_id} "
                    f"payload_keys={sorted(normalized.keys())} "
                    "reasoning_len=0 trace_len=0 metacog_count=0 preview=''",
                    flush=True,
                )
                return normalized
            print(
                "===THINK_HOP=== hop=orch_or_gateway_out "
                f"corr={correlation_id} "
                f"payload_keys={sorted(payload.keys())} "
                "reasoning_len=0 trace_len=0 metacog_count=0 preview=''",
                flush=True,
            )
            return payload

        # Just in case Exec sends back something weird
        normalized = decoded.envelope.model_dump(mode="json")
        print(
            "===THINK_HOP=== hop=orch_or_gateway_out "
            f"corr={correlation_id} "
            f"payload_keys={sorted(normalized.keys()) if isinstance(normalized, dict) else []} "
            "reasoning_len=0 trace_len=0 metacog_count=0 preview=''",
            flush=True,
        )
        return normalized


class StateServiceClient:
    """
    Strict, typed client for requesting the latest Orion state (Spark snapshot)
    from orion-state-service.
    """
    def __init__(self, bus: OrionBusAsync, *, request_channel: str, result_prefix: str):
        self.bus = bus
        self.request_channel = request_channel
        self.result_prefix = result_prefix

    async def get_latest(
        self,
        *,
        source: ServiceRef,
        req: StateGetLatestRequest,
        correlation_id: str,
        timeout_sec: float
    ) -> StateLatestReply:
        reply_channel = f"{self.result_prefix}:{uuid4()}"

        env = BaseEnvelope(
            kind=req.kind,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            payload=req.model_dump(mode="json"),
        )

        msg = await self.bus.rpc_request(
            self.request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec,
        )

        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"State RPC decode failed: {decoded.error}")

        payload = decoded.envelope.payload
        if isinstance(payload, dict):
            return StateLatestReply.model_validate(payload)

        # Fallback: try to validate whole envelope
        return StateLatestReply.model_validate(decoded.envelope.model_dump(mode="json").get("payload") or {})
