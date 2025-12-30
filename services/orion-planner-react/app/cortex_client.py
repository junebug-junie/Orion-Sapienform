# services/orion-planner-react/app/cortex_client.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from loguru import logger

from orion.core.bus.service import OrionBus


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_envelope_dict(
    *,
    kind: str,
    payload: Any,
    correlation_id: str,
    reply_to: Optional[str],
    source_name: str,
    source_node: Optional[str] = None,
    source_version: str = "0.1.0",
    causality_chain: Optional[list[str]] = None,
) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "schema": "orion.envelope",
        "kind": kind,
        "source": {"name": source_name, "node": source_node, "version": source_version},
        "correlation_id": correlation_id,
        "causality_chain": causality_chain or [],
        "created_at": _utc_iso(),
        "payload": payload,
    }
    if reply_to:
        env["reply_to"] = reply_to
    return env


def unwrap_rpc_response(raw: Any) -> Any:
    if isinstance(raw, dict):
        if raw.get("schema") == "orion.envelope":
            return raw.get("payload")
        if "status" in raw and "data" in raw:
            return raw.get("data")
    return raw


async def _rpc_request_best_effort(
    *,
    bus: OrionBus,
    request_channel: str,
    message: Dict[str, Any],
    timeout_s: float,
) -> Any:
    try:
        from orion.core.bus.rpc import rpc_request  # type: ignore

        return await rpc_request(
            bus=bus,
            request_channel=request_channel,
            message=message,
            timeout_s=timeout_s,
        )
    except Exception:
        pass

    if hasattr(bus, "rpc_request"):
        return await bus.rpc_request(request_channel, message, timeout_s=timeout_s)  # type: ignore

    raise RuntimeError(
        "No RPC mechanism found. Expected orion.core.bus.rpc.rpc_request or bus.rpc_request()."
    )


class CortexOrchClient:
    """
    Planner-React → Cortex-Orch client.

    Key fix: stop sending legacy dict soup; send a real Titanium envelope:
      kind="cortex.orch.request"
      payload={"verb_name": ..., "args": ..., "context": ...}
    """

    def __init__(
        self,
        *,
        bus: OrionBus,
        request_channel: str,
        timeout_s: float = 120.0,
        reply_prefix: str = "orion:rpc:reply",
        source_name: str = "orion-planner-react",
        source_node: Optional[str] = None,
        source_version: str = "0.1.0",
    ) -> None:
        self.bus = bus
        self.request_channel = request_channel
        self.timeout_s = timeout_s
        self.reply_prefix = reply_prefix
        self.source_name = source_name
        self.source_node = source_node
        self.source_version = source_version

    async def call_verb(
        self,
        *,
        verb_name: str,
        tool_input: Dict[str, Any],
        parent_correlation_id: Optional[str] = None,
        causality_chain: Optional[list[str]] = None,
        timeout_s: Optional[float] = None,
    ) -> Any:
        corr_id = parent_correlation_id or str(uuid.uuid4())
        reply_to = f"{self.reply_prefix}:{corr_id}"

        # Canonical payload shape expected by Cortex-Orch / Cortex-Exec planning stack
        orch_payload: Dict[str, Any] = {
            "verb_name": verb_name,
            "args": tool_input,       # explicit tool args (structured)
            "context": tool_input,    # context rendering (so exec prompts see the keys)
        }

        req_env = _as_envelope_dict(
            kind="cortex.orch.request",
            payload=orch_payload,
            correlation_id=corr_id,
            reply_to=reply_to,
            causality_chain=causality_chain or [],
            source_name=self.source_name,
            source_node=self.source_node,
            source_version=self.source_version,
        )

        logger.info(
            "[PLANNER] call_verb verb_name={} corr_id={} orch_channel={} reply_to={}",
            verb_name,
            corr_id,
            self.request_channel,
            reply_to,
        )

        raw = await _rpc_request_best_effort(
            bus=self.bus,
            request_channel=self.request_channel,
            message=req_env,
            timeout_s=float(timeout_s or self.timeout_s),
        )

        return unwrap_rpc_response(raw)


# If your existing planner code expects `_call_cortex_verb(tool_id, tool_input, ...)`,
# you can import and use this helper directly.
async def _call_cortex_verb(
    *,
    client: CortexOrchClient,
    tool_id: str,
    tool_input: Dict[str, Any],
    trace_id: Optional[str] = None,
    causality_chain: Optional[list[str]] = None,
    timeout_s: Optional[float] = None,
) -> Any:
    return await client.call_verb(
        verb_name=tool_id,
        tool_input=tool_input,
        parent_correlation_id=trace_id,
        causality_chain=causality_chain,
        timeout_s=timeout_s,
    )
