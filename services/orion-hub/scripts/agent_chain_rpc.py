# services/orion-hub/scripts/agent_chain_rpc.py

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from orion.core.bus.service import OrionBus

from .settings import settings
from .llm_rpc import _request_and_wait  # reuse the existing RPC helper

logger = logging.getLogger("hub.agent-chain-rpc")


class AgentChainRPC:
    """
    Bus-RPC client for the Orion Agent Chain service.

    Mirrors AgentChainRequest in services/orion-agent-chain/app/api.py:

      {
        "text": str,
        "mode": str,
        "session_id": str | None,
        "user_id": str | None,
        "messages": list[{"role": str, "content": str}] | None,
        "tools": list[dict] | None
      }

    And expects AgentChainResult back:

      {
        "mode": str,
        "text": str,
        "structured": dict,
        "planner_raw": dict
      }
    """

    def __init__(self, bus: OrionBus):
        self.bus = bus

        # Logical channels configured in hub settings; must match agent-chain service
        self.request_channel = getattr(
            settings, "CHANNEL_AGENT_CHAIN_INTAKE", "orion-agent-chain:request"
        )
        self.result_prefix = getattr(
            settings, "CHANNEL_AGENT_CHAIN_REPLY_PREFIX", "orion-agent-chain:result"
        )

        if not self.bus or not getattr(self.bus, "enabled", False):
            logger.warning("[AgentChainRPC] OrionBus is disabled; calls will fail.")

        logger.info(
            "[AgentChainRPC] Initialized (request_channel=%s, result_prefix=%s)",
            self.request_channel,
            self.result_prefix,
        )

    async def run(
        self,
        *,
        text: str,
        mode: str = "chat",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a single AgentChainRequest over the bus and wait for AgentChainResult.
        """
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("AgentChainRPC used while OrionBus is disabled")

        trace_id = trace_id or str(uuid.uuid4())
        reply_channel = f"{self.result_prefix}:{trace_id}"

        payload: Dict[str, Any] = {
            "text": text,
            "mode": mode,
            "session_id": session_id,
            "user_id": user_id,
        }
        if messages:
            payload["messages"] = messages
        if tools:
            payload["tools"] = tools

        envelope: Dict[str, Any] = {
            "trace_id": trace_id,
            "reply_channel": reply_channel,
            "payload": payload,
        }

        raw_reply = await _request_and_wait(
            self.bus,
            self.request_channel,
            reply_channel,
            envelope,
            trace_id,
        )

        # Agent Chain publishes AgentChainResult.model_dump()
        if isinstance(raw_reply, dict):
            return raw_reply

        # Fallback: wrap non-dict reply
        return {"raw": raw_reply}


# Optional convenience wrapper if you prefer the old function-style call
async def call_agent_chain(
    bus: OrionBus,
    text: str,
    *,
    mode: str = "chat",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rpc = AgentChainRPC(bus)
    return await rpc.run(
        text=text,
        mode=mode,
        session_id=session_id,
        user_id=user_id,
        messages=messages,
        tools=tools,
    )
