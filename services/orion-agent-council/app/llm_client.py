# services/orion-agent-council/app/llm_client.py
from __future__ import annotations

import logging
import json
from typing import Dict, Optional
from uuid import uuid4
from math import inf

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from .settings import settings
from .models import AgentConfig, PhiSnapshot, SelfField
from .prompt_factory import PromptContext, PromptFactory

logger = logging.getLogger("agent-council.llm")


class LLMClientTimeout(TimeoutError):
    """Raised when the LLM Gateway does not reply within the configured window."""

class LLMClient:
    """
    Async Council-facing LLM client.
    Uses OrionBusAsync.rpc_request for reliable request/reply.
    """

    def __init__(self, bus: OrionBusAsync) -> None:
        self.bus = bus

    async def generate(
        self,
        agent: AgentConfig,
        *,
        prompt: str,
        history: Optional[list[Dict]] = None,
        phi: Optional[PhiSnapshot] = None,
        self_field: Optional[SelfField] = None,
        persona_state: Optional[Dict] = None,
        source: str = "agent-council",
        timeout_sec: Optional[float] = None,
    ) -> str:
        correlation_id = str(uuid4())
        timeout_sec = settings.council_llm_timeout_sec if timeout_sec is None else timeout_sec

        ctx = PromptContext(
            prompt=prompt,
            history=history,
            phi=phi,
            self_field=self_field,
            persona_state=persona_state,
        )

        messages = PromptFactory.build_messages(agent, ctx)

        payload_dict = {
            "messages": messages,
            "options": {
                "temperature": agent.temperature,
            },
            "model": None,
            "profile": None,
            "user_id": None,
            "session_id": None,
        }

        reply_channel = f"{settings.llm_reply_prefix}:{correlation_id}"

        envelope = BaseEnvelope(
            kind="llm.chat.request",
            source=ServiceRef(name=settings.service_name, version=settings.service_version),
            correlation_id=correlation_id,
            reply_to=reply_channel,
            payload=payload_dict
        )

        logger.info(
            "[%s] LLMClient.generate -> agent='%s' via %s",
            correlation_id,
            agent.name,
            settings.llm_intake_channel,
        )

        # RPC Call
        try:
            msg = await self.bus.rpc_request(
                request_channel=settings.llm_intake_channel,
                envelope=envelope,
                reply_channel=reply_channel,
                timeout_sec=timeout_sec,
            )
        except TimeoutError as exc:
            logger.error(
                "[%s] LLMClient timeout waiting for reply on %s (timeout=%.2fs)",
                correlation_id,
                reply_channel,
                timeout_sec if timeout_sec is not None else inf,
            )
            raise LLMClientTimeout(
                 f"LLM Gateway timeout after {timeout_sec:.2f}s waiting on {reply_channel}"
           ) from exc

        # Decode
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            logger.warning("[%s] LLMClient decode error: %s", correlation_id, decoded.error)
            return "[AgentCouncil Error] Decode failure"

        resp_payload = decoded.envelope.payload or {}


        # Surface gateway-side validation errors clearly
        if isinstance(resp_payload, dict) and resp_payload.get("error"):
            err = resp_payload.get("error")
            details = resp_payload.get("details")
            logger.error(
                "[%s] LLMClient gateway error: %s details=%s",
                correlation_id,
                err,
                details,
            )
            return f"[AgentCouncil Error] LLM Gateway error: {err} ({details})"

        # Robust extraction logic
        text = resp_payload.get("content")

        if not text:
            text = resp_payload.get("text")

        if not text:
            if "payload" in resp_payload and isinstance(resp_payload["payload"], dict):
                text = resp_payload["payload"].get("text") or resp_payload["payload"].get("content")
            elif "result" in resp_payload and isinstance(resp_payload["result"], dict):
                text = resp_payload["result"].get("text") or resp_payload["result"].get("content")

        if not text:
            logger.error(
                "[%s] LLMClient EMPTY TEXT. Raw Response keys: %s", 
                correlation_id, 
                list(resp_payload.keys())
            )
            return ""

        return text.strip()
