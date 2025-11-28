# services/orion-agent-council/app/llm_client.py
from __future__ import annotations

import logging
from typing import Dict, Optional
from uuid import uuid4
import time

from orion.core.bus.service import OrionBus

from .settings import settings
from .models import AgentConfig, PhiSnapshot, SelfField
from .prompt_factory import PromptContext, PromptFactory

logger = logging.getLogger("agent-council.llm")


class BusRPCClient:
    """
    Generic request/response helper over OrionBus.

    This is agnostic to LLM vs anything else; council just happens to
    use it for LLM Gateway right now.
    """

    def __init__(
        self,
        bus: OrionBus,
        intake_channel: str,
        reply_prefix: str,
        timeout_sec: float,
    ) -> None:
        self.bus = bus
        self.intake_channel = intake_channel
        self.reply_prefix = reply_prefix
        self.timeout_sec = timeout_sec

    def request(self, envelope: Dict, correlation_id: str) -> Optional[Dict]:
        reply_channel = f"{self.reply_prefix}:{correlation_id}"

        envelope = {
            **envelope,
            "correlation_id": correlation_id,
            "reply_channel": reply_channel,
        }

        self.bus.publish(self.intake_channel, envelope)
        logger.info(
            "[%s] BusRPC -> %s (reply=%s)",
            correlation_id,
            self.intake_channel,
            reply_channel,
        )

        start = time.monotonic()
        for msg in self.bus.raw_subscribe(reply_channel):
            if msg.get("type") != "message":
                continue

            data = msg.get("data") or {}
            if data.get("correlation_id") != correlation_id:
                continue

            logger.info("[%s] BusRPC got matching reply", correlation_id)
            return data

            # (Generator exits automatically when function returns)

            # If we ever refactor to not early-return, keep timeout checks here.
            # if time.monotonic() - start > self.timeout_sec:
            #     break

        if time.monotonic() - start > self.timeout_sec:
            logger.warning("[%s] BusRPC timed out waiting on %s", correlation_id, reply_channel)
        return None


class LLMClient:
    """
    Council-facing LLM client. It:

      - builds PromptContext
      - creates LLM body with PromptFactory
      - wraps it in a gateway envelope
      - retrieves `payload.text` from the reply
    """

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self.rpc = BusRPCClient(
            bus=bus,
            intake_channel=settings.llm_intake_channel,
            reply_prefix=settings.llm_reply_prefix,
            timeout_sec=settings.council_llm_timeout_sec,
        )

    def generate(
        self,
        agent: AgentConfig,
        *,
        prompt: str,
        history: Optional[list[Dict]] = None,
        phi: Optional[PhiSnapshot] = None,
        self_field: Optional[SelfField] = None,
        persona_state: Optional[Dict] = None,
        source: str = "agent-council",
    ) -> str:
        correlation_id = str(uuid4())

        ctx = PromptContext(
            prompt=prompt,
            history=history,
            phi=phi,
            self_field=self_field,
            persona_state=persona_state,
        )

        body = {
            "model": agent.model or settings.default_model,
            "messages": PromptFactory.build_messages(agent, ctx),
            "options": {
                "backend": agent.backend or settings.default_backend,
                "temperature": agent.temperature,
            },
            "stream": False,
            "return_json": False,
            "trace_id": correlation_id,
            "source": source,
        }

        envelope = {
            "event": "chat",
            "service": settings.llm_service_name,
            "payload": {
                "body": body,
                "agent_name": agent.name,
            },
        }

        logger.info(
            "[%s] LLMClient.generate -> agent='%s' via %s",
            correlation_id,
            agent.name,
            settings.llm_intake_channel,
        )

        reply = self.rpc.request(envelope, correlation_id)
        if not reply:
            logger.warning(
                "[%s] LLMClient timeout waiting for reply for agent '%s'",
                correlation_id,
                agent.name,
            )
            return "[AgentCouncil Error] LLM gateway timeout"

        payload = reply.get("payload") or {}
        text = (payload.get("text") or "").strip()
        if not text:
            logger.warning("[%s] LLMClient empty text for agent '%s'", correlation_id, agent.name)
        return text
