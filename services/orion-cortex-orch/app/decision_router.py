from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ServiceRef
from orion.schemas.cortex.contracts import AutoRouteDecisionV1, CortexClientRequest

from .settings import get_settings

logger = logging.getLogger("orion.cortex.orch.router")

_ALLOWED_PACKS = {"executive_pack"}
_ALLOWED_VERBS = {
    "chat_general": ("chat", "brain"),
    "agent_runtime": ("agent", "agent"),
    "council_runtime": ("council", "council"),
}


@dataclass(frozen=True)
class RoutedRequest:
    request: CortexClientRequest
    decision: AutoRouteDecisionV1


class DecisionRouter:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self.settings = get_settings()

    def heuristic_router(self, req: CortexClientRequest) -> AutoRouteDecisionV1:
        text = " ".join(
            part for part in [req.context.raw_user_text or "", req.context.user_message or ""] if part
        ).strip().lower()
        msg_len = len(text)

        agent_terms = (
            "implement", "build", "fix", "refactor", "patch", "run", "test", "debug", "deploy", "code"
        )
        council_terms = (
            "debate", "tradeoff", "compare approaches", "pros and cons", "multi-perspective", "council"
        )
        depth_terms = ("step by step", "deep", "thorough", "reason carefully")

        agent_score = sum(1 for t in agent_terms if t in text)
        council_score = sum(1 for t in council_terms if t in text)
        if msg_len > 900:
            agent_score += 1
        if any(t in text for t in depth_terms):
            council_score += 1

        if council_score >= 2:
            return AutoRouteDecisionV1(
                route_mode="council",
                verb="council_runtime",
                packs=["executive_pack"],
                confidence=0.72,
                reason="heuristic:council_terms",
                source="heuristic",
            )

        if agent_score >= 1:
            return AutoRouteDecisionV1(
                route_mode="agent",
                verb="agent_runtime",
                packs=["executive_pack"],
                confidence=0.78,
                reason="heuristic:execution_intent",
                source="heuristic",
            )

        return AutoRouteDecisionV1(
            route_mode="chat",
            verb="chat_general",
            packs=["executive_pack"],
            confidence=0.66,
            reason="heuristic:default_chat",
            source="heuristic",
        )

    async def llm_router(self, req: CortexClientRequest, *, correlation_id: str, source: ServiceRef) -> AutoRouteDecisionV1:
        prompt = self._build_prompt(req)
        payload = ChatRequestPayload(
            route="chat",
            messages=[
                {"role": "user", "content": prompt},
            ],
            raw_user_text=req.context.raw_user_text or req.context.user_message,
            options={
                "temperature": 0.0,
                "max_tokens": 200,
                "stream": False,
                "response_format": {"type": "json_object"},
            },
            user_id=req.context.user_id,
            session_id=req.context.session_id,
        )

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return await asyncio.wait_for(
                    self._rpc_llm(payload=payload, correlation_id=correlation_id, source=source),
                    timeout=5.0,
                )
            except Exception as exc:
                last_error = exc
                logger.warning("auto_route llm attempt=%s failed corr=%s err=%s", attempt + 1, correlation_id, exc)

        raise RuntimeError(f"llm_router_failed:{last_error}")

    async def route(self, req: CortexClientRequest, *, correlation_id: str, source: ServiceRef) -> RoutedRequest:
        if self.settings.auto_router_llm_enabled:
            try:
                decision = await self.llm_router(req, correlation_id=correlation_id, source=source)
            except Exception:
                fallback = self.heuristic_router(req)
                decision = fallback.model_copy(update={"source": "fallback", "reason": "fallback:llm_failure"})
        else:
            decision = self.heuristic_router(req)

        clamped = self._clamp_decision(req, decision)
        rewritten = req.model_copy(deep=True)
        rewritten.mode = "brain" if clamped.route_mode == "chat" else clamped.route_mode
        rewritten.verb = clamped.verb
        rewritten.packs = list(clamped.packs)
        rewritten.recall.enabled = bool(clamped.recall.enabled)
        rewritten.recall.required = bool(clamped.recall.required)
        rewritten.recall.profile = clamped.recall.profile

        return RoutedRequest(request=rewritten, decision=clamped)

    def _clamp_decision(self, req: CortexClientRequest, decision: AutoRouteDecisionV1) -> AutoRouteDecisionV1:
        verb = decision.verb if decision.verb in _ALLOWED_VERBS else "chat_general"
        route_mode, rewritten_mode = _ALLOWED_VERBS[verb]
        safe_packs = [p for p in decision.packs if p in _ALLOWED_PACKS]
        if not safe_packs:
            safe_packs = ["executive_pack"]

        recall_enabled = bool(decision.recall.enabled)
        recall_required = bool(decision.recall.required and recall_enabled)
        profile = decision.recall.profile if recall_enabled else None

        return AutoRouteDecisionV1(
            route_mode=route_mode,
            verb=verb,
            packs=safe_packs,
            recall={
                "enabled": recall_enabled,
                "required": recall_required,
                "profile": profile,
            },
            confidence=max(0.0, min(1.0, float(decision.confidence))),
            reason=decision.reason or f"clamped_for_{rewritten_mode}",
            source=decision.source,
        )

    async def _rpc_llm(self, *, payload: ChatRequestPayload, correlation_id: str, source: ServiceRef) -> AutoRouteDecisionV1:
        reply_channel = f"{self.settings.auto_router_llm_reply_prefix}:{uuid4()}"
        env = BaseEnvelope(
            kind="llm.chat.request",
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            payload=payload.model_dump(mode="json"),
        )
        message = await self.bus.rpc_request(
            self.settings.auto_router_llm_request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=5.0,
        )
        decoded = self.bus.codec.decode(message.get("data"))
        if not decoded.ok:
            raise RuntimeError(decoded.error or "llm_decode_failed")
        response_payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
        text = str(response_payload.get("content") or response_payload.get("text") or "").strip()
        if not text:
            raw = response_payload.get("raw") or {}
            text = str(raw.get("content") or raw.get("text") or "").strip()
        data = json.loads(text)
        return AutoRouteDecisionV1.model_validate(data)

    def _build_prompt(self, req: CortexClientRequest) -> str:
        try:
            user_text = req.context.raw_user_text or req.context.user_message or ""
            history = req.context.messages[-6:]
            history_lines = [f"- {m.role}: {m.content}" for m in history]
        except Exception:
            user_text = req.context.raw_user_text or req.context.user_message or ""
            history_lines = []

        return (
            "Use schema AutoRouteDecisionV1 and return strict JSON only.\\n"
            "Allowed route_mode: chat|agent|council.\\n"
            "Allowed verb: chat_general|agent_runtime|council_runtime.\\n"
            "Allowed packs: executive_pack.\\n"
            "Prefer agent for execution/build/refactor intents, chat for conversational Q&A, council for multi-perspective deliberation.\\n"
            f"User message: {user_text}\\n"
            f"Recent history:\\n{chr(10).join(history_lines)}"
        )
