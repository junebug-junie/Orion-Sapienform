"""Mind LLM client via Orion bus → LLM gateway (Exec/topic-foundry pattern)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from threading import Thread
from typing import Any, Callable, Dict, Optional, Protocol
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ChatResultPayload, LLMMessage, ServiceRef

from .llm_context import MindLLMRequestContext
from .settings import settings

logger = logging.getLogger("orion-mind.llm")


def _run_blocking(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def extract_json(content: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            return None
    brace_match = re.search(r"\{.*\}", content, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            return None
    return None


class MindLLMClientProtocol(Protocol):
    def request_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        route: str,
        max_tokens: int,
        temperature: float = 0.2,
        thinking: bool = False,
        context: MindLLMRequestContext | None = None,
        timeout_sec: float | None = None,
    ) -> tuple[Optional[Dict[str, Any]], str | None, dict[str, Any]]: ...


class MindLLMClient:
    def __init__(self) -> None:
        self._service_ref = ServiceRef(
            name=settings.SERVICE_NAME,
            node=settings.NODE_NAME,
            version=settings.SERVICE_VERSION,
        )

    def _bus_enabled(self) -> bool:
        return bool(settings.MIND_LLM_USE_BUS and settings.ORION_BUS_ENABLED)

    def request_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        route: str,
        max_tokens: int,
        temperature: float = 0.2,
        thinking: bool = False,
        context: MindLLMRequestContext | None = None,
        timeout_sec: float | None = None,
    ) -> tuple[Optional[Dict[str, Any]], str | None, dict[str, Any]]:
        meta: dict[str, Any] = {"route": route, "phase": context.phase_name if context else None}
        if not self._bus_enabled():
            return None, "bus_disabled", meta
        effective_timeout = float(timeout_sec if timeout_sec is not None else settings.MIND_LLM_TIMEOUT_SEC)
        if effective_timeout <= 0:
            return None, "phase_timeout_budget_exhausted", meta
        options: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "return_json": True,
            "gateway_read_timeout_sec": effective_timeout,
        }
        if thinking:
            options["thinking"] = True
        if context is not None:
            options["mind_run_id"] = context.mind_run_id
            options["mind_phase"] = context.phase_name
            options["mind_router_profile_id"] = context.router_profile_id
        try:
            content, usage, model_used = self._bus_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                route=route,
                options=options,
                context=context,
                timeout_sec=effective_timeout,
            )
            meta["model_used"] = model_used
            meta["usage"] = usage
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Mind LLM request failed route=%s phase=%s correlation_id=%s err=%s",
                route,
                context.phase_name if context else None,
                context.correlation_id if context else None,
                exc,
            )
            return None, str(exc), meta
        if not content:
            return None, "empty_llm_response", meta
        parsed = extract_json(content)
        if parsed is None:
            return None, "json_parse_failed", meta
        return parsed, None, meta

    def _bus_chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        route: str,
        options: Dict[str, Any],
        context: MindLLMRequestContext | None,
        timeout_sec: float,
    ) -> tuple[str, dict[str, Any], str | None]:
        async def _call() -> tuple[str, dict[str, Any], str | None]:
            bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
            await bus.connect()
            try:
                corr = context.envelope_correlation_id() if context else uuid4()
                reply_channel = f"{settings.MIND_LLM_REPLY_PREFIX}:{corr}"
                payload = ChatRequestPayload(
                    messages=[
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=user_prompt),
                    ],
                    route=route,
                    options=options,
                    session_id=context.session_id if context else None,
                )
                trace = context.trace_baggage() if context else {}
                env = BaseEnvelope(
                    kind="llm.chat.request",
                    source=self._service_ref,
                    correlation_id=corr,
                    reply_to=reply_channel,
                    trace=trace,
                    causality_chain=list(context.causality_chain or []) if context else [],
                    payload=payload.model_dump(mode="json"),
                )
                msg = await bus.rpc_request(
                    settings.MIND_LLM_INTAKE_CHANNEL,
                    env,
                    reply_channel=reply_channel,
                    timeout_sec=timeout_sec,
                )
                decoded = bus.codec.decode(msg.get("data"))
                if not decoded.ok:
                    raise RuntimeError(f"LLM bus decode failed: {decoded.error}")
                result = ChatResultPayload.model_validate(decoded.envelope.payload)
                usage = dict(result.usage or {})
                return str(result.content or result.text or ""), usage, result.model_used
            finally:
                await bus.close()

        return _run_blocking(_call())


class FakeMindLLMClient:
    """Deterministic JSON responses for unit tests."""

    def __init__(self, responses: list[Dict[str, Any]] | None = None) -> None:
        self._responses = list(responses or [])
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    def request_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        route: str,
        max_tokens: int,
        temperature: float = 0.2,
        thinking: bool = False,
        context: MindLLMRequestContext | None = None,
        timeout_sec: float | None = None,
    ) -> tuple[Optional[Dict[str, Any]], str | None, dict[str, Any]]:
        self.calls.append(
            {
                "route": route,
                "max_tokens": max_tokens,
                "thinking": thinking,
                "context": context,
                "timeout_sec": timeout_sec,
            }
        )
        if self._idx >= len(self._responses):
            return None, "fake_exhausted", {"route": route}
        payload = self._responses[self._idx]
        self._idx += 1
        return dict(payload), None, {"route": route, "model_used": route}


_client: MindLLMClient | None = None
_override: MindLLMClientProtocol | None = None


def get_llm_client() -> MindLLMClientProtocol:
    global _client
    if _override is not None:
        return _override
    if _client is None:
        _client = MindLLMClient()
    return _client


def set_llm_client_override(client: MindLLMClientProtocol | None) -> None:
    global _override
    _override = client


def set_llm_client_factory(factory: Callable[[], MindLLMClientProtocol] | None) -> None:
    global _client, _override
    if factory is None:
        _override = None
        _client = None
        return
    _override = factory()
