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
    ) -> tuple[Optional[Dict[str, Any]], str | None]: ...


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
    ) -> tuple[Optional[Dict[str, Any]], str | None]:
        if not self._bus_enabled():
            return None, "bus_disabled"
        options: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "return_json": True,
            "gateway_read_timeout_sec": float(settings.MIND_LLM_TIMEOUT_SEC),
        }
        if thinking:
            options["thinking"] = True
        try:
            content = self._bus_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                route=route,
                options=options,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mind LLM request failed route=%s err=%s", route, exc)
            return None, str(exc)
        if not content:
            return None, "empty_llm_response"
        parsed = extract_json(content)
        if parsed is None:
            return None, "json_parse_failed"
        return parsed, None

    def _bus_chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        route: str,
        options: Dict[str, Any],
    ) -> str:
        async def _call() -> str:
            bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
            await bus.connect()
            try:
                correlation_id = str(uuid4())
                reply_channel = f"{settings.MIND_LLM_REPLY_PREFIX}:{correlation_id}"
                payload = ChatRequestPayload(
                    messages=[
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=user_prompt),
                    ],
                    route=route,
                    options=options,
                )
                env = BaseEnvelope(
                    kind="llm.chat.request",
                    source=self._service_ref,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    payload=payload.model_dump(mode="json"),
                )
                msg = await bus.rpc_request(
                    settings.MIND_LLM_INTAKE_CHANNEL,
                    env,
                    reply_channel=reply_channel,
                    timeout_sec=float(settings.MIND_LLM_TIMEOUT_SEC),
                )
                decoded = bus.codec.decode(msg.get("data"))
                if not decoded.ok:
                    raise RuntimeError(f"LLM bus decode failed: {decoded.error}")
                result = ChatResultPayload.model_validate(decoded.envelope.payload)
                return result.content or result.text
            finally:
                await bus.close()

        return str(_run_blocking(_call()) or "")


class FakeMindLLMClient:
    """Deterministic JSON responses for unit tests."""

    def __init__(self, responses: list[Dict[str, Any]] | None = None) -> None:
        self._responses = list(responses or [])
        self._idx = 0

    def request_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        route: str,
        max_tokens: int,
        temperature: float = 0.2,
        thinking: bool = False,
    ) -> tuple[Optional[Dict[str, Any]], str | None]:
        if self._idx >= len(self._responses):
            return None, "fake_exhausted"
        payload = self._responses[self._idx]
        self._idx += 1
        return dict(payload), None


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
