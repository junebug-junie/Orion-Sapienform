from __future__ import annotations

import asyncio
import json
import logging
import re
from threading import Thread
from typing import Any, Dict, Optional
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ChatResultPayload, LLMMessage, ServiceRef

from app.settings import settings


logger = logging.getLogger("topic-foundry.llm")


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


def _extract_json(content: str) -> Optional[Dict[str, Any]]:
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


class TopicFoundryLLMClient:
    def __init__(self) -> None:
        self._service_ref = ServiceRef(
            name=settings.service_name,
            node=settings.node_name,
            version=settings.service_version,
        )

    def _use_bus(self) -> bool:
        return bool(settings.topic_foundry_llm_use_bus and settings.orion_bus_enabled)

    def request_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        use_bus = self._use_bus()
        try:
            if not use_bus:
                logger.warning("LLM request skipped; bus transport disabled")
                return None
            content = self._bus_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM request failed: %s", exc)
            return None

        if not content:
            return None
        return _extract_json(content)

    def probe(self) -> Dict[str, Any]:
        if not settings.topic_foundry_llm_enable:
            return {"ok": True, "detail": "disabled"}
        if self._use_bus():
            return self._probe_bus()
        return {"ok": True, "detail": "bus_disabled"}

    def _probe_bus(self) -> Dict[str, Any]:
        async def _ping() -> Dict[str, Any]:
            bus = OrionBusAsync(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)
            await bus.connect()
            await bus.close()
            return {"ok": True, "detail": "bus_ok"}

        try:
            return _run_blocking(_ping())
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM bus probe failed: %s", exc)
            return {"ok": False, "detail": str(exc)}

    def _bus_chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        async def _call() -> str:
            bus = OrionBusAsync(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)
            await bus.connect()
            try:
                correlation_id = str(uuid4())
                reply_channel = f"{settings.topic_foundry_llm_reply_prefix}:{correlation_id}"
                options: Dict[str, Any] = {"temperature": temperature, "return_json": True}
                if max_tokens is not None:
                    options["max_tokens"] = max_tokens
                payload = ChatRequestPayload(
                    messages=[
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=user_prompt),
                    ],
                    route=settings.topic_foundry_llm_route or None,
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
                    settings.topic_foundry_llm_intake_channel,
                    env,
                    reply_channel=reply_channel,
                    timeout_sec=settings.topic_foundry_llm_timeout_secs,
                )
                decoded = bus.codec.decode(msg.get("data"))
                if not decoded.ok:
                    raise RuntimeError(f"LLM bus decode failed: {decoded.error}")
                result = ChatResultPayload.model_validate(decoded.envelope.payload)
                return result.content or result.text
            finally:
                await bus.close()

        return str(_run_blocking(_call()) or "")


_llm_client: Optional[TopicFoundryLLMClient] = None


def get_llm_client() -> TopicFoundryLLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = TopicFoundryLLMClient()
    return _llm_client
