"""Hub client for context-exec agent lane (read-only / proposal-gated)."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from orion.schemas.agents.schemas import AgentChainRequest
from orion.schemas.context_exec import ContextExecRequestV1, ContextExecRunV1

from scripts.settings import settings

logger = logging.getLogger("orion-hub.context-exec")


class ContextExecClientError(Exception):
    """Controlled context-exec client failure."""


def _base_url() -> str:
    return str(settings.HUB_CONTEXT_EXEC_API_URL or "").strip().rstrip("/")


def _timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=float(settings.HUB_CONTEXT_EXEC_TIMEOUT_SEC))


def agent_lane_enabled() -> bool:
    return bool(settings.HUB_AGENT_CONTEXT_EXEC_ENABLED)


async def run_context_exec(body: ContextExecRequestV1) -> ContextExecRunV1:
    base = _base_url()
    if not base:
        raise ContextExecClientError("HUB_CONTEXT_EXEC_API_URL is not configured")
    url = f"{base}/context-exec/run"
    payload = body.model_dump(mode="json")
    try:
        async with aiohttp.ClientSession(timeout=_timeout()) as session:
            async with session.post(url, json=payload) as response:
                raw = await response.json()
                if response.status >= 400:
                    raise ContextExecClientError(
                        f"context-exec run HTTP {response.status}: {raw!r}"[:400]
                    )
    except aiohttp.ClientError as exc:
        logger.warning("context-exec run unreachable: %s", exc)
        raise ContextExecClientError("context-exec run unreachable") from exc
    return ContextExecRunV1.model_validate(raw)


async def run_agent_chain_compat(body: AgentChainRequest) -> dict[str, Any]:
    base = _base_url()
    if not base:
        raise ContextExecClientError("HUB_CONTEXT_EXEC_API_URL is not configured")
    url = f"{base}/agent/chain/run"
    payload = body.model_dump(mode="json")
    try:
        async with aiohttp.ClientSession(timeout=_timeout()) as session:
            async with session.post(url, json=payload) as response:
                raw = await response.json()
                if response.status >= 400:
                    raise ContextExecClientError(
                        f"context-exec agent compat HTTP {response.status}: {raw!r}"[:400]
                    )
    except aiohttp.ClientError as exc:
        logger.warning("context-exec agent compat unreachable: %s", exc)
        raise ContextExecClientError("context-exec agent compat unreachable") from exc
    if not isinstance(raw, dict):
        raise ContextExecClientError("context-exec agent compat returned non-object payload")
    return raw
