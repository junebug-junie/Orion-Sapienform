# services/orion-hub/scripts/agent_chain_rpc.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .settings import settings


async def call_agent_chain(
    text: str,
    *,
    mode: str = "chat",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Thin HTTP client for the orion-agent-chain /chain/run endpoint.
    Mirrors AgentChainRequest in services/orion-agent-chain/app/api.py
    """
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

    async with httpx.AsyncClient(timeout=settings.default_http_timeout_seconds) as client:
        resp = await client.post(settings.agent_chain_url, json=payload)
        resp.raise_for_status()
        return resp.json()
