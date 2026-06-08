from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    from orion.core.bus.async_service import OrionBusAsync

logger = logging.getLogger("orion.graph-compression.summarizer")

_LLM_PROMPT_TMPL = """You are summarizing a cluster of semantic memory nodes from an AI cognitive system.

Graph scope: {scope}
Cluster kind: {kind}
Number of nodes: {node_count}
Sample node URIs (up to 5):
{sample_nodes}

Write a concise 1-3 sentence summary of what this cluster represents.
Be specific about the topics, entities, or tensions visible in the node URIs.
Max {max_tokens} tokens.
"""


class RegionSummarizer:
    def __init__(
        self,
        *,
        bus: "OrionBusAsync",
        llm_channel: str,
        service_name: str,
        service_version: str,
        max_tokens: int = 200,
        timeout_sec: float = 15.0,
    ) -> None:
        self._bus = bus
        self._llm_channel = llm_channel
        self._service_name = service_name
        self._service_version = service_version
        self._max_tokens = max_tokens
        self._timeout_sec = timeout_sec

    async def summarize(
        self,
        *,
        scope: str,
        kind: str,
        nodes: Set[str],
    ) -> tuple[str, str]:
        """
        Returns (summary_text, summary_kind).
        summary_kind is "llm" or "structural".
        Falls back to structural if LLM times out or bus unavailable.
        """
        try:
            return await asyncio.wait_for(
                self._llm_summarize(scope=scope, kind=kind, nodes=nodes),
                timeout=self._timeout_sec,
            )
        except Exception as exc:
            logger.warning("llm_summarize_failed scope=%s kind=%s reason=%s — using structural", scope, kind, exc)
        return self._structural_summary(scope=scope, kind=kind, nodes=nodes), "structural"

    async def _llm_summarize(self, *, scope: str, kind: str, nodes: Set[str]) -> tuple[str, str]:
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        sample_nodes = "\n".join(sorted(nodes)[:5])
        prompt = _LLM_PROMPT_TMPL.format(
            scope=scope,
            kind=kind,
            node_count=len(nodes),
            sample_nodes=sample_nodes,
            max_tokens=self._max_tokens,
        )
        corr_id = str(uuid.uuid4())
        reply_channel = f"orion:exec:result:LLMGatewayService:{corr_id}"
        payload = {
            "goal": prompt,
            "corr_id": corr_id,
            "reply_to": reply_channel,
            "max_tokens": self._max_tokens,
        }
        envelope = BaseEnvelope(
            kind="chat.request.v1",
            source=ServiceRef(name=self._service_name, version=self._service_version),
            payload=payload,
        )
        await self._bus.publish(self._llm_channel, envelope)
        reply = await self._bus.recv_one(reply_channel, timeout=self._timeout_sec)
        if reply is None:
            raise TimeoutError("llm_gateway_no_reply")
        text = (reply.payload or {}).get("text") or (reply.payload or {}).get("response") or ""
        if not text:
            raise ValueError("llm_gateway_empty_response")
        return text.strip()[:1000], "llm"

    def _structural_summary(self, *, scope: str, kind: str, nodes: Set[str]) -> str:
        sample = sorted(nodes)[:3]
        labels = [n.rsplit("/", 1)[-1].rsplit("#", 1)[-1] for n in sample]
        return (
            f"[structural] {scope} {kind} cluster: {len(nodes)} nodes. "
            f"Sample: {', '.join(labels)}."
        )
