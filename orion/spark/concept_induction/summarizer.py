from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, RecallDirective
from orion.core.bus.bus_schemas import LLMMessage

logger = logging.getLogger("orion.spark.concept.summarizer")


class Summarizer:
    """Optional LLM-based refinement via Cortex-Orch; otherwise heuristic."""

    def __init__(
        self,
        *,
        use_cortex: bool,
        verb_name: str,
        bus: OrionBusAsync,
        request_channel: str,
        result_prefix: str,
        timeout_sec: float = 12.0,
        service_ref: ServiceRef,
    ) -> None:
        self.use_cortex = use_cortex
        self.verb_name = verb_name
        self.bus = bus
        self.request_channel = request_channel
        self.result_prefix = result_prefix
        self.timeout_sec = timeout_sec
        self.service_ref = service_ref

    async def summarize(
        self,
        *,
        subject: str,
        candidates: List[str],
        clusters: List[List[str]],
        evidence: List[str],
    ) -> Dict[str, Any]:
        if not self.use_cortex:
            return {
                "summary": f"{len(candidates)} concepts for {subject}",
                "cluster_labels": [", ".join(c[:3]) for c in clusters],
                "confidence_notes": "heuristic",
            }
        try:
            await self.bus.connect()
            corr = uuid4()
            reply = f"{self.result_prefix}:{corr}"

            context = CortexClientContext(
                messages=[
                    LLMMessage(role="system", content="You are Orion Spark Concept Induction."),
                    LLMMessage(role="user", content=f"Subject: {subject}"),
                    LLMMessage(role="user", content=f"Candidates: {json.dumps(candidates[:50])}"),
                    LLMMessage(role="user", content=f"Clusters: {json.dumps(clusters)}"),
                    LLMMessage(role="user", content=f"Evidence ids: {json.dumps(evidence[:20])}"),
                ]
            )
            req = CortexClientRequest(
                mode="brain",
                verb=self.verb_name,
                packs=[],
                options={"diagnostic": False},
                recall=RecallDirective(enabled=False),
                context=context,
            )
            env = BaseEnvelope(
                kind="cortex.orch.request",
                correlation_id=corr,
                reply_to=reply,
                source=self.service_ref,
                payload=req.model_dump(mode="json"),
            )
            msg = await self.bus.rpc_request(
                self.request_channel,
                env,
                reply_channel=reply,
                timeout_sec=self.timeout_sec,
            )
            decoded = self.bus.codec.decode(msg.get("data"))
            if not decoded.ok:
                raise RuntimeError(f"decode_failed: {decoded.error}")
            payload = decoded.envelope.payload
            if isinstance(payload, dict):
                return payload.get("final_text") or payload
            return {"summary": str(payload)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cortex-Orch summarize failed: %s", exc)
            return {
                "summary": f"{len(candidates)} concepts for {subject}",
                "cluster_labels": [", ".join(c[:3]) for c in clusters],
                "confidence_notes": f"orch_failed:{exc}",
            }
