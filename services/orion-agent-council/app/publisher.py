# services/orion-agent-council/app/publisher.py
from __future__ import annotations

import logging
from typing import Any, Dict

from .models import CouncilResult, BlinkJudgement, BlinkScores, AuditVerdict
from .settings import settings
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

logger = logging.getLogger("agent-council.publisher")


class CouncilPublisher:
    """
    Handles final output formatting and publishing to the bus.
    Now fully async to match the Titanium bus chassis.
    """

    async def _publish(self, ctx: Any, result: CouncilResult, *, tag: str) -> None:
        resp_channel = ctx.req.response_channel or ctx.reply_to or f"{settings.channel_reply_prefix}:{ctx.trace_id}"
        corr = ctx.correlation_id or ctx.trace_id

        env = BaseEnvelope(
            kind="council.result",
            source=ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name),
            correlation_id=corr,
            payload=result.model_dump(mode="json"),
            reply_to=None,
        )

        logger.info("[%s] CouncilPublisher: published (%s) to %s", ctx.trace_id, tag, resp_channel)
        await ctx.bus.publish(resp_channel, env)

    async def publish_final(self, ctx: Any) -> None:
        """
        Publishes the final result when decision is ACCEPT.
        """
        trace_id = ctx.trace_id

        verdict = ctx.verdict
        judgement = ctx.judgement
        round_res = ctx.round_result or ctx.last_round
        
        # Populate all required CouncilResult fields
        result = CouncilResult(
            trace_id=trace_id,
            prompt=ctx.req.prompt,  # <--- Added
            final_text=judgement.proposed_answer if judgement else "", # <--- Added (mapped to final_text)
            opinions=round_res.opinions if round_res else [], # <--- Added
            blink=judgement, # <--- Added
            verdict=verdict,
            meta={
                "decision": "accept",
                "history_used": len(ctx.req.history or []),
                "rounds_used": ctx.round_index + 1,
            }
        )

        await self._publish(ctx, result, tag="accept")

    async def publish_best_effort(self, ctx: Any) -> None:
        """
        Publishes whatever we have if we hit max rounds or timeout.
        """
        trace_id = ctx.trace_id

        # Try to grab the last judgement or verdict
        judgement = ctx.last_judgement or ctx.judgement
        verdict = ctx.verdict
        round_res = ctx.round_result or ctx.last_round

        # Build safe fallbacks to satisfy CouncilResult schema
        if judgement is None:
            judgement = BlinkJudgement(
                proposed_answer="[No response from council]",
                scores=BlinkScores(),
                disagreement={"level": 1.0, "notes": "timeout_or_error"},
                notes="Blink judgement missing; generated fallback.",
            )

        if verdict is None:
            verdict = AuditVerdict(
                action="accept",
                reason="timeout_or_error",
                constraints={},
                override_answer=None,
            )

        # Populate all required CouncilResult fields
        result = CouncilResult(
            trace_id=trace_id,
            prompt=ctx.req.prompt,
            final_text=judgement.proposed_answer if judgement else "[No consensus reached]",
            opinions=round_res.opinions if round_res else [],
            blink=judgement,
            verdict=verdict,
            meta={
                "decision": "best_effort",
                "history_used": len(ctx.req.history or []),
                "rounds_used": ctx.round_index + 1,
                "note": "Max rounds reached or stopped"
            }
        )

        logger.warning("[%s] CouncilPublisher: best_effort", trace_id)
        await self._publish(ctx, result, tag="best_effort")
