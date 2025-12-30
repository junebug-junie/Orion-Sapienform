# services/orion-agent-council/app/publisher.py
from __future__ import annotations

import logging
from typing import Any, Dict

from .models import CouncilResult
from .settings import settings

logger = logging.getLogger("agent-council.publisher")


class CouncilPublisher:
    """
    Handles final output formatting and publishing to the bus.
    Now fully async to match the Titanium bus chassis.
    """

    async def publish_final(self, ctx: Any) -> None:
        """
        Publishes the final result when decision is ACCEPT.
        """
        trace_id = ctx.trace_id
        
        resp_channel = f"{settings.channel_reply_prefix}:{trace_id}"

        verdict = ctx.verdict
        judgement = ctx.judgement
        round_res = ctx.round_result or ctx.last_round
        
        # [FIX] Populate all required CouncilResult fields
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

        logger.info(
            "[%s] CouncilPublisher: published council_result to %s (action=accept)",
            trace_id,
            resp_channel,
        )

        await ctx.bus.publish(resp_channel, result.model_dump(mode="json"))

    async def publish_best_effort(self, ctx: Any) -> None:
        """
        Publishes whatever we have if we hit max rounds or timeout.
        """
        trace_id = ctx.trace_id
        resp_channel = f"{settings.channel_reply_prefix}:{trace_id}"

        # Try to grab the last judgement or verdict
        judgement = ctx.last_judgement or ctx.judgement
        verdict = ctx.verdict
        round_res = ctx.round_result or ctx.last_round

        # [FIX] Populate all required CouncilResult fields
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

        logger.warning(
            "[%s] CouncilPublisher: published best_effort result to %s",
            trace_id,
            resp_channel,
        )

        await ctx.bus.publish(resp_channel, result.model_dump(mode="json"))
