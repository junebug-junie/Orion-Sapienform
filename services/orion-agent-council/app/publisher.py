# services/orion-agent-council/app/publisher.py
from __future__ import annotations

import logging

from .settings import settings
from .models import CouncilResult, BlinkScores, BlinkJudgement, AuditVerdict, RoundResult

logger = logging.getLogger("agent-council.publisher")


class CouncilPublisher:
    """
    Responsible for turning context into a bus message.
    """

    def _build_result_from_ctx(self, ctx: DeliberationContext) -> CouncilResult:
        req = ctx.req
        trace_id = ctx.trace_id

        round_result = ctx.round_result or ctx.last_round or RoundResult(round_index=0, opinions=[])
        judgement = ctx.judgement or ctx.last_judgement
        verdict = ctx.verdict

        if judgement is None:
            dummy_scores = BlinkScores()
            judgement = BlinkJudgement(
                proposed_answer="[AgentCouncil Warning] Unable to reach consensus.",
                scores=dummy_scores,
            )

        if verdict is None:
            verdict = AuditVerdict(
                action="accept",
                reason="missing_verdict_fallback",
                constraints={},
                override_answer=None,
            )

        final_text = verdict.override_answer or judgement.proposed_answer

        return CouncilResult(
            trace_id=trace_id,
            prompt=req.prompt,
            final_text=final_text,
            opinions=round_result.opinions,
            blink=judgement,
            verdict=verdict,
            meta={
                "source": req.source,
                "tags": req.tags or [],
                "universe": req.universe or "core",
                "agent_count": len(round_result.opinions),
            },
        )

    def publish_final(self, ctx: DeliberationContext) -> None:
        result = self._build_result_from_ctx(ctx)
        req = ctx.req
        resp_channel = req.response_channel or f"{settings.channel_reply_prefix}:{ctx.trace_id}"

        ctx.bus.publish(resp_channel, result.model_dump())
        logger.info(
            "[%s] CouncilPublisher: published council_result to %s (action=%s)",
            ctx.trace_id,
            resp_channel,
            result.verdict.action,
        )

    def publish_best_effort(self, ctx: DeliberationContext) -> None:
        """
        Used when the pipeline hits max rounds / stop without a clean verdict.
        """
        result = self._build_result_from_ctx(ctx)
        req = ctx.req
        resp_channel = req.response_channel or f"{settings.channel_reply_prefix}:{ctx.trace_id}"

        ctx.bus.publish(resp_channel, result.model_dump())
        logger.warning(
            "[%s] CouncilPublisher: published best-effort council_result to %s",
            ctx.trace_id,
            resp_channel,
        )
