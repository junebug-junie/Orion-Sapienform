# services/orion-agent-council/app/pipeline.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync

from .settings import settings
from .models import DeliberationRequest, RoundResult, BlinkJudgement, AuditVerdict
from .llm_client import LLMClient
from .stages import Stage, AgentRoundStage, ArbiterStage, AuditorStage
from .council_policy import CouncilPolicy, CouncilDecision
from .publisher import CouncilPublisher

logger = logging.getLogger("agent-council.pipeline")


@dataclass
class DeliberationContext:
    bus: OrionBusAsync
    llm: LLMClient
    req: DeliberationRequest
    trace_id: str
    correlation_id: str | None = None
    reply_to: str | None = None

    round_index: int = 0
    round_result: Optional[RoundResult] = None
    judgement: Optional[BlinkJudgement] = None
    verdict: Optional[AuditVerdict] = None
    last_round: Optional[RoundResult] = None
    last_judgement: Optional[BlinkJudgement] = None
    stop: bool = False


class DeliberationPipeline:
    """
    Orchestrates the deliberation flow:
      loop:
        stages (agents -> arbiter -> auditor)
        policy decides (accept, revise, delegate)
        publisher sends final output
    """

    def __init__(
        self,
        ctx: DeliberationContext,
        stages: List[Stage],
        policy: CouncilPolicy,
        publisher: CouncilPublisher,
    ) -> None:
        self.ctx = ctx
        self.stages = stages
        self.policy = policy
        self.publisher = publisher

    async def run(self, req: DeliberationRequest) -> None:
        ctx = self.ctx
        if not req.trace_id:
            req.trace_id = ctx.trace_id
        if not req.response_channel:
            req.response_channel = ctx.reply_to

        while ctx.round_index < settings.max_rounds and not ctx.stop:
            # 1) Run the stage chain: agents → arbiter → auditor
            for stage in self.stages:
                await stage.run(ctx)  # Natively await async stages
                if ctx.stop:
                    break

            if ctx.stop or ctx.verdict is None:
                break

            # 2) Apply Council Policy to the result
            decision = self.policy.decide(ctx)

            logger.info(
                "[%s] Council decision: %s (round=%d)",
                ctx.trace_id,
                decision,
                ctx.round_index,
            )

            if decision is CouncilDecision.DELEGATE:
                # [PHASE 2.3] Return signal for Hub to hand off to Planner
                await self.publisher.publish_final(ctx, mode="delegate")
                return

            if decision is CouncilDecision.ACCEPT:
                await self.publisher.publish_final(ctx)
                return

            if decision is CouncilDecision.REVISE_SAME_ROUND:
                self.policy.prepare_revision(ctx)
                # Re-run arbiter + auditor with revised prompt constraints
                for stage in self.stages[1:]:
                    await stage.run(ctx)
                    if ctx.stop:
                        break
                await self.publisher.publish_final(ctx)
                return

            if decision is CouncilDecision.NEW_ROUND:
                self.policy.prepare_new_round(ctx)
                ctx.round_index += 1
                continue

            # Fallback
            await self.publisher.publish_final(ctx)
            return

        logger.warning("[%s] Max rounds reached; publishing best-effort.", ctx.trace_id)
        await self.publisher.publish_final(ctx)


def build_default_pipeline(
    bus: OrionBusAsync,
    req: DeliberationRequest,
    *,
    reply_to: str | None,
    correlation_id: str | None,
) -> DeliberationPipeline:
    trace_id = req.trace_id or str(uuid4())
    llm = LLMClient(bus)
    ctx = DeliberationContext(
        bus=bus,
        llm=llm,
        req=req,
        trace_id=trace_id,
        correlation_id=correlation_id,
        reply_to=reply_to,
    )

    stages: List[Stage] = [
        AgentRoundStage(),
        ArbiterStage(),
        AuditorStage(),
    ]

    return DeliberationPipeline(
        ctx=ctx,
        stages=stages,
        policy=CouncilPolicy(),
        publisher=CouncilPublisher()
    )
