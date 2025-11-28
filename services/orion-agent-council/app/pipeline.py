# services/orion-agent-council/app/pipeline.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List
from uuid import uuid4

from orion.core.bus.service import OrionBus

from .settings import settings
from .models import DeliberationRequest, RoundResult, BlinkJudgement, AuditVerdict
from .llm_client import LLMClient
from .stages import Stage, AgentRoundStage, ArbiterStage, AuditorStage
from .council_policy import CouncilPolicy, CouncilDecision
from .publisher import CouncilPublisher

logger = logging.getLogger("agent-council.pipeline")


@dataclass
class DeliberationContext:
    bus: OrionBus
    llm: LLMClient
    req: DeliberationRequest
    trace_id: str

    round_index: int = 0

    round_result: Optional[RoundResult] = None
    judgement: Optional[BlinkJudgement] = None
    verdict: Optional[AuditVerdict] = None

    last_round: Optional[RoundResult] = None
    last_judgement: Optional[BlinkJudgement] = None

    stop: bool = False


class DeliberationPipeline:
    """
    Orchestrates the *shape* of deliberation:

      loop:
        stages (agents -> arbiter -> auditor)
        policy decides what to do
        publisher sends output

    It does NOT know about prompt strings or persona wiring.
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

    def run(self, req: DeliberationRequest) -> None:
        ctx = self.ctx

        # ensure trace_id
        if not req.trace_id:
            req.trace_id = ctx.trace_id

        while ctx.round_index < settings.max_rounds and not ctx.stop:
            # 1) Run the stage chain once: agents → arbiter → auditor
            for stage in self.stages:
                stage.run(ctx)
                if ctx.stop:
                    break

            if ctx.stop or ctx.verdict is None:
                break

            # 2) Ask policy what to do next
            decision = self.policy.decide(ctx)

            logger.info(
                "[%s] Council decision: %s (round=%d)",
                ctx.trace_id,
                decision,
                ctx.round_index,
            )

            if decision is CouncilDecision.ACCEPT:
                self.publisher.publish_final(ctx)
                return

            if decision is CouncilDecision.REVISE_SAME_ROUND:
                self.policy.prepare_revision(ctx)
                # re-run arbiter + auditor with same opinions
                for stage in self.stages[1:]:
                    stage.run(ctx)
                    if ctx.stop:
                        break
                self.publisher.publish_final(ctx)
                return

            if decision is CouncilDecision.NEW_ROUND:
                self.policy.prepare_new_round(ctx)
                ctx.round_index += 1
                continue

            # unknown → fallback to accept
            logger.warning(
                "[%s] Unknown decision=%s; treating as ACCEPT.",
                ctx.trace_id,
                decision,
            )
            self.publisher.publish_final(ctx)
            return

        # 3) Hit max rounds or stop flag
        logger.warning(
            "[%s] Max rounds (%d) reached or stop flag set; publishing best-effort.",
            ctx.trace_id,
            settings.max_rounds,
        )
        self.publisher.publish_best_effort(ctx)


def build_default_pipeline(bus: OrionBus, req: DeliberationRequest) -> DeliberationPipeline:
    """
    Factory used by DeliberationRouter.

    Later you can branch here by:
      - req.universe
      - req.tags
      - req.source
    to build different stage stacks / policies per universe.
    """
    trace_id = req.trace_id or str(uuid4())
    llm = LLMClient(bus)
    ctx = DeliberationContext(bus=bus, llm=llm, req=req, trace_id=trace_id)

    stages: List[Stage] = [
        AgentRoundStage(),
        ArbiterStage(),
        AuditorStage(),
    ]

    policy = CouncilPolicy()
    publisher = CouncilPublisher()

    return DeliberationPipeline(ctx=ctx, stages=stages, policy=policy, publisher=publisher)
