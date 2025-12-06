# services/orion-agent-council/app/stages.py
from __future__ import annotations

import json
import logging
from typing import List

from .models import AgentOpinion, RoundResult, BlinkJudgement, BlinkScores
from .persona_factory import (
    get_agents_for_universe,
    get_chair_agent,
    get_auditor_agent,
)
from .council_prompts import (
    build_chair_prompt,
    build_auditor_prompt,
)

logger = logging.getLogger("agent-council.stages")


class Stage:
    def run(self, ctx: dict) -> None:  # pragma: no cover
        raise NotImplementedError


# ─────────────────────────────────────────────
# Agent round
# ─────────────────────────────────────────────

class AgentRoundStage(Stage):
    def run(self, ctx: dict) -> None:
        req = ctx.req
        agents = get_agents_for_universe(req.universe)
        persona_state_map = req.persona_state or {}
        phi = req.phi
        self_field = req.self_field

        opinions: List[AgentOpinion] = []

        for agent in agents:
            state_for_agent = persona_state_map.get(agent.name) if persona_state_map else None
            text = ctx.llm.generate(
                agent=agent,
                prompt=req.prompt,
                history=req.history or [],
                phi=phi,
                self_field=self_field,
                persona_state=state_for_agent,
                source=f"agent:{agent.name}",
            )
            opinions.append(
                AgentOpinion(
                    agent_name=agent.name,
                    text=text,
                )
            )

        result = RoundResult(round_index=ctx.round_index, opinions=opinions)
        ctx.round_result = result
        ctx.last_round = result

        logger.info(
            "[%s] AgentRoundStage: round=%d opinions=%d",
            ctx.trace_id,
            ctx.round_index,
            len(opinions),
        )


# ─────────────────────────────────────────────
# Arbiter (Chair / blink)
# ─────────────────────────────────────────────

class ArbiterStage(Stage):
    def run(self, ctx: dict) -> None:
        req = ctx.req
        round_result = ctx.round_result
        if round_result is None:
            logger.error("[%s] ArbiterStage: missing round_result", ctx.trace_id)
            ctx.stop = True
            return

        chair = get_chair_agent()
        prompt = build_chair_prompt(req, round_result)

        raw = ctx.llm.generate(
            agent=chair,
            prompt=prompt,
            history=None,
            phi=req.phi,
            self_field=req.self_field,
            persona_state=None,
            source="agent:Chair",
        )

        try:
            data = json.loads(raw)
            judgement = BlinkJudgement(**data)
        except Exception as e:
            logger.error(
                "[%s] ArbiterStage: JSON parse error, using fallback. error=%s raw=%r",
                ctx.trace_id,
                e,
                raw[:400],
            )
            fallback = BlinkScores()
            judgement = BlinkJudgement(
                proposed_answer=raw,
                scores=fallback,
                disagreement={"level": 0.5, "notes": "parse_error_fallback"},
                notes="Arbiter output was not valid JSON; raw used as answer.",
            )

        ctx.judgement = judgement
        ctx.last_judgement = judgement

        logger.info(
            "[%s] ArbiterStage: overall=%.3f risk=%.3f",
            ctx.trace_id,
            judgement.scores.overall,
            judgement.scores.risk,
        )


# ─────────────────────────────────────────────
# Auditor
# ─────────────────────────────────────────────

class AuditorStage(Stage):
    def run(self, ctx: dict) -> None:
        req = ctx.req
        round_result = ctx.round_result
        judgement = ctx.judgement

        if round_result is None or judgement is None:
            logger.error("[%s] AuditorStage: missing round_result/judgement", ctx.trace_id)
            ctx.stop = True
            return

        auditor = get_auditor_agent()
        prompt = build_auditor_prompt(req, round_result, judgement)

        raw = ctx.llm.generate(
            agent=auditor,
            prompt=prompt,
            history=None,
            phi=req.phi,
            self_field=req.self_field,
            persona_state=None,
            source="agent:Auditor",
        )

        from .models import AuditVerdict  # avoid circular

        try:
            verdict = AuditVerdict(**json.loads(raw))
        except Exception as e:
            logger.error(
                "[%s] AuditorStage: JSON parse error, defaulting to accept. error=%s raw=%r",
                ctx.trace_id,
                e,
                raw[:400],
            )
            verdict = AuditVerdict(
                action="accept",
                reason="parse_error_fallback",
                constraints={},
                override_answer=None,
            )

        ctx.verdict = verdict

        logger.info(
            "[%s] AuditorStage: action=%s reason=%s",
            ctx.trace_id,
            verdict.action,
            verdict.reason,
        )
