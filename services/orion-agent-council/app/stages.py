# services/orion-agent-council/app/stages.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional

from .models import AgentOpinion, RoundResult, BlinkJudgement, BlinkScores
from .llm_client import LLMClientTimeout
from .settings import settings
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


def _extract_json(raw: str) -> Optional[str]:
    """
    Extract the first JSON object from a raw string that may contain preamble text.
    """
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return raw[start : end + 1]


class Stage:
    async def run(self, ctx: dict) -> None:  # pragma: no cover
        raise NotImplementedError


async def _llm_with_retry(
    ctx,
    *,
    agent,
    prompt: str,
    source: str,
    history: Optional[list] = None,
    persona_state: Optional[dict] = None,
) -> Optional[str]:
    """
    Shared retry helper.
    Returns text on success, or None on timeout/exhaustion.
    """
    attempts_left = max(1, settings.council_llm_max_attempts)
    per_call_timeout = settings.council_llm_timeout_sec

    async def _call() -> str:
        return await ctx.llm.generate(
            agent=agent,
            prompt=prompt,
            history=history or [],
            phi=ctx.req.phi,
            self_field=ctx.req.self_field,
            persona_state=persona_state,
            source=source,
            timeout_sec=per_call_timeout,
        )

    while attempts_left > 0:
        try:
            return await _call()
        except LLMClientTimeout as exc:
            attempts_left -= 1
            if attempts_left <= 0:
                logger.error(
                    "[%s] %s: timeout (%s)",
                    ctx.trace_id,
                    source,
                    exc,
                )
                return None
            logger.warning(
                "[%s] %s: retrying attempts_left=%d",
                ctx.trace_id,
                source,
                attempts_left,
            )

    return None


# ─────────────────────────────────────────────
# Agent round (parallelized, no hard round cutoff)
# ─────────────────────────────────────────────

class AgentRoundStage(Stage):
    async def run(self, ctx: dict) -> None:
        req = ctx.req
        agents = get_agents_for_universe(req.universe)
        persona_state_map = req.persona_state or {}

        async def _run_agent(agent) -> AgentOpinion:
            state_for_agent = persona_state_map.get(agent.name) if persona_state_map else None
            text = await _llm_with_retry(
                ctx,
                agent=agent,
                prompt=req.prompt,
                source=f"agent:{agent.name}",
                history=req.history or [],
                persona_state=state_for_agent,
            )
            if text is None:
                text = f"[LLM timeout for {agent.name}]"
            return AgentOpinion(agent_name=agent.name, text=text)

        tasks = [asyncio.create_task(_run_agent(agent)) for agent in agents]
        opinions_results = await asyncio.gather(*tasks, return_exceptions=True)

        opinions: List[AgentOpinion] = []
        for agent, result in zip(agents, opinions_results):
            if isinstance(result, AgentOpinion):
                opinions.append(result)
            else:
                logger.error(
                    "[%s] agent:%s: failure (%s)",
                    ctx.trace_id,
                    agent.name,
                    result,
                )
                opinions.append(
                    AgentOpinion(
                        agent_name=agent.name,
                        text=f"[LLM failure for {agent.name}]",
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
    async def run(self, ctx: dict) -> None:
        req = ctx.req
        round_result = ctx.round_result
        if round_result is None:
            logger.error("[%s] ArbiterStage: missing round_result", ctx.trace_id)
            ctx.stop = True
            return

        chair = get_chair_agent()
        prompt = build_chair_prompt(req, round_result)

        raw = await _llm_with_retry(
            ctx=ctx,
            agent=chair,
            prompt=prompt,
            source="agent:Chair",
        )
        if raw is None:
            logger.error(
                "[%s] ArbiterStage: timeout/exhaustion; using fallback judgement",
                ctx.trace_id,
            )
            fallback_scores = BlinkScores()
            judgement = BlinkJudgement(
                proposed_answer="[No arbiter response due to timeout]",
                scores=fallback_scores,
                disagreement={"level": 1.0, "notes": "timeout_or_error"},
                notes="Arbiter LLM timed out; generated fallback judgement.",
            )
            ctx.judgement = judgement
            ctx.last_judgement = judgement
            return

        try:
            candidate = _extract_json(raw) or raw
            data = json.loads(candidate)
            judgement = BlinkJudgement(**data)
        except Exception as e:
            logger.error(
                "[%s] ArbiterStage: JSON parse error, using fallback. error=%s raw=%r",
                ctx.trace_id,
                e,
                raw[:400],
            )
            fallback_scores = BlinkScores()
            judgement = BlinkJudgement(
                proposed_answer=raw,
                scores=fallback_scores,
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
# ────────────────────────────────────────────

class AuditorStage(Stage):
    async def run(self, ctx: dict) -> None:
        req = ctx.req
        round_result = ctx.round_result
        judgement = ctx.judgement

        if round_result is None or judgement is None:
            logger.error("[%s] AuditorStage: missing round_result/judgement", ctx.trace_id)
            ctx.stop = True
            return

        auditor = get_auditor_agent()
        prompt = build_auditor_prompt(req, round_result, judgement)

        raw = await _llm_with_retry(
            ctx=ctx,
            agent=auditor,
            prompt=prompt,
            source="agent:Auditor",
        )
        if raw is None:
            logger.error(
                "[%s] AuditorStage: timeout/exhaustion; using fallback verdict",
                ctx.trace_id,
            )
            from .models import AuditVerdict  # avoid circular

            verdict = AuditVerdict(
                action="accept",
                reason="timeout_or_error",
                constraints={},
                override_answer=None,
            )
            ctx.verdict = verdict
            return

        from .models import AuditVerdict  # avoid circular

        try:
            candidate = _extract_json(raw) or raw
            verdict = AuditVerdict(**json.loads(candidate))
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
