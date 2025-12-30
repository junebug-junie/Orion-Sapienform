# services/orion-agent-council/app/council_policy.py
from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING

from .settings import settings

if TYPE_CHECKING:
    from .pipeline import DeliberationContext

logger = logging.getLogger("agent-council.policy")


class CouncilDecision(enum.Enum):
    ACCEPT = "accept"
    REVISE_SAME_ROUND = "revise_same_round"
    NEW_ROUND = "new_round"
    DELEGATE = "delegate"  # [PHASE 2.3] Added for Planner hand-off


class CouncilPolicy:
    """
    Encapsulates:
      - how we interpret auditor verdict + blink scores
      - how we rewrite prompts for revision/new_round
      - recognizes delegation signals from the Director
    """

    def decide(self, ctx: DeliberationContext) -> CouncilDecision:
        v = ctx.verdict
        j = ctx.judgement

        if v is None or j is None:
            logger.warning("[%s] CouncilPolicy.decide: missing verdict/judgement", ctx.trace_id)
            return CouncilDecision.ACCEPT

        # [PHASE 2.3] Check for delegation signal from Director/Auditor JSON
        # Look for 'delegate' in the action or decision fields
        if getattr(j, "decision", None) == "DELEGATE" or getattr(v, "action", None) == "delegate":
            logger.info("[%s] CouncilPolicy: Delegation requested", ctx.trace_id)
            return CouncilDecision.DELEGATE

        action = v.action or "accept"
        scores = j.scores
        disagreement = float((j.disagreement or {}).get("level", 0.0) or 0.0)

        # Soft override: if risk/disagreement is high, force a revision
        if action == "accept":
            if scores.risk >= settings.risk_threshold or disagreement >= settings.disagreement_threshold:
                logger.info(
                    "[%s] CouncilPolicy: overriding accept -> revise_same_round "
                    "(risk=%.3f disagreement=%.3f)",
                    ctx.trace_id,
                    scores.risk,
                    disagreement,
                )
                return CouncilDecision.REVISE_SAME_ROUND

        if action == "revise_same_round":
            return CouncilDecision.REVISE_SAME_ROUND

        if action == "new_round":
            return CouncilDecision.NEW_ROUND

        return CouncilDecision.ACCEPT

    def prepare_revision(self, ctx: DeliberationContext) -> None:
        """Adjusts prompt in-place using auditor constraints."""
        req = ctx.req
        v = ctx.verdict
        if v is None:
            return

        c = v.constraints or {}
        emphasize = c.get("emphasize") or []
        avoid = c.get("avoid") or []
        notes = c.get("notes") or ""

        lines = [req.prompt, "\n\n[Auditor constraints for the next pass]\n"]
        if emphasize:
            lines.append(f"Emphasize: {', '.join(emphasize)}")
        if avoid:
            lines.append(f"Avoid: {', '.join(avoid)}")
        if notes:
            lines.append(f"Notes: {notes}")

        req.prompt = "\n".join(lines)

    def prepare_new_round(self, ctx: DeliberationContext) -> None:
        """Rewrites prompt for a new round of agent thinking."""
        self.prepare_revision(ctx)
