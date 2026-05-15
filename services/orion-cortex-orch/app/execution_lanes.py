# services/orion-cortex-orch/app/execution_lanes.py
from __future__ import annotations

import logging
from dataclasses import dataclass

from orion.cognition.fast_chat_verbs import FAST_SINGLE_PASS_CHAT_VERBS
from orion.schemas.cortex.contracts import CortexClientRequest

logger = logging.getLogger("orion.cortex.orch.execution_lanes")

VALID_EXECUTION_LANES = frozenset({"chat", "spark", "background"})


@dataclass(frozen=True)
class ExecutionLaneDecision:
    lane: str
    reason: str
    explicit: bool
    requested: str | None = None


def resolve_execution_lane(req: CortexClientRequest) -> ExecutionLaneDecision:
    options = req.options if isinstance(req.options, dict) else {}
    raw_explicit = str(options.get("execution_lane") or "").strip().lower()
    if raw_explicit in VALID_EXECUTION_LANES:
        return ExecutionLaneDecision(
            lane=raw_explicit,
            reason="explicit_options",
            explicit=True,
            requested=raw_explicit,
        )
    if raw_explicit:
        logger.warning(
            "invalid_explicit_lane verb=%s mode=%s requested=%r",
            req.verb,
            req.mode,
            raw_explicit,
        )

    verb = str(req.verb or "").strip()
    mode = str(req.mode or "").strip().lower()

    if verb == "chat_general" or verb in FAST_SINGLE_PASS_CHAT_VERBS:
        return ExecutionLaneDecision(lane="chat", reason="verb_chat", explicit=False, requested=None)

    if verb == "introspect_spark":
        return ExecutionLaneDecision(lane="spark", reason="verb_spark", explicit=False, requested=None)

    if verb in {"dream_cycle", "log_orion_metacognition", "journal.compose"}:
        return ExecutionLaneDecision(lane="background", reason="verb_background", explicit=False, requested=None)

    if mode in {"quick", "brain"}:
        return ExecutionLaneDecision(lane="chat", reason="mode_chat", explicit=False, requested=None)

    if mode in {"agent", "council"}:
        return ExecutionLaneDecision(lane="background", reason="mode_background", explicit=False, requested=None)

    # Contract: new interactive verbs should set options.execution_lane or extend heuristics above;
    # otherwise auto/unknown traffic maps here (low priority pool).
    logger.debug(
        "execution_lane_fallback_background verb=%r mode=%r",
        verb,
        mode,
    )
    return ExecutionLaneDecision(lane="background", reason="fallback_background", explicit=False, requested=None)
