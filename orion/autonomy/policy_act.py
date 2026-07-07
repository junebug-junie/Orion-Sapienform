from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Sequence

from orion.autonomy.capability_policy import CapabilityEvaluationContext, evaluate_capability
from orion.autonomy.episode_fetch import EpisodeFetchRequest, execute_readonly_fetch
from orion.autonomy.models import ActionOutcomeRefV1, CapabilityDecisionV1
from orion.core.schemas.drives import DriveStateV1, GoalProposalV1
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1

logger = logging.getLogger(__name__)

_READONLY_CAPABILITY = "web.fetch.readonly"
_EPISODE_JOURNAL_CAPABILITY = "journal.compose.episode"
_GAP_SIGNAL = "world_coverage_gap"


def curiosity_strength_from_signals(signals: Sequence[FrontierInvocationSignalV1]) -> float:
    strengths = [float(sig.signal_strength or 0.0) for sig in signals if sig.signal_type == _GAP_SIGNAL]
    return max(strengths) if strengths else 0.0


def signal_kinds_from_curiosity(signals: Sequence[FrontierInvocationSignalV1]) -> list[str]:
    return sorted({str(sig.signal_type) for sig in signals if str(sig.signal_type or "").strip()})


def build_readonly_fetch_query(signals: Sequence[FrontierInvocationSignalV1]) -> str:
    for sig in signals:
        if sig.signal_type != _GAP_SIGNAL:
            continue
        for ref in sig.focal_node_refs:
            section = str(ref or "").strip()
            if section.startswith("section:"):
                label = section.split(":", 1)[1].replace("_", " ")
                return f"{label} recent news coverage"
    return "world coverage gap research"


async def maybe_execute_readonly_fetch_after_goal(
    *,
    goal: GoalProposalV1,
    drive_state: DriveStateV1,
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
    spawned_correlation_id: str | None,
    fetch_backend: Callable[..., Awaitable[dict]] | None = None,
    budget_used: dict[str, int] | None = None,
) -> tuple[CapabilityDecisionV1, ActionOutcomeRefV1 | None]:
    """Layer C gate + Tier B readonly fetch for substrate-fed motivation bus tick."""
    if not curiosity_signals or _GAP_SIGNAL not in signal_kinds_from_curiosity(curiosity_signals):
        decision = CapabilityDecisionV1(
            capability_id=_READONLY_CAPABILITY,
            outcome="denied",
            reason_code="missing_signal_kinds",
            auto_execute=False,
        )
        return decision, None

    if not spawned_correlation_id:
        decision = CapabilityDecisionV1(
            capability_id=_READONLY_CAPABILITY,
            outcome="denied",
            reason_code="missing_spawned_correlation_id",
            auto_execute=False,
        )
        return decision, None

    ctx = CapabilityEvaluationContext(
        predictive_pressure=float(drive_state.pressures.get("predictive", 0.0)),
        curiosity_strength=curiosity_strength_from_signals(curiosity_signals),
        signal_kinds=signal_kinds_from_curiosity(curiosity_signals),
        goal=goal,
        budget_used=budget_used or {},
    )
    decision = evaluate_capability(_READONLY_CAPABILITY, ctx)
    logger.info(
        "substrate_policy_act capability=%s outcome=%s reason=%s auto_execute=%s goal=%s spawned=%s",
        decision.capability_id,
        decision.outcome,
        decision.reason_code,
        decision.auto_execute,
        goal.artifact_id,
        spawned_correlation_id,
    )
    if decision.outcome != "allowed" or not decision.auto_execute:
        return decision, None

    query = build_readonly_fetch_query(curiosity_signals)
    req = EpisodeFetchRequest(
        subject=goal.subject,
        goal_artifact_id=goal.artifact_id,
        spawned_correlation_id=spawned_correlation_id,
        query=query,
    )
    if fetch_backend is None:
        outcome = await execute_readonly_fetch(req)
    else:
        outcome = await execute_readonly_fetch(req, fetch_backend=fetch_backend)
    if budget_used is not None:
        budget_used[_READONLY_CAPABILITY] = budget_used.get(_READONLY_CAPABILITY, 0) + 1
    return decision, outcome


async def maybe_compose_autonomy_episode_after_fetch(
    *,
    goal: GoalProposalV1,
    drive_state: DriveStateV1,
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
    spawned_correlation_id: str | None,
    fetch_outcome: ActionOutcomeRefV1 | None,
    journal_dispatch: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    budget_used: dict[str, int] | None = None,
) -> tuple[CapabilityDecisionV1, dict[str, Any] | None]:
    """Layer C gate + episode journal compose after successful readonly fetch."""
    del curiosity_signals
    if fetch_outcome is None:
        decision = CapabilityDecisionV1(
            capability_id=_EPISODE_JOURNAL_CAPABILITY,
            outcome="denied",
            reason_code="fetch_outcome_missing",
            auto_execute=False,
        )
        return decision, None

    if not spawned_correlation_id:
        decision = CapabilityDecisionV1(
            capability_id=_EPISODE_JOURNAL_CAPABILITY,
            outcome="denied",
            reason_code="missing_spawned_correlation_id",
            auto_execute=False,
        )
        return decision, None

    ctx = CapabilityEvaluationContext(
        predictive_pressure=float(drive_state.pressures.get("predictive", 0.0)),
        curiosity_strength=0.0,
        signal_kinds=[],
        goal=goal,
        budget_used=budget_used or {},
    )
    decision = evaluate_capability(_EPISODE_JOURNAL_CAPABILITY, ctx)
    logger.info(
        "substrate_policy_act capability=%s outcome=%s reason=%s auto_execute=%s goal=%s spawned=%s",
        decision.capability_id,
        decision.outcome,
        decision.reason_code,
        decision.auto_execute,
        goal.artifact_id,
        spawned_correlation_id,
    )
    if decision.outcome != "allowed" or not decision.auto_execute:
        return decision, None
    if journal_dispatch is None:
        return decision, None

    narrative_seed = (
        f"fetch outcome: {fetch_outcome.summary}"
        if fetch_outcome.success
        else f"fetch failed: {fetch_outcome.summary}"
    )
    result = await journal_dispatch(
        goal_artifact_id=goal.artifact_id,
        spawned_correlation_id=spawned_correlation_id,
        narrative_seed=narrative_seed,
    )
    if budget_used is not None:
        budget_used[_EPISODE_JOURNAL_CAPABILITY] = budget_used.get(_EPISODE_JOURNAL_CAPABILITY, 0) + 1
    return decision, result
