from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Sequence

from orion.autonomy.capability_policy import CapabilityEvaluationContext, evaluate_capability
from orion.autonomy.episode_fetch import EpisodeFetchRequest, execute_readonly_fetch
from orion.autonomy.fetch_backend_resolve import resolve_fetch_backend
from orion.autonomy.models import ActionOutcomeRefV1, CapabilityDecisionV1, SubstrateActResultV1, SubstrateEpisodeIntentV1
from orion.autonomy.salience import gap_terms_from_signals, iter_gap_section_labels
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
    label = next(iter_gap_section_labels(signals), None)
    if label:
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
    gap_terms = gap_terms_from_signals(curiosity_signals, fallback_query=query)
    req = EpisodeFetchRequest(
        subject=goal.subject,
        goal_artifact_id=goal.artifact_id,
        spawned_correlation_id=spawned_correlation_id,
        query=query,
        gap_terms=tuple(sorted(gap_terms)),
    )
    if fetch_backend is None:
        fetch_backend = resolve_fetch_backend()
    outcome = await execute_readonly_fetch(req, fetch_backend=fetch_backend)
    if budget_used is not None:
        budget_used[_READONLY_CAPABILITY] = budget_used.get(_READONLY_CAPABILITY, 0) + 1
    return decision, outcome


_MAX_SEED_DESC_CHARS = 300


def _gap_section_label(signals: Sequence[FrontierInvocationSignalV1]) -> str:
    return next(iter_gap_section_labels(signals), "")


def build_episode_narrative_seed(
    goal: GoalProposalV1,
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
    fetch_outcome: ActionOutcomeRefV1,
) -> str:
    """Structured multi-line compose seed: why + what + salience + satiation ask.

    `goal` is part of the stable interface (spec contract + future goal_statement
    enrichment seam) even though the current body does not read it.
    """
    del goal  # part of stable interface; not read yet
    if not fetch_outcome.success:
        return f"fetch failed: {fetch_outcome.summary}"

    lines: list[str] = []
    strength = curiosity_strength_from_signals(curiosity_signals)
    section = _gap_section_label(curiosity_signals)
    if section:
        lines.append(f'Why: predictive coverage gap in "{section}" (strength {strength:.2f}).')
    else:
        lines.append(f"Why: predictive coverage gap (strength {strength:.2f}).")
    if fetch_outcome.query:
        lines.append(f'Query: "{fetch_outcome.query}"')

    articles = fetch_outcome.articles
    if articles:
        lines.append(f"Fetched {len(articles)} article(s):")
        # "scored" iff the fetch had gap terms to score against (mirrors the
        # gap_terms the fetch used). A genuine 0.0 overlap is honestly "salience
        # 0.00", not "unscored"; "unscored" means there was nothing to score by.
        scored = bool(
            gap_terms_from_signals(curiosity_signals, fallback_query=fetch_outcome.query or "")
        )
        for idx, art in enumerate(articles, start=1):
            marker = f"salience {art.salience:.2f}" if scored else "unscored"
            title = art.title or "(untitled)"
            lines.append(f"  {idx}. [{marker}] {title} — {art.url}")
            desc = (art.description or "").strip()
            if desc:
                if len(desc) > _MAX_SEED_DESC_CHARS:
                    desc = desc[:_MAX_SEED_DESC_CHARS].rstrip() + "…"
                lines.append(f"     {desc}")
    else:
        lines.append(f"fetch outcome: {fetch_outcome.summary}")

    lines.append(
        "Reflect: summarize each article and assess whether it closes the gap that "
        "drove this fetch. Name what is still missing. Do not invent sources."
    )
    return "\n".join(lines)


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

    narrative_seed = build_episode_narrative_seed(goal, curiosity_signals, fetch_outcome)
    result = await journal_dispatch(
        goal_artifact_id=goal.artifact_id,
        spawned_correlation_id=spawned_correlation_id,
        narrative_seed=narrative_seed,
    )
    if budget_used is not None:
        budget_used[_EPISODE_JOURNAL_CAPABILITY] = budget_used.get(_EPISODE_JOURNAL_CAPABILITY, 0) + 1
    return decision, result


def resolve_episode_intent(
    *,
    store,
    subject: str,
    run_id: str,
    drive_origin: str = "predictive",
) -> SubstrateEpisodeIntentV1:
    slot = store.load_goal_slot(subject, drive_origin)
    artifact_id = slot.get("artifact_id") if isinstance(slot, dict) else None
    if isinstance(artifact_id, str) and artifact_id.strip():
        return SubstrateEpisodeIntentV1(
            goal_artifact_id=artifact_id.strip(),
            drive_origin=drive_origin,
            spawned_correlation_id=run_id,
            subject=subject,
        )
    return SubstrateEpisodeIntentV1(
        goal_artifact_id=f"episode-{run_id}",
        drive_origin="predictive",
        spawned_correlation_id=run_id,
        subject=subject,
    )


def goal_proposal_from_episode_intent(intent: SubstrateEpisodeIntentV1) -> GoalProposalV1:
    return GoalProposalV1.model_validate(
        {
            "artifact_id": intent.goal_artifact_id,
            "subject": intent.subject,
            "model_layer": "self-model",
            "entity_id": f"self:{intent.subject}",
            "kind": "memory.goals.proposed.v1",
            "goal_statement": "Substrate episode intent (synthetic goal for policy).",
            "proposal_signature": f"episode-{intent.spawned_correlation_id}",
            "drive_origin": intent.drive_origin,
            "proposal_status": "proposed",
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
        }
    )


async def maybe_execute_substrate_act_after_metabolism(
    *,
    episode_intent: SubstrateEpisodeIntentV1,
    drive_state: DriveStateV1,
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
    spawned_correlation_id: str | None = None,
    fetch_backend: Callable[..., Awaitable[dict]] | None = None,
    journal_dispatch: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    budget_used: dict[str, int] | None = None,
    prefetched_outcome: ActionOutcomeRefV1 | None = None,
    episode_journal_enabled: bool = False,
) -> SubstrateActResultV1:
    run_id = spawned_correlation_id or episode_intent.spawned_correlation_id
    synthetic_goal = goal_proposal_from_episode_intent(episode_intent)
    result = SubstrateActResultV1()

    if prefetched_outcome is not None:
        # Single shared fetch: world-pulse already fetched this gap section and put
        # the findings on the run result. Reuse them; do not call the backend again.
        fetch_outcome = prefetched_outcome
        result = result.model_copy(
            update={
                "fetch_attempted": True,
                "fetch_outcome_id": fetch_outcome.action_id,
                "fetch_outcome": fetch_outcome,
            }
        )
    else:
        fetch_decision, fetch_outcome = await maybe_execute_readonly_fetch_after_goal(
            goal=synthetic_goal,
            drive_state=drive_state,
            curiosity_signals=curiosity_signals,
            spawned_correlation_id=run_id,
            fetch_backend=fetch_backend,
            budget_used=budget_used,
        )
        if fetch_decision.outcome == "allowed" and fetch_outcome is not None:
            result = result.model_copy(
                update={
                    "fetch_attempted": True,
                    "fetch_outcome_id": fetch_outcome.action_id,
                    "fetch_outcome": fetch_outcome,
                }
            )

    if not episode_journal_enabled or fetch_outcome is None:
        return result

    # The journal compose step issues an RPC (cortex-exec) that can time out. A journal
    # failure must NOT discard an already-successful fetch outcome: isolate it so the
    # caller still receives `result` (with fetch_outcome) and can persist the fetch.
    try:
        journal_decision, journal_payload = await maybe_compose_autonomy_episode_after_fetch(
            goal=synthetic_goal,
            drive_state=drive_state,
            curiosity_signals=curiosity_signals,
            spawned_correlation_id=run_id,
            fetch_outcome=fetch_outcome,
            journal_dispatch=journal_dispatch,
            budget_used=budget_used,
        )
    except Exception:
        logger.warning(
            "substrate_episode_journal_failed goal=%s spawned=%s",
            synthetic_goal.artifact_id,
            run_id,
            exc_info=True,
        )
        return result
    if journal_decision.outcome == "allowed" and journal_payload is not None:
        entry_id = None
        if isinstance(journal_payload.get("write"), dict):
            entry_id = journal_payload["write"].get("entry_id")
        result = result.model_copy(update={"journal_attempted": True, "journal_entry_id": entry_id})
    return result
