from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from orion.autonomy.capability_policy import CapabilityEvaluationContext, evaluate_capability
from orion.autonomy.episode_fetch import EpisodeFetchRequest, execute_readonly_fetch
from orion.autonomy.fetch_backend_resolve import resolve_fetch_backend
from orion.autonomy.salience import tokenize_terms
from orion.core.schemas.drives import GoalProposalV1
from orion.schemas.world_pulse import (
    CuriosityFindingV1,
    CuriosityFollowupV1,
    SectionCoverageV1,
)

logger = logging.getLogger("orion-world-pulse.curiosity")

_READONLY_CAPABILITY = "web.fetch.readonly"


def _synthetic_goal(run_id: str) -> GoalProposalV1:
    return GoalProposalV1.model_validate(
        {
            "artifact_id": f"world-pulse-gap-{run_id}",
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "memory.goals.proposed.v1",
            "goal_statement": "World-pulse coverage-gap fetch (synthetic goal for capability policy).",
            "proposal_signature": f"world-pulse-gap-{run_id}",
            "drive_origin": "predictive",
            "proposal_status": "proposed",
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
        }
    )


def _gate_open(run_id: str) -> bool:
    ctx = CapabilityEvaluationContext(
        predictive_pressure=1.0,
        curiosity_strength=1.0,
        signal_kinds=["world_coverage_gap"],
        goal=_synthetic_goal(run_id),
        budget_used={},
    )
    decision = evaluate_capability(_READONLY_CAPABILITY, ctx)
    logger.info(
        "world_pulse_curiosity_gate outcome=%s reason=%s auto_execute=%s run_id=%s",
        decision.outcome,
        decision.reason_code,
        decision.auto_execute,
        run_id,
    )
    return decision.outcome == "allowed" and bool(decision.auto_execute)


def build_curiosity_followups(
    *,
    run_id: str,
    section_coverage: dict[str, SectionCoverageV1],
    enabled: bool,
    dry_run: bool,
    max_articles_per_section: int,
    max_sections: int,
    fetch_backend: Callable[..., Awaitable[dict]] | None = None,
) -> list[CuriosityFollowupV1]:
    """Inline gap-fill fetch for under-covered sections.

    Returns [] unless enabled, non-dry, there is at least one under-covered
    section, and the web.fetch.readonly capability gate allows. Each section is
    fetched independently; a fetch error degrades to an empty followup and never
    raises (so it can never fail the world-pulse run).
    """
    if not enabled or dry_run:
        return []
    under_covered = [
        section
        for section, coverage in section_coverage.items()
        if coverage.status != "covered"
    ][: max(0, int(max_sections))]
    if not under_covered:
        return []
    if not _gate_open(run_id):
        return []

    backend = fetch_backend or resolve_fetch_backend()
    followups: list[CuriosityFollowupV1] = []
    for section in under_covered:
        label = section.replace("_", " ")
        query = f"{label} recent news coverage"
        gap_terms = tuple(sorted(tokenize_terms(label)))
        req = EpisodeFetchRequest(
            subject="orion",
            goal_artifact_id=f"world-pulse-gap-{run_id}-{section}",
            spawned_correlation_id=run_id,
            query=query,
            max_articles=max(1, int(max_articles_per_section)),
            gap_terms=gap_terms,
        )
        try:
            outcome = asyncio.run(execute_readonly_fetch(req, fetch_backend=backend))
        except Exception:
            logger.warning(
                "world_pulse_curiosity_fetch_failed section=%s run_id=%s",
                section,
                run_id,
                exc_info=True,
            )
            continue
        followups.append(
            CuriosityFollowupV1(
                section=section,
                driving_gap=section_coverage[section].status,
                query=query,
                articles=[
                    CuriosityFindingV1(
                        url=a.url,
                        title=a.title,
                        description=a.description,
                        salience=a.salience,
                    )
                    for a in outcome.articles
                ],
                action_id=outcome.action_id,
                correlation_id=run_id,
            )
        )
    logger.info(
        "world_pulse_curiosity_followups run_id=%s sections=%s articles=%s",
        run_id,
        len(followups),
        sum(len(f.articles) for f in followups),
    )
    return followups
