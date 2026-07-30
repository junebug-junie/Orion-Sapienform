from __future__ import annotations

import asyncio
import logging
import threading
from typing import Awaitable, Callable

from orion.autonomy.capability_policy import CapabilityEvaluationContext, evaluate_capability
from orion.autonomy.episode_fetch import EpisodeFetchRequest, SurpriseSource, execute_readonly_fetch
from orion.autonomy.fetch_backend_resolve import resolve_fetch_backend
from orion.autonomy.salience import tokenize_terms
from orion.core.schemas.drives import GoalProposalV1
from orion.schemas.world_pulse import (
    CuriosityFindingV1,
    CuriosityFollowupV1,
    SectionCoverageV1,
)
from orion.substrate.bus_synaptic_surprise import latest_bus_synaptic_prediction_error

logger = logging.getLogger("orion-world-pulse.curiosity")

_READONLY_CAPABILITY = "web.fetch.readonly"

# Lazy, cached Postgres engine for the real ambient bus_synaptic surprise signal -- only
# built if ORION_ACTION_OUTCOME_DB_URL is configured (see app/settings.py). This module
# has no other Postgres dependency, so an unconfigured URL keeps
# _default_bus_synaptic_surprise_source() inert (always None), matching this codebase's
# established fail-open convention (orion/spark/concept_induction/bus_worker.py's
# ConceptWorker._get_surprise_engine()) -- but unlike that class's own per-instance
# attribute, this is a genuine process-wide module global: FastAPI runs sync route
# handlers in a threadpool (services/orion-world-pulse/app/routers/runs.py's
# world_pulse_run), so concurrent requests can reach this check-then-set from different
# threads of the same process. Guarded with a lock (found in review) so two concurrent
# first-callers can't both build and leak a real Postgres engine.
_surprise_engine: object | None = None
_surprise_engine_lock = threading.Lock()


def _get_surprise_engine():
    from app.settings import settings

    if not settings.action_outcome_db_url:
        return None
    global _surprise_engine
    if _surprise_engine is None:
        with _surprise_engine_lock:
            if _surprise_engine is None:
                from sqlalchemy import create_engine

                _surprise_engine = create_engine(settings.action_outcome_db_url, pool_pre_ping=True)
    return _surprise_engine


def _default_bus_synaptic_surprise_source() -> float | None:
    """Real `surprise_source` used automatically when a caller doesn't supply one --
    see `build_curiosity_followups`/`_gate_open` below. Fail-open: any exception here
    (DB down, engine unbuildable) degrades to `None`, never raises into a real
    world-pulse run.
    """
    engine = _get_surprise_engine()
    if engine is None:
        return None
    try:
        return latest_bus_synaptic_prediction_error(engine)
    except Exception:
        logger.warning("world_pulse_bus_synaptic_surprise_fetch_failed", exc_info=True)
        return None


def _synthetic_goal(run_id: str) -> GoalProposalV1:
    """Never published to the bus -- built solely to satisfy
    `evaluate_capability`'s `requires_goal`/`proposal_status` checks (see
    `orion/autonomy/capability_policy.py`). Still needed post-2026-07-30
    (chore/delete-orion-drives): the gate that used to also match
    `ctx.goal.drive_origin` against `rule.required_drive_origins` was removed
    in that sprint's Wave 2a (drive_origin is no longer read by any gate), but
    `ctx.goal` itself remains required for the existence/proposal_status
    checks below it. `drive_origin` stays set below only because it is still
    a required field on the (kept, write-never) `GoalProposalV1` schema --
    its value is inert now, read by nothing.
    """
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


def _resolve_domain_surprise(surprise_source: SurpriseSource | None) -> float | None:
    """Same real ambient bus_synaptic value orion/autonomy/policy_act.py's identically-named
    helper reads. `None` (the default -- no explicit surprise_source passed by the caller)
    falls back to `_default_bus_synaptic_surprise_source`, this module's own real reader --
    closing the disclosed gap this service had (no Postgres access at all) as of
    2026-07-28. Still resolves to `None` in practice unless `ORION_ACTION_OUTCOME_DB_URL`
    is configured for this service.
    """
    real_source = surprise_source or _default_bus_synaptic_surprise_source
    try:
        return real_source()
    except Exception:
        return None


def _gate_open(run_id: str, *, surprise_source: SurpriseSource | None = None) -> bool:
    _domain_surprise = _resolve_domain_surprise(surprise_source)
    ctx = CapabilityEvaluationContext(
        # predictive_pressure removed 2026-07-30 (chore/delete-orion-drives
        # Wave 2a): DriveStateV1.pressures["predictive"], its source, no
        # longer exists (DriveEngine deleted in Wave 1). curiosity_strength
        # is capability_policy.py's real, field-native replacement gate.
        curiosity_strength=1.0,
        signal_kinds=["world_coverage_gap"],
        goal=_synthetic_goal(run_id),
        budget_used={},
        domain_surprise_score=_domain_surprise,
        domain_surprise_source="bus_synaptic" if _domain_surprise is not None else None,
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
    surprise_source: SurpriseSource | None = None,
) -> list[CuriosityFollowupV1]:
    """Inline gap-fill fetch for under-covered sections.

    Returns [] unless enabled, non-dry, there is at least one under-covered
    section, and the web.fetch.readonly capability gate allows. Each section is
    fetched independently; a fetch or mapping error degrades gracefully (empty
    or skipped followup) and never raises (so it can never fail the run).
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
    try:
        if not _gate_open(run_id, surprise_source=surprise_source):
            return []
    except Exception:
        # A capability-policy load/eval failure (e.g. missing policy config in a
        # stripped-down image) must never fail the world-pulse run: degrade to
        # no followups.
        logger.warning(
            "world_pulse_curiosity_gate_error run_id=%s", run_id, exc_info=True
        )
        return []

    backend = fetch_backend or resolve_fetch_backend()
    followups: list[CuriosityFollowupV1] = []
    # Cost is bounded by max_sections (one fetch per under-covered section, up to
    # max_sections). The capability policy's budget_per_cycle is an on/off gate
    # here, not a per-run fetch-count limit; that bound is max_sections by design.
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
        # The fetch AND the mapping into schema models are both guarded: any
        # failure degrades to skipping this section, never failing the run.
        try:
            outcome = asyncio.run(
                execute_readonly_fetch(
                    req,
                    fetch_backend=backend,
                    surprise_source=surprise_source or _default_bus_synaptic_surprise_source,
                )
            )
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
        except Exception:
            logger.warning(
                "world_pulse_curiosity_fetch_failed section=%s run_id=%s",
                section,
                run_id,
                exc_info=True,
            )
            continue
    logger.info(
        "world_pulse_curiosity_followups run_id=%s sections=%s articles=%s",
        run_id,
        len(followups),
        sum(len(f.articles) for f in followups),
    )
    return followups
