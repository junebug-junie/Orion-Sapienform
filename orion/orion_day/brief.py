"""What Hub calls on Prime's order (living Prime Part 3): gather -> budget -> brief -> durable request.

    brief = await build_orion_day_brief(conn, letter_date)       # raises OrionDayEmptyError on a blank day
    request = build_orion_day_request(brief, run_id=..., correlation_id=..., deadline_at=...)
    # submit: POST {DURABLE_RUNS_URL}/runs with request.model_dump(mode="json"). The run writes the letter
    # as one unified turn (orion/orion_day/letter_turn.py); the row lands in orion_day_letter.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from orion.orion_day.day_input import DAY_INPUT_BUDGET_TOKENS as DEFAULT_BUDGET_TOKENS, render_day_input
from orion.orion_day.gather import DEFAULT_GITHUB_REPO, gather_orion_day
from orion.world_pulse_read.findings import PriorLookup
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.orion_day import (
    ORION_DAY_LLM_ROUTE,
    ORION_DAY_TIMEZONE,
    ORION_DAY_WORKFLOW,
    MissedLetterV1,
    OrionDayMaterialV1,
    OrionDayRunBriefV1,
    orion_day_run_id,
)
from orion.schemas.resource_admission import ResourceRequirementV1


# What one letter turn needs besides the record: the letter and its threads (the retired note verb's
# 12,000-token completion budget plus room for the threads list), the instructions, and the unified
# turn's own harness prefix (situation, stance, Prime block, tool notes) and a little tool output.
LETTER_COMPLETION_TOKENS = 13_000
PROMPT_FRAME_TOKENS = 2_000
HARNESS_PREFIX_TOKENS = 12_000


def minimum_context_tokens(brief: OrionDayRunBriefV1) -> int:
    """The context a card needs for this run's turn. Sent as the admission's
    ``requirements.minimum_context_tokens`` (the pool only grants a card whose live
    ctx_per_slot covers it): a heavy day must not be placed on the 65,536-token chat card
    (live /props 2026-09-30) when the pool spills agent work there; a light day still may."""
    return (brief.llm_view.approx_tokens + PROMPT_FRAME_TOKENS + HARNESS_PREFIX_TOKENS
            + LETTER_COMPLETION_TOKENS)


class OrionDayEmptyError(RuntimeError):
    """The day holds nothing to write about. No letter is better than an empty-shell one."""


def brief_from_material(
    material: OrionDayMaterialV1,
    *,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    timeout_sec: float = 1800.0,
    prime_act_id: str | None = None,
    missed_letters: list[MissedLetterV1] | None = None,
) -> OrionDayRunBriefV1:
    if material.is_empty():
        raise OrionDayEmptyError(f"orion_day_empty:{material.letter_date}")
    return OrionDayRunBriefV1(
        letter_date=material.letter_date,
        timezone=material.timezone,
        window_start=material.window_start,
        window_end=material.window_end,
        material=material,
        # The honest input (living Prime design #3224, constraint 7): Juniper protected, reveries
        # by theme, self-sense once each, readings and news as a trimmed tail.
        llm_view=render_day_input(material, budget_tokens=budget_tokens),
        llm_route=ORION_DAY_LLM_ROUTE,
        timeout_sec=timeout_sec,
        prime_act_id=prime_act_id,
        missed_letters=list(missed_letters or []),
    )


async def build_orion_day_brief(
    conn: Any,
    letter_date: date,
    *,
    now: datetime | None = None,
    github_repo: str = DEFAULT_GITHUB_REPO,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    timeout_sec: float = 1800.0,
    temporal_day: bool = True,
    prior_lookup: PriorLookup | None = None,
    prime_act_id: str | None = None,
    missed_letters: list[MissedLetterV1] | None = None,
) -> OrionDayRunBriefV1:
    material = await gather_orion_day(conn, letter_date, now=now, github_repo=github_repo,
                                      temporal_day=temporal_day, prior_lookup=prior_lookup)
    return brief_from_material(material, budget_tokens=budget_tokens, timeout_sec=timeout_sec,
                               prime_act_id=prime_act_id, missed_letters=missed_letters)


def default_deadline(letter_date: date, *, tz_name: str = ORION_DAY_TIMEZONE) -> datetime:
    """End of the day AFTER the letter's day (local): the run may queue through the 06:00
    congestion and most of the day before it gives up."""
    local_end = datetime.combine(letter_date + timedelta(days=2), time.min, tzinfo=ZoneInfo(tz_name))
    return local_end.astimezone(timezone.utc)


def build_orion_day_request(
    brief: OrionDayRunBriefV1,
    *,
    attempt: int = 1,
    run_id: str | None = None,
    deadline_at: datetime | None = None,
    correlation_id: str | None = None,
    requested_at: datetime | None = None,
) -> DurableRunRequestV1:
    """``run_id`` (Prime's order) wins over ``attempt``."""
    run_id = run_id or orion_day_run_id(brief.letter_date, attempt)
    return DurableRunRequestV1(
        run_id=run_id,
        workflow=ORION_DAY_WORKFLOW,
        correlation_id=correlation_id or run_id,
        **({"requested_at": requested_at} if requested_at is not None else {}),
        brief=brief,
        admission=ResourceRequirementV1(
            resource=f"llm.route.{ORION_DAY_LLM_ROUTE}",
            preferred_lane=ORION_DAY_LLM_ROUTE,
            # Tier 0 (Juniper 2026-11-12: the letter "needs to happen each day"): ahead of background work, behind
            # Juniper's own chat (interactive) and perception. Live 11-11 it queued at "background", position 4.
            priority="system",
            requirements={"minimum_context_tokens": minimum_context_tokens(brief)},
            deadline_at=deadline_at or default_deadline(brief.letter_date, tz_name=brief.timezone),
        ),
    )
