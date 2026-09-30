"""The interface the Hub scheduler calls: gather -> budget -> brief -> durable request.

    brief = await build_orion_day_brief(conn, letter_date)       # raises OrionDayEmptyError on a blank day
    request = build_orion_day_request(brief, attempt=1)          # DurableRunRequestV1, admitted, agent lane
    # submit: POST {DURABLE_RUNS_URL}/runs with request.model_dump(mode="json"), or publish it on
    # orion:durable:run:request (kind durable.run.request.v1). The row lands in orion_day_letter.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from orion.orion_day.budget import DEFAULT_BUDGET_TOKENS, build_llm_view
from orion.orion_day.gather import DEFAULT_GITHUB_REPO, gather_orion_day
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.orion_day import (
    ORION_DAY_LLM_ROUTE,
    ORION_DAY_TIMEZONE,
    ORION_DAY_WORKFLOW,
    OrionDayMaterialV1,
    OrionDayRunBriefV1,
    orion_day_run_id,
)
from orion.schemas.resource_admission import ResourceRequirementV1


# Completion budgets the hold must leave room for (cortex-exec LLM_ORION_DAY_NOTE_MAX_TOKENS /
# LLM_ORION_DAY_CARRY_FORWARD_MAX_TOKENS defaults): the carry-forward prompt is the digest PLUS
# the note, so one hold must fit digest + note + carry-forward.
NOTE_COMPLETION_TOKENS = 12_000
CARRY_FORWARD_COMPLETION_TOKENS = 4_000
PROMPT_FRAME_TOKENS = 1_500


def minimum_context_tokens(brief: OrionDayRunBriefV1) -> int:
    """The context a card needs for this run's two calls. Sent as the admission's
    ``requirements.minimum_context_tokens`` (the pool only grants a card whose live
    ctx_per_slot covers it): a heavy day must not be placed on the 65,536-token chat card
    (live /props 2026-09-30) when the pool spills agent work there; a light day still may."""
    return (brief.llm_view.approx_tokens + PROMPT_FRAME_TOKENS
            + NOTE_COMPLETION_TOKENS + CARRY_FORWARD_COMPLETION_TOKENS)


class OrionDayEmptyError(RuntimeError):
    """The day holds nothing to write about. No letter is better than an empty-shell one."""


def brief_from_material(
    material: OrionDayMaterialV1,
    *,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    timeout_sec: float = 1800.0,
    carry_forward_ttl_hours: float = 48.0,
) -> OrionDayRunBriefV1:
    if material.is_empty():
        raise OrionDayEmptyError(f"orion_day_empty:{material.letter_date}")
    return OrionDayRunBriefV1(
        letter_date=material.letter_date,
        timezone=material.timezone,
        window_start=material.window_start,
        window_end=material.window_end,
        material=material,
        llm_view=build_llm_view(material, budget_tokens=budget_tokens),
        llm_route=ORION_DAY_LLM_ROUTE,
        timeout_sec=timeout_sec,
        carry_forward_ttl_hours=carry_forward_ttl_hours,
    )


async def build_orion_day_brief(
    conn: Any,
    letter_date: date,
    *,
    now: datetime | None = None,
    github_repo: str = DEFAULT_GITHUB_REPO,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    timeout_sec: float = 1800.0,
    carry_forward_ttl_hours: float = 48.0,
) -> OrionDayRunBriefV1:
    material = await gather_orion_day(conn, letter_date, now=now, github_repo=github_repo)
    return brief_from_material(material, budget_tokens=budget_tokens, timeout_sec=timeout_sec,
                               carry_forward_ttl_hours=carry_forward_ttl_hours)


def default_deadline(letter_date: date, *, tz_name: str = ORION_DAY_TIMEZONE) -> datetime:
    """End of the day AFTER the letter's day (local): the run may queue through the 06:00
    congestion and most of the day before it gives up."""
    local_end = datetime.combine(letter_date + timedelta(days=2), time.min, tzinfo=ZoneInfo(tz_name))
    return local_end.astimezone(timezone.utc)


def build_orion_day_request(
    brief: OrionDayRunBriefV1,
    *,
    attempt: int = 1,
    deadline_at: datetime | None = None,
    correlation_id: str | None = None,
    requested_at: datetime | None = None,
) -> DurableRunRequestV1:
    run_id = orion_day_run_id(brief.letter_date, attempt)
    return DurableRunRequestV1(
        run_id=run_id,
        workflow=ORION_DAY_WORKFLOW,
        correlation_id=correlation_id or run_id,
        **({"requested_at": requested_at} if requested_at is not None else {}),
        brief=brief,
        admission=ResourceRequirementV1(
            resource=f"llm.route.{ORION_DAY_LLM_ROUTE}",
            preferred_lane=ORION_DAY_LLM_ROUTE,
            priority="background",
            requirements={"minimum_context_tokens": minimum_context_tokens(brief)},
            deadline_at=deadline_at or default_deadline(brief.letter_date, tz_name=brief.timezone),
        ),
    )
