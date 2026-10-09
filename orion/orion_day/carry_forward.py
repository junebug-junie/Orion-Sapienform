"""Carry-forward: how yesterday's letter reaches one waking curiosity run, exactly once.

Same offer-once shape as orion/dream/hypotheses.py (TAKE_FOR_OFFER_SQL / RELEASE_FOR_RUN_SQL):
Hub's regular curiosity kickoff claims the newest unexpired, never-offered letter's
``carry_forward_md`` and stamps it offered to that run. A run cancelled before Orion saw it
gives the claim back.

Only ``carry_forward_md`` travels. ``note_md`` (Orion's freeform note about the day) is never
selected here, so it cannot reach a curiosity prompt through this seam.

Freshness: ``carry_forward_expires_at`` is set by the durable run at persist time
(``created_at + brief.carry_forward_ttl_hours``; Hub sends HUB_ORION_DAY_CARRY_FORWARD_TTL_HOURS,
default 36). An expired or already-offered letter is never offered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

logger = logging.getLogger("orion.orion_day.carry_forward")

TAKE_CARRY_FORWARD_SQL = """
UPDATE orion_day_letter
   SET carry_forward_offered_at = now(), carry_forward_offered_run_id = $1
 WHERE letter_date = (
        SELECT letter_date FROM orion_day_letter
         WHERE carry_forward_offered_at IS NULL
           AND carry_forward_expires_at > now()
         ORDER BY letter_date DESC
         LIMIT 1
         FOR UPDATE SKIP LOCKED)
RETURNING letter_date, carry_forward_md
"""

RELEASE_CARRY_FORWARD_SQL = """
UPDATE orion_day_letter
   SET carry_forward_offered_at = NULL, carry_forward_offered_run_id = NULL
 WHERE carry_forward_offered_run_id = $1
"""


@dataclass(frozen=True)
class OfferedCarryForward:
    letter_date: date
    text: str


async def take_carry_forward(pool: Any, *, run_id: str) -> OfferedCarryForward | None:
    """Claim the freshest unoffered carry-forward for one run. None on any failure.

    Silence over a false section: a missing table (migration not applied) or a down pool
    must not break the kickoff."""
    if pool is None or not run_id:
        return None
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(TAKE_CARRY_FORWARD_SQL, run_id)
    except Exception as exc:  # noqa: BLE001 -- see docstring
        logger.warning("orion_day_carry_forward_take_failed run=%s err=%s", run_id, exc)
        return None
    if row is None:
        return None
    text = str(row["carry_forward_md"] or "").strip()
    if not text:
        return None
    return OfferedCarryForward(letter_date=row["letter_date"], text=text)


async def release_carry_forward(pool: Any, *, run_id: str) -> None:
    """Un-claim a run's carry-forward when the run was cancelled before Orion saw it. Never raises."""
    if pool is None or not run_id:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(RELEASE_CARRY_FORWARD_SQL, run_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("orion_day_carry_forward_release_failed run=%s err=%s", run_id, exc)


def carry_forward_header(letter_date: date | str) -> str:
    return f"THREADS CARRIED FORWARD FROM YESTERDAY'S REFLECTION (Orion's Day, {letter_date})"


def format_carry_forward_section(offered: OfferedCarryForward | None) -> list[str]:
    """Kickoff lines. Empty when nothing was claimed. Its own section, never merged with the
    dream section or the study material."""
    if offered is None or not offered.text.strip():
        return []
    return [
        carry_forward_header(offered.letter_date) + " (optional).",
        "Each morning you look back over the previous day and name threads worth "
        "picking up. These are the ones you named. They are yours, not an assignment: "
        "follow one if it still pulls at you, or leave them. Shown to you once.",
        "",
        offered.text.strip(),
        "",
    ]
