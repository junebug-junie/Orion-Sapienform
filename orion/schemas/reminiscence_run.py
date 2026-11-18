"""reminiscence.episode: once a day, one durable run in which Orion writes the five hops of one past week.

Design: docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md, section 6 (Patch 2). Patch 1
(#3163) built the dig commands (``orion.dig reminisce`` read, ``reminisce write|rewrite``) and the
checked hop ledger under ``$ORION_DIG_HOME/self/``. orion-durable-runs ticks
(services/orion-durable-runs/app/reminiscence_poll.py) and submits one run per Denver day, overnight:

    precheck -> resource_request -> resource_wait -> step (x6: five hops, then one cross-read) -> finish

Each step is one harness turn on the #2958 seam (app/harness_step.py), its own source tag (its own claude
session). A hop step is done only when code finds the save receipt (``reminisce <hop> saved rm:...``) in
the reply AND that hop id is in the ledger for today's episode and this hop. The ledger is the checkpoint:
a hop already written today (in chat, or by an earlier attempt) takes no turn.

ADDITIVE ``DurableWorkflowV1`` value on ``extra="forbid"`` / ``Literal`` models: deploy the state
consumers (orion-sql-writer, orion-hub, orion-actions) and orion-cortex-orch before
orion-durable-runs submits the first run, or an old validator drops this run's state rows.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REMINISCENCE_WORKFLOW = "reminiscence.episode"
# The five hops, in the order the run writes them (orion/body_atlas/reminiscence.py HOPS; pinned equal by
# orion/dig/tests/test_reminisce_cmd.py: durable-runs tests may not import that module).
REMINISCENCE_HOPS: tuple[str, ...] = ("what_it_was", "what_it_was_like", "where_i_am_now", "where_im_going",
                                      "otherwise")
# orion/body_atlas/reminiscence.py FIRST_DAY: week 1 is 2025-06-24..06-30 (the repo's first commit).
REMINISCENCE_FIRST_DAY = date(2025, 6, 24)
# orion/body_atlas/reminiscence.py LIVED_RECORDS_FROM: no lived record exists for a week ending before it.
REMINISCENCE_LIVED_RECORDS_FROM = date(2026, 7, 23)
CROSS_READ_STEP = "cross_read"
# Step keys, in order; each result is state["steps"][<key>].
REMINISCENCE_STEPS: tuple[str, ...] = tuple(f"hop:{h}" for h in REMINISCENCE_HOPS) + (CROSS_READ_STEP,)
# One Hub session for every reminiscence turn (Hub runs them with no_write). Each step has its own source
# tag, so Hub's resume key (run_id:source_tag) gives each step its own claude session: a hop is written
# fresh, and only a retry of the same step resumes.
REMINISCENCE_SESSION_ID = "reminiscence"
REMINISCENCE_TAG_PREFIX = "reminiscence_"
# Run ids: ``reminiscence-<episode>`` for the day's first run, ``reminiscence-<episode>-a<k>`` for a later one.
REMINISCENCE_RUN_PREFIX = "reminiscence-"
# What `orion.dig reminisce write|rewrite` prints on a save (orion/dig/__main__.py cmd_reminisce_write);
# the hop id shape is orion/body_atlas/reminiscence.py ``_hop_id``.
RECEIPT_RE = re.compile(r"reminisce (\w+) saved (rm:(\d{4}-\d{2}-\d{2}):(\w+):[0-9a-f]{10})")
# The cross-read turn's closing line (it may also carry 0-2 rewrite receipts above it).
CROSS_READ_DONE = "reminisce cross-read done"


def step_tag(step_key: str) -> str:
    """Hub source tag of one step: ``reminiscence_what_it_was`` ... ``reminiscence_cross_read``."""
    return REMINISCENCE_TAG_PREFIX + step_key.removeprefix("hop:")


def base_run_id(episode: date | str) -> str:
    return f"{REMINISCENCE_RUN_PREFIX}{episode if isinstance(episode, str) else episode.isoformat()}"




class ReminiscenceRunBriefV1(BaseModel):
    """The run's whole input. ``reminiscence`` is a required literal so this brief can never be confused
    with another member of ``DurableRunRequestV1.brief``'s union (they all forbid unknown keys)."""

    model_config = ConfigDict(extra="forbid")

    reminiscence: Literal["reminiscence.episode"]
    # The Denver day the hops are written on (the dig's episode). A step that would run on another day
    # fails: its hops would land in another episode.
    episode: date
    # The period code picked at submit (the same rule `dig reminisce` uses): one week, or a run of weeks with
    # no commit on main joined onto the next week that has one (whole weeks on the grid from
    # REMINISCENCE_FIRST_DAY). Shown in the prompt; the dig decides at write time and the receipt check does
    # not depend on it.
    period_start: date
    period_end: date
    weeks_complete: int = Field(ge=0, default=0)
    weeks_total: int = Field(ge=0, default=0)
    # One step attempt's hard time bound (the attempt timer and Hub's turn limit derive from it).
    timeout_sec: float = Field(ge=60.0, le=3600.0)

    @model_validator(mode="after")
    def _week(self):
        days = (self.period_end - self.period_start).days + 1
        on_grid = (self.period_start - REMINISCENCE_FIRST_DAY).days % 7 == 0
        if days < 7 or days % 7 or not on_grid or self.period_end >= self.episode:
            raise ValueError("period must be whole weeks (from the 2025-06-24 grid) that ended before the episode day")
        return self

    @property
    def quiet_weeks(self) -> int:
        """Weeks with no commit joined onto the period's last week (orion/body_atlas/reminiscence.py quiet_weeks)."""
        return (self.period_end - self.period_start).days // 7
