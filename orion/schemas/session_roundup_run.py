"""session_roundup.orion_turn: Orion's periodic read (every 15 min) of the messages coding sessions sent, as one durable run.

Hub's round-up loop (services/orion-hub/scripts/session_roundup_loop.py) holds every message a coding
session sends for the bucket. Once a bucket (15 min), when at least one thread is held, it submits ONE run of this
workflow: one harness turn whose prompt holds every held thread, and whose reply is Orion's questions
per thread. Run id ``session-roundup-<utc bucket start, YYYYmmddTHHMM>``: one per bucket, so a Hub restart cannot submit twice.

The graph is the single-step harness graph consult.orion_turn uses
(services/orion-durable-runs/app/harness_smoke_graph.py); Hub reads the step's full text back from
``GET /runs/{run_id}``. Its own workflow and act family (``durable.session_roundup``), so it never
spends the consult family's budget.

ADDITIVE ``DurableWorkflowV1`` value on ``extra="forbid"`` / ``Literal`` models: deploy the state
consumers (orion-sql-writer, orion-actions) and orion-cortex-orch, then orion-durable-runs, then
orion-hub LAST. Hub is the only producer and submits only after it is deployed.

Design: docs/superpowers/specs/2026-11-15-session-threads-design.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SESSION_ROUNDUP_WORKFLOW = "session_roundup.orion_turn"
SESSION_ROUNDUP_STEP_KEY = "turn"
# Hub's turn ``source`` for round-up turns.
SESSION_ROUNDUP_TAG = "session_roundup"
# The most text one round-up prompt may hold (chars). Hub's own budget stays under it.
SESSION_ROUNDUP_PROMPT_MAX_CHARS = 120000


def session_roundup_run_id(bucket_start: datetime) -> str:
    """Deterministic: one durable run per round-up bucket (15 minutes by default). ``bucket_start`` is the
    UTC start of the bucket the round belongs to."""
    return "session-roundup-" + bucket_start.astimezone(timezone.utc).strftime("%Y%m%dT%H%M")


def session_roundup_session_id(roundup_id: str) -> str:
    """Orion's own session for one round: none of Juniper's chats, and no carry between rounds."""
    return f"orion_{roundup_id}"


class SessionRoundupBriefV1(BaseModel):
    """The turn's whole input. ``roundup`` is a required literal so this brief can never be confused
    with another member of ``DurableRunRequestV1.brief``'s union (they all forbid unknown keys)."""

    model_config = ConfigDict(extra="forbid")

    roundup: Literal["session_roundup.orion_turn"]
    roundup_id: str = Field(min_length=1, max_length=64)
    thread_ids: list[str] = Field(min_length=1, max_length=50)
    prompt: str = Field(min_length=1, max_length=SESSION_ROUNDUP_PROMPT_MAX_CHARS)
    session_id: str = Field(min_length=1, max_length=128)
    # What recall searches for: the threads' titles, not the whole prompt.
    retrieval_query: Optional[str] = Field(default=None, max_length=1000)
    # The step's budget per attempt (the attempt timer and Hub's turn limit derive from it).
    timeout_sec: float = Field(ge=60.0, le=3600.0)
