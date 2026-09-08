"""Bus payload for the one Claude budget signal that is real.

`orion:substrate:claude_limit`   ClaudeLimitObservationV1
    producer: orion-cocreation-signals (the only service that mounts
              Juniper's `~/.claude/projects` tree, read-only)
    consumer: orion-hub (budget surface + the ask_claude dry-run gate)

WHY THIS AND NOT DOLLARS. `orion/autonomy/quota_budget.py` denominates
Claude scarcity in dollars against an operator allowance. That denominator
was measured and refuted -- no dollar threshold separates limited from
not-limited (`docs/superpowers/specs/2026-08-27-quota-window-calibration-
finding.md`). `orion/dev_economics/rate_limit_events.py` replaced it by
reading the constraint's own first-party announcement, which carries the
reset time:

    You've hit your session limit · resets 3:30am (UTC)

That module has been correct and completely unconsumed since it shipped.
This schema is the wire shape that gives it a consumer.

THREE STATES, AND `unknown` IS NOT `clear`. `observed` is carried as its own
field rather than left to be inferred from `event_count == 0`, because those
two are the same number for "the pool is full" and "nobody has looked".
Every consumer of this payload has to be able to tell an empty window from
an unread one -- the `bus_synaptic_prediction_error` / `node:substrate.route`
incidents in CLAUDE.md §0A exist to force exactly that distinction, and a
budget that reads "plenty left" during a transcript-mount outage is the same
class of defect.

`event_count` is GRADED PRESSURE, deliberately distinct from `state`. Six
limit events in five hours and one event five hours ago both read `clear`
now and mean very different things about how contended the pool is.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

ClaudeLimitState = Literal["clear", "limited", "unknown"]


class ClaudeLimitObservationV1(BaseModel):
    """One scan of the transcripts on disk, for one trailing window.

    Mirrors `orion.dev_economics.rate_limit_events.LimitObservation` field for
    field rather than summarising it, so a consumer never has to re-derive a
    property the producer already computed correctly. The individual
    `RateLimitEvent`s are NOT carried: they hold message text, and the
    dev-economics privacy boundary is token counts and timestamps only.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate.claude_limit.v1"] = "substrate.claude_limit.v1"
    event_id: str = Field(default_factory=lambda: f"claude-limit-{uuid4()}")
    observed_at: datetime

    window_hours: float
    window_start: datetime
    window_end: datetime

    state: ClaudeLimitState
    # False means UNOBSERVED, not quiet. See the module docstring.
    observed: bool
    # How often the limit bound inside this window. Graded pressure, not state.
    event_count: int
    # Populated only while a limit is in force AND its message stated a reset
    # time. `state == "limited"` with `resets_at is None` is a real case: the
    # constraint bound but did not say when it lifts.
    resets_at: Optional[datetime] = None
    seconds_until_reset: Optional[float] = None
    # Age of the freshest observation, or None when nothing was observed. A
    # `clear` reading with large staleness is weaker evidence than a fresh one;
    # threshold on this rather than treating `state` as instantaneous.
    staleness_sec: Optional[float] = None

    observed_message_count: int
    scanned_file_count: int
