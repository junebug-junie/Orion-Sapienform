"""Patch 1 of `docs/superpowers/specs/2026-09-09-curiosity-supervisor-design.md`.

`HopReadingV1` is what the supervisor THINKS one hop was doing -- a READING,
not a fact, and not a write to `orion_worldview`. It exists so a human can
sample fifty of these and check whether the supervisor is reading Orion's own
prose correctly, per the spec's own acceptance check 1 and 7. Field names and
types match the spec's contract verbatim.

NOT registered in `orion/schemas/registry.py` / `orion/bus/channels.yaml`.
Registration matters once something publishes this on the bus; Patch 1
explicitly does not (see the spec's "Recommended next patch": "No bus wiring,
no interventions, no live subscription"). The next patch that arms a channel
for this should register it there.

`extra="forbid"` on both models: nothing downstream consumes these yet, so
there is no compatibility reason to accept unknown fields, and forbidding
them makes a schema-drifted LLM response fail loudly during parsing rather
than silently dropping a field nobody noticed was missing.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HopReadingV1(BaseModel):
    """What the supervisor thinks one hop was doing. A READING, not a fact."""

    model_config = ConfigDict(extra="forbid")

    hop_run_id: str
    hop_n: int
    # Derived from the note's prose. Recorded so a human can check fifty of
    # these and see whether it is reading them right -- the supervisor must
    # not be the only witness to its own accuracy. `None` is a real answer:
    # a hop can be about no live-or-closed prior at all (e.g. bookkeeping,
    # or a claim formed and never persisted).
    about_prior_id: Optional[str] = None
    # Free-form on purpose -- CLAUDE.md 0A bans fixing a taxonomy up front.
    # Let the kinds fall out of what the supervisor actually writes.
    kind: str
    # None = could not tell, a real third state alongside True/False.
    moved_the_claim: Optional[bool] = None
    reading_confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class HopReadingBatchV1(BaseModel):
    """The structured-output wrapper for one run's worth of hops.

    One LLM call answers every hop in a run at once -- cheaper than one call
    per hop, and it gives the model the run's full context (a run usually
    investigates one thread) instead of one isolated note at a time.
    """

    model_config = ConfigDict(extra="forbid")

    readings: list[HopReadingV1]
