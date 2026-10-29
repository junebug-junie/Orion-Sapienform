"""Retro leisure start: the motor allocator asks Hub to start one solo Zelda leisure session now.

Orion plays Zelda, section E (docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md).
Producer: orion-execution-dispatch-runtime (route ``retro_leisure_session``, executor key
``orion.retro_leisure.start.v1``, act ``play:hub.retro_leisure:artifact_produced``). Consumer:
orion-hub (``scripts/retro_leisure_listener.py``), which asks the leisure runner's own
``should_start()`` gate (arousal, cabinet heat, human present, caps, ``RETRO_LEISURE_ENABLED``) and,
when it passes, starts the session and opens the ``play`` act.

One operation. The session's object is minted by the dispatch (its act key is
``retro_leisure_dispatch:<dispatch_id>``), so there is no control arm to draw and no need for a
side-effect-free ``check`` first (contrast orion/schemas/world_pulse_read_pull.py).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RETRO_LEISURE_START_REQUEST_CHANNEL = "orion:retro_leisure:start:request"
RETRO_LEISURE_START_REPLY_PREFIX = "orion:retro_leisure:start:reply:"
RETRO_LEISURE_START_REQUEST_KIND = "retro_leisure.start.request.v1"
RETRO_LEISURE_START_RESULT_KIND = "retro_leisure.start.result.v1"

# The act ledger key (grammar_acts.test_before) of the `play` act a started session opens.
RETRO_LEISURE_OBJECT_PREFIX = "retro_leisure_dispatch:"


def retro_leisure_object_ref(dispatch_id: str) -> str:
    return f"{RETRO_LEISURE_OBJECT_PREFIX}{dispatch_id}"


StartState = Literal["started", "refused"]


class RetroLeisureStartRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_id: str = Field(..., min_length=1, max_length=256)
    correlation: dict[str, Any] = Field(default_factory=dict)


class RetroLeisureStartResultV1(BaseModel):
    """``refusal`` is set only for ``state=refused``: the runner's own named reason (for example
    ``arousal_strained`` | ``cabinet_hot`` | ``human_present`` | ``session_cap`` | ``daily_cap`` |
    ``disabled``), or the listener's ``runner_unavailable`` | ``start_failed``.
    ``retry_after_sec`` is how long the same refusal will hold, from the gate's own clock."""

    model_config = ConfigDict(extra="forbid")

    dispatch_id: str
    state: StartState
    session_id: str | None = Field(None, max_length=256)
    # The act ledger key the session's `play` act carries (retro_leisure_object_ref).
    object_ref: str | None = None
    refusal: str | None = Field(None, max_length=200)
    retry_after_sec: float | None = Field(None, ge=0.0)
    gates: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "RETRO_LEISURE_OBJECT_PREFIX",
    "RETRO_LEISURE_START_REPLY_PREFIX",
    "RETRO_LEISURE_START_REQUEST_CHANNEL",
    "RETRO_LEISURE_START_REQUEST_KIND",
    "RETRO_LEISURE_START_RESULT_KIND",
    "RetroLeisureStartRequestV1",
    "RetroLeisureStartResultV1",
    "StartState",
    "retro_leisure_object_ref",
]
