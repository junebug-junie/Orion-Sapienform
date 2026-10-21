"""Reading pull: the motor allocator asks Hub to start one extra world-pulse read now.

Thinking rung 3 follow-up (Juniper 2026-10-21). Producer: orion-execution-dispatch-runtime (route
``read_world_pulse``, executor key ``orion.world_pulse_read.pull.v1``). Consumer: orion-hub
(``scripts/world_pulse_read_pull_listener.py``), which runs the reading pipeline's own gates plus the
pull gates (kill switch, daily cap, arousal, thermal) and picks the seed with the queue's own order.

Two operations, so a refused pull writes no episode row:

- ``check``: gates + selection, no side effect. Reply ``ready`` (with ``seed_id``) or ``refused``.
- ``act``: gates again, then ``arm=treated`` starts the read (``started``) and ``arm=control`` only
  marks the seed as this decision's control (``control_recorded``); the baseline lane may still read
  it. Both arms mark the seed, so it is never chosen by a later pull.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

WORLD_PULSE_READ_PULL_REQUEST_CHANNEL = "orion:world_pulse_read:pull:request"
WORLD_PULSE_READ_PULL_REPLY_PREFIX = "orion:world_pulse_read:pull:reply:"
WORLD_PULSE_READ_PULL_REQUEST_KIND = "world_pulse_read.pull.request.v1"
WORLD_PULSE_READ_PULL_RESULT_KIND = "world_pulse_read.pull.result.v1"

PullState = Literal["ready", "started", "control_recorded", "refused"]


class WorldPulseReadPullRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["check", "act"]
    dispatch_id: str = Field(..., min_length=1, max_length=256)
    # Required for ``act``; ignored for ``check``.
    arm: Literal["treated", "control"] | None = None
    # The seed the ``check`` returned. ``act`` re-selects; a different seed is reported, not hidden.
    expected_seed_id: str | None = Field(None, max_length=512)
    correlation: dict[str, Any] = Field(default_factory=dict)


class WorldPulseReadPullResultV1(BaseModel):
    """``refusal`` is set only for ``state=refused``: ``disabled`` | ``daily_cap`` |
    ``arousal_strained`` | ``thermal_hot`` | a reading-pipeline gate (``outside_window`` |
    ``refund_backoff``) | ``no_candidate`` | ``already_pulled`` | ``queue_unavailable``.
    ``retry_after_sec`` is how long the same refusal will hold, from the gate's own clock."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["check", "act"]
    dispatch_id: str
    state: PullState
    arm: Literal["treated", "control"] | None = None
    seed_id: str | None = None
    # The act ledger key a `look` close for this seed carries (grammar_acts.test_before).
    object_ref: str | None = None
    refusal: str | None = None
    retry_after_sec: float | None = Field(None, ge=0.0)
    pulled_today: int | None = None
    daily_cap: int | None = None
    gates: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "PullState",
    "WORLD_PULSE_READ_PULL_REPLY_PREFIX",
    "WORLD_PULSE_READ_PULL_REQUEST_CHANNEL",
    "WORLD_PULSE_READ_PULL_REQUEST_KIND",
    "WORLD_PULSE_READ_PULL_RESULT_KIND",
    "WorldPulseReadPullRequestV1",
    "WorldPulseReadPullResultV1",
]
