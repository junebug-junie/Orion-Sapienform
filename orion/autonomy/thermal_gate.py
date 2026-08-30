"""A budget denominated in something Orion cannot argue with: the room.

WHY THIS ONE IS DIFFERENT
-------------------------
Every other budget in this repo meters a resource Orion spends on itself, and
grades the result itself. `orion/autonomy/budget.py` meters motor-seconds and
`quota_budget.py` meters dollars; both are real, but the *value* side of each is
Orion's own estimate of what it learned. That is the wireheading shape the
self-calibration roadmap names directly: Orion sets the weights, the weights
decide what Orion does, Orion scores the outcome, and nothing outside settles it.

Ambient temperature settles it. The room is 30.7C with Juniper in it, GPU work
heats the room, and no amount of favourable self-assessment makes it cooler.
This is the first budget here whose referent is external, physical, and
adversarial to the thing spending it -- which is exactly what the power prior
already demonstrated for prediction (hardware settles it) and what the
introspective action space never had.

WHAT IT DOES NOT DO
-------------------
It does not protect hardware. GPUs have their own thermal throttling and it is
better than this. This protects the *person in the room*, which is a different
threshold, reached much earlier, and one no hardware sensor is watching.

DESIGN NOTES
------------
Pure arithmetic. The caller supplies the reading; this module does no I/O, the
same shape as `budget.py`, so the policy is testable without a sensor.

Hysteresis is not optional. A bare `temp_c > threshold` on a reading that
wanders across the line yields a gate that flaps open and closed every tick,
which is worse than no gate: the work still happens, and now it happens
unpredictably. Re-arming is deliberately COOLER than tripping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ThermalState = Literal["normal", "elevated", "hot", "unknown"]

# Defaults chosen against the live reading (30.74C on 2026-08-30, with Juniper
# reporting ~34C in the office) rather than from a datasheet. `hot` is set below
# the reported room temperature on purpose: a threshold nothing ever crosses is
# a switch that changes nothing.
DEFAULT_HOT_C = 32.0
DEFAULT_HOT_REARM_C = 30.5
DEFAULT_ELEVATED_C = 29.5
DEFAULT_ELEVATED_REARM_C = 28.0

# A reading older than this is not evidence about the room now.
DEFAULT_MAX_READING_AGE_SEC = 300.0


@dataclass(frozen=True)
class ThermalVerdict:
    state: ThermalState
    temp_c: Optional[float]
    age_sec: Optional[float]
    allows_gpu_work: bool
    reason: str

    @property
    def degraded(self) -> bool:
        """True when the verdict rests on no usable reading."""
        return self.state == "unknown"


def thermal_state(
    *,
    temp_c: Optional[float],
    age_sec: Optional[float],
    previous_state: ThermalState = "normal",
    hot_c: float = DEFAULT_HOT_C,
    hot_rearm_c: float = DEFAULT_HOT_REARM_C,
    elevated_c: float = DEFAULT_ELEVATED_C,
    elevated_rearm_c: float = DEFAULT_ELEVATED_REARM_C,
    max_age_sec: float = DEFAULT_MAX_READING_AGE_SEC,
) -> ThermalVerdict:
    """Classify the room, with hysteresis, and say whether GPU work may proceed.

    `previous_state` is what makes this hysteretic: once `hot`, the gate stays
    hot until the room falls to `hot_rearm_c`, not merely back under `hot_c`.

    A missing or stale reading returns `unknown` and ALLOWS work. That direction
    is deliberate and is the opposite of what a hardware thermal cutout should
    do. The cost of wrongly allowing is that a room stays warm for one more
    cycle; the cost of wrongly blocking is that a dead sensor silently removes a
    capability with no error anywhere -- the exact failure this repo keeps
    finding. `degraded` is set so the caller can say so out loud instead of
    reporting a clean allow.
    """
    if temp_c is None:
        return ThermalVerdict(
            state="unknown",
            temp_c=None,
            age_sec=age_sec,
            allows_gpu_work=True,
            reason="no_reading",
        )
    if age_sec is not None and age_sec > max_age_sec:
        return ThermalVerdict(
            state="unknown",
            temp_c=temp_c,
            age_sec=age_sec,
            allows_gpu_work=True,
            reason=f"reading_stale_{age_sec:.0f}s",
        )

    was_hot = previous_state == "hot"
    was_elevated = previous_state in ("hot", "elevated")

    # Trip on the hot threshold; stay hot until the room reaches the cooler
    # re-arm point. `>=` on the trip so a threshold set exactly at a held
    # reading still fires.
    if temp_c >= hot_c or (was_hot and temp_c > hot_rearm_c):
        return ThermalVerdict(
            state="hot",
            temp_c=temp_c,
            age_sec=age_sec,
            allows_gpu_work=False,
            reason=f"room_at_{temp_c:.1f}c_hot_threshold_{hot_c:.1f}c",
        )
    if temp_c >= elevated_c or (was_elevated and temp_c > elevated_rearm_c):
        return ThermalVerdict(
            state="elevated",
            temp_c=temp_c,
            age_sec=age_sec,
            allows_gpu_work=True,
            reason=f"room_at_{temp_c:.1f}c_elevated",
        )
    return ThermalVerdict(
        state="normal",
        temp_c=temp_c,
        age_sec=age_sec,
        allows_gpu_work=True,
        reason=f"room_at_{temp_c:.1f}c",
    )
