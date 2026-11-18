"""Run-cap override: a time-limited, runtime hold on one want family's start gate (Family C, 2026-11-16).

Design: docs/superpowers/specs/2026-11-16-family-c-first-experiment-economy-native.md, "Actuator: a run-cap
TTL override". Juniper's yes: #3482 comment, 2026-11-16.

One Redis key per family (``run_cap_key``), value ``RunCapOverrideV1`` JSON, always written with a Redis
``EX`` TTL. The family's own controller reads override-else-env at its start gate. A key with no TTL, a
value the family's bound does not allow, or a ``set_at`` older than the bound's ``max_hold_s`` is ignored,
never obeyed (``orion/physiology/run_cap_override.py``). The only writer today is orion-prediction's
town-hold experiment runner. Stored in Redis only; no bus channel.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

RUN_CAP_OVERRIDE_KIND = "physiology.run_cap_override.v1"
RUN_CAP_KEY_PREFIX = "orion:physiology:run_cap:"


def run_cap_key(family: str) -> str:
    return f"{RUN_CAP_KEY_PREFIX}{family}"


class RunCapOverrideV1(BaseModel):
    """One override. ``set_at`` is when the hold began (a block's start). The writer refreshes the key's
    TTL and ``expires_at`` but never moves ``set_at``, so a writer stuck in its refresh loop still cannot
    hold the family past ``max_hold_s``. ``phase``: ``pending`` (held while the writer decides; at most a
    minute or two) or ``block`` (a decided OFF block)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["physiology.run_cap_override.v1"] = RUN_CAP_OVERRIDE_KIND
    family: str = Field(min_length=1, max_length=120)
    knob: str = Field(min_length=1, max_length=60)
    value: bool
    experiment_id: str = Field(min_length=1, max_length=120)
    block_index: int = Field(ge=0)
    phase: Literal["pending", "block"] = "block"
    set_at: AwareDatetime
    expires_at: AwareDatetime
    writer: Optional[str] = Field(default=None, max_length=120)


__all__ = ["RUN_CAP_KEY_PREFIX", "RUN_CAP_OVERRIDE_KIND", "RunCapOverrideV1", "run_cap_key"]
