"""The `physiology:` block every cortex verb declares (body economy Patch C2).

Spec: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, "Execution seam"
and Phase 2 Gate 1 row "Every verb declares its physiology".

Each verb YAML in ``orion/cognition/verbs/`` carries::

    physiology:
      tier: 2                      # 0 vital (never bids) | 1 critical upkeep | 2 discretionary
      family: thought.painting     # want family; matches the act-ledger actor where one exists
      lanes: [diffusion, metacog]  # GPU pool roles/classes the verb's work lands on ([] = no GPU)
      deferrable_window_s: 5400    # how long it can wait and still be useful (0 = perishable)

This module only parses and validates the block. Nothing acts on it yet: Patch C3 makes
cortex-orch read it at dispatch and bid for tier 1/2 verbs. Tier 0 never bids, so a verb that
runs only on Juniper's turn, or as autonomic plumbing, declares ``tier: 0`` with family
``turn`` or ``autonomic``; that keeps the gate total without letting tier 0 become an escape
hatch for an Orion-chosen act (a tier-0 verb with any other family is rejected, and so is a
``turn``/``autonomic`` verb on tier 1 or 2). Tier 0 is never deferred, so its window is 0.

Lanes: each name is a class in ``config/gpu_pool.yaml`` when one of that name exists (``fast`` the
class spans every role that may serve fast work), otherwise a single role (``agent-gpu2``,
``diffusion``). Class names come first because a verb reaches the pool through a route, and a
route names a class.

Non-zero windows are authored priors (the act horizon where one exists, e.g. painting 5400 s),
not measurements. C3 should calibrate them from the act ledger before pricing on them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

#: Families that mean "never bids". Tier 0 must use one of these, and these only appear on tier 0.
TIER0_FAMILIES = frozenset({"turn", "autonomic"})

#: A family is a dotted slug, e.g. ``turn``, ``dream.cycle``, ``durable.self_inquiry``.
_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$")

#: Pool role/class names are kebab-or-snake slugs, e.g. ``agent-gpu2``, ``memory_distill``.
_LANE_RE = re.compile(r"^[a-z][a-z0-9_-]*$")

PHYSIOLOGY_KEY = "physiology"


class VerbPhysiologyV1(BaseModel):
    """One verb's declared economics. Every field is required when the block is present."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: Literal[0, 1, 2]
    family: str = Field(min_length=1)
    lanes: list[str]
    deferrable_window_s: StrictInt = Field(ge=0)

    @field_validator("tier", mode="before")
    @classmethod
    def _tier_is_an_int(cls, value: Any) -> Any:
        # Literal[0, 1, 2] alone accepts True/False (bool is an int) and "1" in lax mode.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"tier must be the integer 0, 1 or 2, got {value!r}")
        return value

    @field_validator("family")
    @classmethod
    def _family_is_a_slug(cls, value: str) -> str:
        if not _FAMILY_RE.match(value):
            raise ValueError(f"family must be a lower-case dotted slug (e.g. 'dream.cycle'), got {value!r}")
        return value

    @field_validator("lanes")
    @classmethod
    def _lanes_are_slugs(cls, value: list[str]) -> list[str]:
        for lane in value:
            if not isinstance(lane, str) or not _LANE_RE.match(lane):
                raise ValueError(f"lane must be a GPU pool role or class name, got {lane!r}")
        if len(set(value)) != len(value):
            raise ValueError(f"lanes must not repeat, got {value!r}")
        return value

    @model_validator(mode="after")
    def _tier_and_family_agree(self) -> "VerbPhysiologyV1":
        if self.tier == 0 and self.family not in TIER0_FAMILIES:
            raise ValueError(
                f"tier 0 never bids, so its family must be one of {sorted(TIER0_FAMILIES)}; "
                f"got family {self.family!r}. An act Orion chooses is tier 1 or 2."
            )
        if self.tier != 0 and self.family in TIER0_FAMILIES:
            raise ValueError(f"family {self.family!r} is reserved for tier 0, got tier {self.tier}")
        if self.tier == 0 and self.deferrable_window_s != 0:
            raise ValueError(f"tier 0 is never deferred, so deferrable_window_s must be 0, got {self.deferrable_window_s}")
        return self


def parse_verb_physiology(
    data: dict[str, Any] | None, *, known_lanes: Optional[Iterable[str]] = None
) -> Optional[VerbPhysiologyV1]:
    """The verb's ``physiology`` block, or None when the YAML has none.

    Raises ``ValueError`` (pydantic's ``ValidationError`` is one) when the block is present but
    invalid. ``known_lanes``, when given, also rejects a lane the GPU pool does not define.
    """
    if not isinstance(data, dict) or PHYSIOLOGY_KEY not in data:
        return None
    raw = data[PHYSIOLOGY_KEY]
    if not isinstance(raw, dict):
        raise ValueError(f"physiology must be a mapping, got {type(raw).__name__}")
    block = VerbPhysiologyV1.model_validate(raw)
    if known_lanes is not None:
        known = set(known_lanes)
        unknown = [lane for lane in block.lanes if lane not in known]
        if unknown:
            raise ValueError(f"unknown lane(s) {unknown}: not a role or class in config/gpu_pool.yaml")
    return block


def load_pool_lanes(pool_path: Path) -> frozenset[str]:
    """Every role and class name in ``config/gpu_pool.yaml``: the names a verb's lanes may use.

    One namespace on purpose: a name that is both a class and a role reads as the class (see the
    module docstring)."""
    data = yaml.safe_load(pool_path.read_text(encoding="utf-8")) or {}
    roles = data.get("roles") or {}
    classes = data.get("classes") or {}
    if not isinstance(roles, dict) or not isinstance(classes, dict) or not roles:
        raise ValueError(f"{pool_path}: expected non-empty 'roles' and a 'classes' mapping")
    return frozenset(str(name) for name in (*roles.keys(), *classes.keys()))
