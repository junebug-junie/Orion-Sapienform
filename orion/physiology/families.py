"""Want families as data: ``config/physiology/families.yaml`` (body economy Patch C1, Organ 3b).

One family per act-ledger actor. Each has a tier (0 never bids, 1 upkeep with a reserved share,
2 discretionary), a prior weight (Juniper's instinct, fading as evidence grows), a deferrable
window, acceptable lanes and how the act is placed on them. Pure parsing plus one file read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

SCHEMA_VERSION = 1
FIXED_COST_FIELDS = frozenset({"seconds", "gpu_seconds", "tokens", "watts", "juniper_msgs", "claude_limit_events"})


@dataclass(frozen=True)
class FamilyConfig:
    family: str
    tier: int
    prior_weight: float = 1.0
    deferrable_window_s: float = 0.0
    lanes: tuple[str, ...] = ()
    placement: str = "one_of"          # one_of | fixed
    fixed_cost: Mapping[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def bids(self) -> bool:
        return self.tier in (1, 2)


@dataclass(frozen=True)
class FamiliesConfig:
    families: Mapping[str, FamilyConfig]

    def get(self, family: str) -> Optional[FamilyConfig]:
        return self.families.get(family)

    def in_tier(self, tier: int) -> list[str]:
        return sorted(f for f, c in self.families.items() if c.tier == tier)


def parse_families(doc: Any) -> FamiliesConfig:
    if not isinstance(doc, dict) or doc.get("version") != SCHEMA_VERSION:
        raise ValueError(f"families config must be a mapping with version: {SCHEMA_VERSION}")
    raw = doc.get("families")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("families: must be a non-empty mapping")
    out: dict[str, FamilyConfig] = {}
    for name, body in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"bad family name {name!r}")
        body = body or {}
        if not isinstance(body, dict):
            raise ValueError(f"{name}: must be a mapping")
        tier = body.get("tier")
        if tier not in (0, 1, 2):
            raise ValueError(f"{name}: tier must be 0, 1 or 2, got {tier!r}")
        prior = float(body.get("prior_weight", 1.0))
        if not (math.isfinite(prior) and prior > 0):
            raise ValueError(f"{name}: prior_weight must be finite and > 0")
        window = float(body.get("deferrable_window_s", 0.0))
        if not (math.isfinite(window) and window >= 0):
            raise ValueError(f"{name}: deferrable_window_s must be finite and >= 0")
        lanes = body.get("lanes") or []
        if not isinstance(lanes, list) or not all(isinstance(x, str) and x for x in lanes):
            raise ValueError(f"{name}: lanes must be a list of pool roles")
        if any(x.startswith("lane:") for x in lanes):
            raise ValueError(f"{name}: lanes are pool roles, without the 'lane:' prefix")
        placement = body.get("placement", "one_of")
        if placement not in ("one_of", "fixed"):
            raise ValueError(f"{name}: placement must be one_of or fixed")
        fixed = body.get("fixed_cost") or {}
        if not isinstance(fixed, dict) or not set(fixed) <= FIXED_COST_FIELDS:
            raise ValueError(f"{name}: fixed_cost keys must be in {sorted(FIXED_COST_FIELDS)}")
        if tier != 0 and window <= 0:
            raise ValueError(f"{name}: a bidding family needs deferrable_window_s > 0 (its hunger clock)")
        out[name] = FamilyConfig(family=name, tier=int(tier), prior_weight=prior, deferrable_window_s=window,
                                 lanes=tuple(lanes), placement=placement,
                                 fixed_cost={k: float(v) for k, v in fixed.items()}, note=str(body.get("note") or ""))
    return FamiliesConfig(families=out)


def load_families(path: str | Path) -> tuple[Optional[FamiliesConfig], Optional[str]]:
    """(config, None) or (None, reason). Never raises."""
    import yaml

    p = Path(path)
    try:
        text = p.read_text()
    except FileNotFoundError:
        return None, f"missing:{p.name}"
    except OSError as exc:
        return None, f"read_failed:{p.name}:{type(exc).__name__}"
    try:
        return parse_families(yaml.safe_load(text)), None
    except Exception as exc:  # noqa: BLE001 -- a malformed config is a reason, not a crash
        return None, f"malformed:{p.name}:{str(exc)[:160]}"


__all__ = ["FamiliesConfig", "FamilyConfig", "load_families", "parse_families"]
