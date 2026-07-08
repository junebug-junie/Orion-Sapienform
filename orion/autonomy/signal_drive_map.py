"""Load + validate the structural signal->drive map (spec 2026-07-07 §3).

The map is the entire, closed mapping surface from a signal's ``(kind, dimension)``
to drive impulses. It carries no free text and no lexical matching; growth
requires a typed YAML entry, not prose. This module only reads typed fields and
never inspects any natural-language content of a signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import yaml

from orion.spark.concept_induction.drives import DRIVE_KEYS

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "autonomy" / "signal_drive_map.yaml"
)
_VALID_WORSE = {"up", "down"}


@dataclass(frozen=True)
class DimensionRule:
    dimension: str
    worse: str  # "up" | "down"
    drives: Mapping[str, float]


class SignalDriveMap:
    """Immutable typed lookup of dimension rules per signal_kind."""

    def __init__(self, rules: Dict[str, List[DimensionRule]]) -> None:
        self._rules = rules

    def rules_for(self, signal_kind: str) -> List[DimensionRule]:
        """Dimension rules for a signal_kind; empty list if unmapped."""
        return self._rules.get(signal_kind, [])

    def signal_kinds(self) -> List[str]:
        return sorted(self._rules.keys())


class SignalDriveMapError(ValueError):
    pass


def _validate_rule(kind: str, dim: str, spec: Mapping) -> DimensionRule:
    if not isinstance(spec, Mapping):
        raise SignalDriveMapError(f"{kind}.{dim}: expected mapping, got {type(spec).__name__}")
    worse = spec.get("worse")
    if worse not in _VALID_WORSE:
        raise SignalDriveMapError(f"{kind}.{dim}: worse must be one of {_VALID_WORSE}, got {worse!r}")
    drives = spec.get("drives") or {}
    if not isinstance(drives, Mapping) or not drives:
        raise SignalDriveMapError(f"{kind}.{dim}: drives must be a non-empty mapping")
    clean: Dict[str, float] = {}
    for drive, weight in drives.items():
        if drive not in DRIVE_KEYS:
            raise SignalDriveMapError(f"{kind}.{dim}: unknown drive {drive!r} (must be in {DRIVE_KEYS})")
        w = float(weight)
        if not 0.0 <= w <= 1.0:
            raise SignalDriveMapError(f"{kind}.{dim}.{drive}: weight {w} out of [0,1]")
        clean[drive] = w
    return DimensionRule(dimension=dim, worse=worse, drives=clean)


def load_signal_drive_map(path: str | Path | None = None) -> SignalDriveMap:
    """Parse and validate the YAML map. Raises SignalDriveMapError on any
    structural problem (unknown drive, bad ``worse``, empty drives)."""
    p = Path(path) if path is not None else _DEFAULT_PATH
    raw = yaml.safe_load(p.read_text()) or {}
    kinds = raw.get("signal_kinds") or {}
    if not isinstance(kinds, Mapping):
        raise SignalDriveMapError("signal_kinds must be a mapping")
    rules: Dict[str, List[DimensionRule]] = {}
    for kind, dims in kinds.items():
        if not isinstance(dims, Mapping):
            raise SignalDriveMapError(f"{kind}: dimensions must be a mapping")
        rules[kind] = [_validate_rule(kind, dim, spec) for dim, spec in dims.items()]
    return SignalDriveMap(rules)
