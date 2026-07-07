from __future__ import annotations

from typing import Sequence

from orion.core.schemas.drives import TensionEventV1
from orion.spark.concept_induction.tensions import clamp01

DRIVE_KEYS = ("coherence", "continuity", "capability", "relational", "predictive", "autonomy")

_GAP_KIND = "substrate.world_coverage_gap"

_PRIMARY_DRIVE_BY_KIND: dict[str, str] = {
    _GAP_KIND: "predictive",
    "tension.contradiction.v1": "coherence",
    "tension.distress.v1": "relational",
    "tension.identity_drift.v1": "continuity",
    "tension.cognitive_load.v1": "capability",
}


def compute_tick_attribution(
    tensions: Sequence[TensionEventV1],
    *,
    metabolism_deltas: dict[str, float] | None = None,
) -> dict[str, float]:
    """Sum magnitude × drive_impact weight per drive for THIS tick only."""
    attribution: dict[str, float] = {key: 0.0 for key in DRIVE_KEYS}
    for tension in tensions:
        mag = clamp01(tension.magnitude)
        for drive, weight in (tension.drive_impacts or {}).items():
            if drive not in attribution:
                continue
            attribution[drive] += mag * clamp01(float(weight))
    for drive, delta in (metabolism_deltas or {}).items():
        if drive in attribution:
            attribution[drive] += float(delta)
    return attribution


def primary_drive_for_tension_kind(
    kind: str,
    *,
    drive_impacts: dict[str, float] | None = None,
) -> str | None:
    """Structural map for tie-break; not digest keyword matching."""
    if kind == "tension.drive_competition.v1" and drive_impacts:
        ranked = sorted(drive_impacts.items(), key=lambda item: (-float(item[1]), item[0]))
        return ranked[0][0] if ranked else None
    return _PRIMARY_DRIVE_BY_KIND.get(kind)


def _tied_at_max(attribution: dict[str, float]) -> list[str]:
    if not attribution:
        return []
    max_val = max(attribution.values())
    if max_val <= 0.0:
        return []
    return sorted([drive for drive, val in attribution.items() if val == max_val])


def dominant_drive_from_attribution(
    attribution: dict[str, float],
    *,
    lead_tension: TensionEventV1 | None = None,
) -> str | None:
    """Argmax attribution; tie-break via primary_drive(lead_tension.kind)."""
    tied = _tied_at_max(attribution)
    if not tied:
        return None
    if len(tied) == 1:
        return tied[0]

    if lead_tension is not None:
        primary = primary_drive_for_tension_kind(
            lead_tension.kind,
            drive_impacts=lead_tension.drive_impacts,
        )
        if primary and primary in tied:
            return primary
        impacts = lead_tension.drive_impacts or {}
        ranked = sorted(
            ((drive, float(impacts.get(drive, 0.0))) for drive in tied),
            key=lambda item: (-item[1], item[0]),
        )
        if ranked and ranked[0][1] > 0.0:
            return ranked[0][0]

    return tied[0]


def select_lead_tension(tensions: Sequence[TensionEventV1]) -> TensionEventV1 | None:
    """Highest magnitude this tick; substrate gap preferred on magnitude tie."""
    if not tensions:
        return None

    def _sort_key(t: TensionEventV1) -> tuple[float, int, str]:
        gap_bias = 1 if t.kind == _GAP_KIND else 0
        return (-float(t.magnitude), -gap_bias, t.kind)

    return sorted(tensions, key=_sort_key)[0]
