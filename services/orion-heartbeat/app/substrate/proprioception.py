"""Tick-level proprioception for the five heartbeat organs.

Dark seats and organ distinctness come from who actually talked in the
recent absorb window. Smear comes from the current ensemble mean entropy
profile (same near/far cuts as the lattice kick probe). Occupancy and
profile are independent inputs — one is a count, the other is geometry.

Not a heart rate. Not IIT. These answers are: which organs are silent,
whether coupling already looks the same far away as nearby, and whether
the seats are speaking as one blob or as distinct organs.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .routing import ORGAN_SITE_MAP

FIRE_WINDOW = 64
SMEAR_MIN = 0.5
NEAR_FLOOR = 1e-6
_NEAR_CUTS = (0, 1)
_FAR_CUTS = (7, 8)
_PROFILE_CUTS = 9


@dataclass(frozen=True)
class ProprioceptionV1:
    dark_seats: list[str]
    organ_fire_counts: dict[str, int]
    organ_distinctness: float | None
    smear: float | None
    smeared: bool | None


def dark_seats(fire_counts: dict[str, int]) -> list[str]:
    """Organs in ORGAN_SITE_MAP with zero (or missing) fires this window."""
    return sorted(
        name for name in ORGAN_SITE_MAP if int(fire_counts.get(name, 0) or 0) <= 0
    )


def occupancy_distinctness(fire_counts: dict[str, int]) -> float | None:
    """1 when one organ does all the talking, 0 when every organ talks equally.

    Shannon entropy of the occupancy distribution over the five allowlisted
    organs, divided by log(5), then flipped. Empty window is absent, not 0
    — 0 would mean "all seats equally busy," which is the opposite of silence.
    """
    counts = [max(0, int(fire_counts.get(name, 0) or 0)) for name in ORGAN_SITE_MAP]
    total = sum(counts)
    if total <= 0:
        return None
    n = len(counts)
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(n)))


def profile_smear(mean_profile: list[float]) -> tuple[float | None, bool | None]:
    """far/near entropy of the current 9-cut profile.

    Same cut indices as the lattice kick probe (near = cuts 0-1, far = 7-8).
    Live tick uses the profile itself, not a kick delta: if the far end is
    already at least half as entangled as the near end, the lattice looks
    smeared. Dead near-end is absent, not +inf — JSON and the self-model
    cannot carry infinity as a real reading.
    """
    if len(mean_profile) != _PROFILE_CUTS:
        raise ValueError(f"kick profiles must be 9-cut ratio vectors, got {len(mean_profile)}")
    near = (float(mean_profile[_NEAR_CUTS[0]]) + float(mean_profile[_NEAR_CUTS[1]])) / 2.0
    far = (float(mean_profile[_FAR_CUTS[0]]) + float(mean_profile[_FAR_CUTS[1]])) / 2.0
    if near < NEAR_FLOOR:
        return None, None
    smear = far / near
    return smear, smear >= SMEAR_MIN


def compute_proprioception(
    *,
    fire_counts: dict[str, int] | None,
    mean_profile: list[float],
) -> ProprioceptionV1:
    smear, smeared = profile_smear(mean_profile)
    if fire_counts is None:
        return ProprioceptionV1(
            dark_seats=[],
            organ_fire_counts={},
            organ_distinctness=None,
            smear=smear,
            smeared=smeared,
        )
    counts = {name: max(0, int(fire_counts.get(name, 0) or 0)) for name in ORGAN_SITE_MAP}
    return ProprioceptionV1(
        dark_seats=dark_seats(counts),
        organ_fire_counts=counts,
        organ_distinctness=occupancy_distinctness(counts),
        smear=smear,
        smeared=smeared,
    )


class OrganFireWindow:
    """Rolling last-N allowlisted absorbs. Unknown organs are ignored."""

    def __init__(self, maxlen: int = FIRE_WINDOW) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self._events: deque[str] = deque(maxlen=maxlen)

    def record(self, organ: str) -> None:
        if organ in ORGAN_SITE_MAP:
            self._events.append(organ)

    def counts(self) -> dict[str, int]:
        out = {name: 0 for name in ORGAN_SITE_MAP}
        for organ in self._events:
            out[organ] += 1
        return out
