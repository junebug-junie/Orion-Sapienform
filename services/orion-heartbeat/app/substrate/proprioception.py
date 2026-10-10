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
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .routing import ORGAN_SITE_MAP

# Wall-clock occupancy window. A 64-event COUNT window read rare organs as dark
# by sampling (biometrics ~11k/h vs cortex-orch ~26/h) -- see settings
# HEARTBEAT_ORGAN_FIRE_WINDOW_SEC.
FIRE_WINDOW_SEC = 300.0
# Hard memory bound per organ inside the window (oldest dropped first).
MAX_EVENTS_PER_ORGAN = 50_000
SMEAR_MIN = 0.5
# Absolute 0/0 guard only (both ends dead). NOT what keeps the reading honest:
# 1e-6 bits of near-end entropy still let far/near reach 1.46e6 live.
NEAR_FLOOR = 1e-6
# Near end counts as dead (smear absent) once it carries less than a tenth of
# the far end's entanglement, i.e. far/near > SMEAR_DEAD_RATIO. Derived from
# the live distribution, not picked: 7 days of persisted heartbeat_smear
# (substrate_attention_self_model, 2026-10-03..10, n=14,668) is bimodal on a
# log10 axis. Alive body: 94% of rows in 0.56..4.54 (median 2.62, p90 3.18),
# nothing between 4.54 and 5.12. Collapsed population: ~830 rows from ~18 up
# to 1.46e6, i.e. near ~ far/1e6. The least-populated log10 quarter-decade bin
# between them is [10, 17.8) with 9 rows (0.06%), so its lower edge, 10, is
# the cut. Scale-free on purpose: the absolute entropy scale moves with
# BOND_DIM/PHYS_DIM, the ratio does not. Re-derive from the same query if
# the lattice dimensions or dissipation change.
SMEAR_DEAD_RATIO = 10.0
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
    # Additive: lets a consumer tell "dark" from "just rare". None = never
    # seen since boot (not "never fires"). Empty window -> all dicts empty.
    organ_last_fired_at: dict[str, str | None] = field(default_factory=dict)
    organ_seconds_since_last_fire: dict[str, float | None] = field(default_factory=dict)
    fire_window_sec: float | None = None


@dataclass(frozen=True)
class FireSnapshot:
    window_sec: float
    counts: dict[str, int]
    last_fired_at: dict[str, str | None]
    seconds_since_last_fire: dict[str, float | None]
    # False until the process has been up for a full window: before that,
    # a quiet organ may just not have had time to fire yet.
    warm: bool = True


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
    cannot carry infinity as a real reading. "Dead" means relative to the
    far end (far/near > SMEAR_DEAD_RATIO), not only below the absolute 0/0
    guard: a near end at 1e-5 bits is dead, and dividing by it produced live
    readings of 1e4..1.5e6 that every consumer took as "hugely smeared".
    """
    if len(mean_profile) != _PROFILE_CUTS:
        raise ValueError(f"kick profiles must be 9-cut ratio vectors, got {len(mean_profile)}")
    near = (float(mean_profile[_NEAR_CUTS[0]]) + float(mean_profile[_NEAR_CUTS[1]])) / 2.0
    far = (float(mean_profile[_FAR_CUTS[0]]) + float(mean_profile[_FAR_CUTS[1]])) / 2.0
    if near < NEAR_FLOOR or far > SMEAR_DEAD_RATIO * near:
        return None, None
    smear = far / near
    return smear, smear >= SMEAR_MIN


def compute_proprioception(
    *,
    fire_counts: dict[str, int] | None,
    mean_profile: list[float],
    fire_snapshot: FireSnapshot | None = None,
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
    cold = fire_snapshot is not None and not fire_snapshot.warm
    if cold or sum(counts.values()) <= 0:
        # Empty window (idle) or warm-up after restart is unknown, not
        # "everyone is dark" and not a zero-count reading. Recency still
        # reported (null after a fresh boot) so silent-for-40-min is visible.
        return ProprioceptionV1(
            dark_seats=[],
            organ_fire_counts={},
            organ_distinctness=None,
            smear=smear,
            smeared=smeared,
            organ_last_fired_at=dict(fire_snapshot.last_fired_at) if fire_snapshot else {},
            organ_seconds_since_last_fire=(
                dict(fire_snapshot.seconds_since_last_fire) if fire_snapshot else {}
            ),
            fire_window_sec=fire_snapshot.window_sec if fire_snapshot else None,
        )
    return ProprioceptionV1(
        dark_seats=dark_seats(counts),
        organ_fire_counts=counts,
        organ_distinctness=occupancy_distinctness(counts),
        smear=smear,
        smeared=smeared,
        organ_last_fired_at=dict(fire_snapshot.last_fired_at) if fire_snapshot else {},
        organ_seconds_since_last_fire=(
            dict(fire_snapshot.seconds_since_last_fire) if fire_snapshot else {}
        ),
        fire_window_sec=fire_snapshot.window_sec if fire_snapshot else None,
    )


class OrganFireWindow:
    """Rolling wall-clock window of allowlisted absorbs. Unknown organs ignored.

    Pruning is by clock on every read, so an organ that stops talking ages out
    even when no events arrive. Memory is bounded by MAX_EVENTS_PER_ORGAN per
    organ. ``last_fire`` outlives pruning (one float per organ) so a consumer
    can see how long an organ has been quiet beyond the window.
    """

    def __init__(
        self,
        window_sec: float = FIRE_WINDOW_SEC,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        max_events_per_organ: int = MAX_EVENTS_PER_ORGAN,
    ) -> None:
        if window_sec <= 0:
            raise ValueError(f"window_sec must be > 0, got {window_sec}")
        if max_events_per_organ < 1:
            raise ValueError("max_events_per_organ must be >= 1")
        self.window_sec = float(window_sec)
        self._clock = clock
        self._wall_clock = wall_clock
        self._started = clock()
        self._events: dict[str, deque[float]] = {
            name: deque(maxlen=max_events_per_organ) for name in ORGAN_SITE_MAP
        }
        # (monotonic, wall) of last fire; survives pruning.
        self._last_fire: dict[str, tuple[float, float]] = {}

    def record(self, organ: str) -> None:
        if organ in ORGAN_SITE_MAP:
            now = self._clock()
            self._events[organ].append(now)
            self._last_fire[organ] = (now, self._wall_clock())

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        for q in self._events.values():
            while q and q[0] < cutoff:
                q.popleft()

    def counts(self) -> dict[str, int]:
        self._prune(self._clock())
        return {name: len(q) for name, q in self._events.items()}

    def snapshot(self) -> FireSnapshot:
        now = self._clock()
        self._prune(now)
        last_at: dict[str, str | None] = {}
        since: dict[str, float | None] = {}
        for name in ORGAN_SITE_MAP:
            fired = self._last_fire.get(name)
            if fired is None:
                last_at[name] = None
                since[name] = None
            else:
                mono, wall = fired
                last_at[name] = datetime.fromtimestamp(wall, timezone.utc).isoformat()
                since[name] = max(0.0, now - mono)
        return FireSnapshot(
            window_sec=self.window_sec,
            counts={name: len(q) for name, q in self._events.items()},
            last_fired_at=last_at,
            seconds_since_last_fire=since,
            warm=(now - self._started) >= self.window_sec,
        )
