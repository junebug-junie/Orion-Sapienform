"""Fail criteria for the neighbor-link probe.

Pre-reg: docs/research/preregistration/2026-09-20-heartbeat-neighbor-links.md

Pure scoring — no quimb. The replay harness imports these thresholds so the
live report cannot quietly invent a different pass/fail.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable

DELTA_MIN_ABS = 0.05
REL_DELTA_FRAC = 0.25
REL_DELTA_FLOOR = 1e-4
SCALE_SPLIT = 0.05
MIN_WINDOWS_PER_CELL = 10
WINDOW_SEC = 10.0
WINDOW_SEC_FALLBACK = 5.0
MIN_ROUTED_WINDOWS = 40
MIN_FIRE = 1
MIN_OTHER = 4
WINDOW_SEC_V2 = 5.0
DECIDING_PAIR: tuple[str, str] = ("orion-cortex-exec", "orion-bus")

# (organ_a, organ_b, site_a, site_b)
PAIRS: tuple[tuple[str, str, int, int], ...] = (
    ("orion-biometrics", "orion-cortex-exec", 1, 2),
    ("orion-cortex-exec", "orion-bus", 2, 3),
)

V0_ORGANS: tuple[str, ...] = (
    "orion-hub",
    "orion-biometrics",
    "orion-cortex-exec",
    "orion-bus",
    "orion-cortex-orch",
)


@dataclass(frozen=True)
class PairScore:
    organ_a: str
    organ_b: str
    site_a: int
    site_b: int
    n_cofire: int
    n_elsewhere: int
    mean_i_cofire: float
    mean_i_elsewhere: float
    delta: float
    holds: bool
    unverified: bool
    reason: str


@dataclass(frozen=True)
class NeighborVerdict:
    thermometer: bool
    relational: bool
    mixed: bool
    reason: str
    pairs: tuple[PairScore, ...]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def window_medians(
    counts_a: Iterable[int],
    counts_b: Iterable[int],
    counts_other: Iterable[int],
) -> tuple[float, float, float]:
    a = list(counts_a)
    b = list(counts_b)
    other = list(counts_other)
    if not a or not b or not other:
        raise ValueError("cannot median-split an empty window list")
    return float(median(a)), float(median(b)), float(median(other))


def classify_window(
    n_a: int,
    n_b: int,
    n_other: int,
    *,
    median_a: float,
    median_b: float,
    median_other: float,
) -> str:
    """v1 median split. Kept so the UNVERIFIED run stays reproducible."""
    if n_a >= median_a and n_b >= median_b:
        return "cofire"
    if n_a < median_a and n_b < median_b and n_other >= median_other:
        return "elsewhere"
    return "neither"


def classify_window_firing(n_a: int, n_b: int, n_other: int) -> str:
    """v2: quiet means quiet. Return cofire | elsewhere | neither."""
    if n_a >= MIN_FIRE and n_b >= MIN_FIRE:
        return "cofire"
    if n_a == 0 and n_b == 0 and n_other >= MIN_OTHER:
        return "elsewhere"
    return "neither"


def score_pair(
    *,
    organ_a: str,
    organ_b: str,
    site_a: int,
    site_b: int,
    i_cofire: list[float],
    i_elsewhere: list[float],
) -> PairScore:
    n_co = len(i_cofire)
    n_el = len(i_elsewhere)
    if n_co < MIN_WINDOWS_PER_CELL or n_el < MIN_WINDOWS_PER_CELL:
        return PairScore(
            organ_a=organ_a,
            organ_b=organ_b,
            site_a=site_a,
            site_b=site_b,
            n_cofire=n_co,
            n_elsewhere=n_el,
            mean_i_cofire=_mean(i_cofire),
            mean_i_elsewhere=_mean(i_elsewhere),
            delta=_mean(i_cofire) - _mean(i_elsewhere),
            holds=False,
            unverified=True,
            reason=(
                f"UNVERIFIED: need ≥{MIN_WINDOWS_PER_CELL} windows in each cell, "
                f"got cofire={n_co} elsewhere={n_el}"
            ),
        )
    mean_co = _mean(i_cofire)
    mean_el = _mean(i_elsewhere)
    delta = mean_co - mean_el
    threshold = pair_threshold(mean_co, mean_el)
    holds = delta >= threshold
    return PairScore(
        organ_a=organ_a,
        organ_b=organ_b,
        site_a=site_a,
        site_b=site_b,
        n_cofire=n_co,
        n_elsewhere=n_el,
        mean_i_cofire=mean_co,
        mean_i_elsewhere=mean_el,
        delta=delta,
        holds=holds,
        unverified=False,
        reason=(
            f"{'holds' if holds else 'thermometer-like'}: "
            f"delta={delta:.4f} (need ≥{threshold:.4f})"
        ),
    )


def pair_threshold(mean_co: float, mean_el: float) -> float:
    if max(mean_co, mean_el) >= SCALE_SPLIT:
        return DELTA_MIN_ABS
    return REL_DELTA_FRAC * max(mean_el, REL_DELTA_FLOOR)


def decide_probe(*, n_windows: int, pairs: list[PairScore]) -> NeighborVerdict:
    if n_windows < MIN_ROUTED_WINDOWS:
        return NeighborVerdict(
            thermometer=False,
            relational=False,
            mixed=False,
            reason=f"UNVERIFIED: fewer than {MIN_ROUTED_WINDOWS} routed windows",
            pairs=tuple(pairs),
        )
    if any(p.unverified for p in pairs):
        return NeighborVerdict(
            thermometer=False,
            relational=False,
            mixed=False,
            reason="UNVERIFIED: at least one pair lacked both cells",
            pairs=tuple(pairs),
        )
    holding = [p for p in pairs if p.holds]
    if len(holding) == len(pairs):
        return NeighborVerdict(
            thermometer=False,
            relational=True,
            mixed=False,
            reason="relational: both neighbor pairs moved with co-firing",
            pairs=tuple(pairs),
        )
    if not holding:
        return NeighborVerdict(
            thermometer=True,
            relational=False,
            mixed=False,
            reason="thermometer: neither neighbor pair beat busy-elsewhere",
            pairs=tuple(pairs),
        )
    names = " and ".join(f"{p.organ_a}/{p.organ_b}" for p in holding)
    return NeighborVerdict(
        thermometer=False,
        relational=False,
        mixed=True,
        reason=f"mixed: only {names} moved with co-firing",
        pairs=tuple(pairs),
    )


def decide_probe_v2(*, n_windows: int, pairs: list[PairScore]) -> NeighborVerdict:
    """v2: execution/bus decides. Biometrics/execution is reported, not headline."""
    if n_windows < MIN_ROUTED_WINDOWS:
        return NeighborVerdict(
            thermometer=False,
            relational=False,
            mixed=False,
            reason=f"UNVERIFIED: fewer than {MIN_ROUTED_WINDOWS} routed windows",
            pairs=tuple(pairs),
        )
    deciding = next(
        (p for p in pairs if (p.organ_a, p.organ_b) == DECIDING_PAIR),
        None,
    )
    if deciding is None:
        return NeighborVerdict(
            thermometer=False,
            relational=False,
            mixed=False,
            reason="UNVERIFIED: deciding pair missing from scores",
            pairs=tuple(pairs),
        )
    if deciding.unverified:
        return NeighborVerdict(
            thermometer=False,
            relational=False,
            mixed=False,
            reason="UNVERIFIED: deciding pair lacked both cells",
            pairs=tuple(pairs),
        )
    if deciding.holds:
        return NeighborVerdict(
            thermometer=False,
            relational=True,
            mixed=False,
            reason="relational: execution/bus link rose when those two talked",
            pairs=tuple(pairs),
        )
    return NeighborVerdict(
        thermometer=True,
        relational=False,
        mixed=False,
        reason="thermometer: biometrics traffic smeared into execution/bus as much as they talked",
        pairs=tuple(pairs),
    )
