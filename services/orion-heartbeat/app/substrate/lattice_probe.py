"""Fail criteria for the lattice-vs-thermometer probe.

Pre-reg: docs/research/preregistration/2026-09-19-heartbeat-lattice-vs-thermometer.md

Pure scoring — no quimb. The replay harness imports these thresholds so the
live report cannot quietly invent a different pass/fail.
"""
from __future__ import annotations

from dataclasses import dataclass

CYCLIC_ORGAN_SHIFT: dict[str, str] = {
    "orion-hub": "orion-biometrics",
    "orion-biometrics": "orion-cortex-exec",
    "orion-cortex-exec": "orion-bus",
    "orion-bus": "orion-cortex-orch",
    "orion-cortex-orch": "orion-hub",
}

REL_SHUFFLE_MAX = 0.05
MEAN_RATIO_DELTA_MAX = 0.01
STD_RATIO_DELTA_MAX = 0.005
KICK_SMEAR_MIN = 0.5
NEAR_FLOOR = 1e-6
MIN_ROUTED_ATOMS = 400
KICK_N = 50


@dataclass(frozen=True)
class ShuffleScore:
    d_shuffle: float
    rel_shuffle: float
    mean_ratio_a: float
    mean_ratio_b: float
    std_ratio_a: float
    std_ratio_b: float
    verdict_a: str
    verdict_b: str
    null: bool


@dataclass(frozen=True)
class KickScore:
    near: float
    far: float
    smear: float
    smeared: bool


@dataclass(frozen=True)
class ProbeVerdict:
    thermometer: bool
    reason: str
    shuffle: ShuffleScore
    kick: KickScore


def shuffle_source_service(source_service: str) -> str:
    """Cyclic organ permutation. Unknown organs pass through (caller skips)."""
    return CYCLIC_ORGAN_SHIFT.get(source_service, source_service)


def _rms(profile: list[float]) -> float:
    if not profile:
        return 0.0
    return float((sum(x * x for x in profile) / len(profile)) ** 0.5)


def _l2(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"profile length mismatch {len(a)} vs {len(b)}")
    return float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)


def score_shuffle(
    *,
    profile_a: list[float],
    profile_b: list[float],
    mean_ratio_a: float,
    mean_ratio_b: float,
    std_ratio_a: float,
    std_ratio_b: float,
    verdict_a: str,
    verdict_b: str,
) -> ShuffleScore:
    d = _l2(profile_a, profile_b)
    rms = _rms(profile_a)
    rel = 0.0 if rms == 0.0 else d / rms
    null = (
        rel < REL_SHUFFLE_MAX
        and verdict_a == verdict_b
        and abs(mean_ratio_a - mean_ratio_b) < MEAN_RATIO_DELTA_MAX
        and abs(std_ratio_a - std_ratio_b) < STD_RATIO_DELTA_MAX
    )
    return ShuffleScore(
        d_shuffle=d,
        rel_shuffle=rel,
        mean_ratio_a=mean_ratio_a,
        mean_ratio_b=mean_ratio_b,
        std_ratio_a=std_ratio_a,
        std_ratio_b=std_ratio_b,
        verdict_a=verdict_a,
        verdict_b=verdict_b,
        null=null,
    )


def score_kick(*, profile_before: list[float], profile_after: list[float]) -> KickScore:
    if len(profile_before) != 9 or len(profile_after) != 9:
        raise ValueError("kick profiles must be 9-cut ratio vectors")
    delta = [abs(a - b) for a, b in zip(profile_after, profile_before)]
    near = (delta[0] + delta[1]) / 2.0
    far = (delta[7] + delta[8]) / 2.0
    if near < NEAR_FLOOR:
        smear = float("inf")
        smeared = True
    else:
        smear = far / near
        smeared = smear >= KICK_SMEAR_MIN
    return KickScore(near=near, far=far, smear=smear, smeared=smeared)


def decide_probe(*, n_routed: int, shuffle: ShuffleScore, kick: KickScore) -> ProbeVerdict:
    if n_routed < MIN_ROUTED_ATOMS:
        return ProbeVerdict(
            thermometer=False,
            reason="UNVERIFIED: fewer than 400 routed atoms",
            shuffle=shuffle,
            kick=kick,
        )
    if shuffle.null and kick.smeared:
        return ProbeVerdict(
            thermometer=True,
            reason="thermometer: shuffle null and kick smeared",
            shuffle=shuffle,
            kick=kick,
        )
    if shuffle.null and not kick.smeared:
        return ProbeVerdict(
            thermometer=False,
            reason="mixed: shuffle null but kick still local",
            shuffle=shuffle,
            kick=kick,
        )
    if not shuffle.null and kick.smeared:
        return ProbeVerdict(
            thermometer=False,
            reason="mixed: shuffle moved but kick smeared",
            shuffle=shuffle,
            kick=kick,
        )
    return ProbeVerdict(
        thermometer=False,
        reason="lattice: shuffle moved and kick stayed local",
        shuffle=shuffle,
        kick=kick,
    )
