"""Cross-drive tension detection — pure function, no wiring.

`DriveEngine.update()` (see `drives.py`) produces two independent per-tick
dicts: `pressures` (0.0-1.0 per drive) and `activations` (bool per drive,
hysteresis-gated between `DriveMathConfig.activate_threshold` and
`DriveMathConfig.deactivate_threshold`). Each drive is scored on its own —
there is no notion today of two drives being in *tension* with each other
(e.g. `capability` dominating while `relational` sits suppressed — a
"heads-down, disengaged" pattern).

This module adds exactly one thing: a pure function that takes those same
two dicts and reports pairwise tensions between drives, using the existing
`DriveMathConfig.deactivate_threshold` as the "suppressed" bar (no new
magic number is introduced).

IMPORTANT — this is correlational co-occurrence, NOT a causal claim. A
`DriveTensionV1` says "drive A is active while drive B is currently low, at
the same tick." It does not claim A caused B's suppression, that the two
drives are mechanistically linked, or that the pattern will persist across
ticks. Treat it as a candidate signal for future inspection, not a verified
mechanism.

This module has zero side effects, is not imported by any live path
(`bus_worker.py`, `audit.py`, etc.), is not registered on the bus or in the
schema registry, and is not consumed anywhere yet. It exists to let the
definition be reviewed and tested on paper before anything wires it up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from orion.spark.concept_induction.drives import DRIVE_KEYS, DriveMathConfig

# Tension kind name for the one definition this module implements today.
# "inverse_coactivation": one drive is active while a different drive's
# pressure is simultaneously low (below the suppressed bar). Named
# specifically (not just "tension") because future patches may add other
# tension kinds (e.g. two drives both racing upward, or oscillation) and
# those must not be conflated with this one.
INVERSE_COACTIVATION = "inverse_coactivation"


@dataclass(frozen=True)
class DriveTensionV1:
    """One detected tension between two drives at a single tick.

    `drive_a` is the drive that is currently *active* (per the caller's
    `activations` dict). `drive_b` is the drive whose *pressure* is
    currently below the suppressed bar. The pair is directional: swapping
    the roles is a different (and not necessarily co-occurring) tension —
    see `detect_drive_tensions` for why both directions cannot hold at
    once for the same pair under DriveEngine-consistent inputs.
    """

    drive_a: str
    drive_b: str
    tension_kind: str
    magnitude: float


def detect_drive_tensions(
    pressures: Dict[str, float],
    activations: Dict[str, bool],
    *,
    cfg: DriveMathConfig | None = None,
) -> List[DriveTensionV1]:
    """Detect pairwise inverse-coactivation tensions between drives.

    Definition: a tension `(drive_a, drive_b)` is emitted when
    `activations[drive_a] is True` and `pressures[drive_b] < suppressed_bar`,
    where `suppressed_bar` is `DriveMathConfig.deactivate_threshold`
    (reused, not reinvented — this is the same level at which an already-
    active drive would itself deactivate, so "below this" is a principled
    reading of "low" for any drive, active or not).

    The boundary is exclusive: `pressures[drive_b] == suppressed_bar`
    exactly does NOT qualify. The threshold marks the level at which an
    active drive is still considered (barely) active; treating a drive
    sitting exactly on that line as "suppressed" would contradict that
    same semantics.

    `magnitude = pressures[drive_a] * (1.0 - pressures[drive_b])`: it rises
    with how strongly A is asserting itself (A's own pressure, not just its
    boolean activation) and with how thoroughly B is suppressed (1 minus
    B's pressure). Both terms live in [0, 1] so the product does too. This
    is the simplest formula that satisfies both properties; no other terms
    are folded in.

    Only pairs that cross the bar are emitted — the function does not
    enumerate all `len(DRIVE_KEYS) * (len(DRIVE_KEYS) - 1)` ordered pairs
    on every call, only the ones that actually qualify as tension.

    Drive names are not restricted to `DRIVE_KEYS`: any key present in
    both an "active" role and a "pressure" role is evaluated, so the
    function stays useful even if callers pass a subset or superset. Keys
    missing from one dict simply cannot participate in the role that dict
    supplies (e.g. a key absent from `activations` can never be `drive_a`).
    Missing/empty input degrades to an empty result rather than raising.
    """
    threshold = (cfg or DriveMathConfig()).deactivate_threshold

    tensions: List[DriveTensionV1] = []
    for drive_a, is_active in activations.items():
        if not is_active:
            continue
        pressure_a = pressures.get(drive_a)
        if pressure_a is None:
            # drive_a has no pressure reading at all; nothing to score with.
            continue
        for drive_b, pressure_b in pressures.items():
            if drive_b == drive_a:
                continue
            if pressure_b >= threshold:
                continue
            magnitude = max(0.0, min(1.0, pressure_a)) * (1.0 - max(0.0, min(1.0, pressure_b)))
            tensions.append(
                DriveTensionV1(
                    drive_a=drive_a,
                    drive_b=drive_b,
                    tension_kind=INVERSE_COACTIVATION,
                    magnitude=magnitude,
                )
            )
    return tensions
