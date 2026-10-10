"""Arousal, pure (Temporal Self rev 4, R3). No IO, no clock: ``now`` is always passed in.

One slow reading of Orion's situation, ``engaged`` / ``idle`` / ``strained`` / ``unknown``,
which existing dials will later read to set their own gain (spec order 6; no reader in this
patch). Arousal never acts on its own.

Rule, evaluated in this order (spec R3):

1. A FRESH strain input holds -- the cabinet reflex says ``cabinet_hot`` (immediate), or the GPU
   pool queue has held at or above its floor for ``gpu_sustain_sec`` (default 5 min) -> strained.
   One fresh, true strain input is enough even when others are stale.
2. Strain latched earlier holds until every strain input reads fresh AND clear for ``clear_sec``
   (default 10 min). Stale time never earns clear time: a latched step with a stale strain input
   reads ``unknown`` and restarts the clear clock.
3. Any input older than 3x its cadence -> unknown. Unknown is never idle.
4. A Juniper turn within ``engaged_minutes`` (the dream's DREAM_IDLE_MINUTES, 45) -> engaged.
5. Otherwise idle.

Two moves are immediate by design: a Juniper turn (idle -> engaged) and the heat reflex.

Excluded inputs and why (spec R3 metric gate): biometrics ``strain`` (a blend of S1 and S2),
Spark ``arousal`` (dead since 2026-07-28), the rest drive (it is meant to READ arousal; a loop),
and Orion's own speech or thought (Orion would arouse itself).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence, Tuple

from orion.autonomy.cabinet_heat import REFLEX_CABINET_HOT, REFLEX_CABINET_UNKNOWN
from orion.schemas.regulation import ArousalInputsV1, ArousalReadingV1

# GPU pool publishes GpuPoolStateV1 every 5 s; sql-writer lands each in ~10 ms (live 10-10).
GPU_STATE_CADENCE_SEC = 5.0
STALE_FACTOR = 3.0
GPU_STATE_STALE_SEC = GPU_STATE_CADENCE_SEC * STALE_FACTOR

DEFAULT_ENGAGED_MINUTES = 45.0
DEFAULT_GPU_QUEUE_FLOOR = 2
DEFAULT_GPU_SUSTAIN_SEC = 300.0
DEFAULT_CLEAR_SEC = 600.0


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def gpu_queue_evidence(
    snapshots: Sequence[Tuple[datetime, dict]],
    now: datetime,
    *,
    floor: int = DEFAULT_GPU_QUEUE_FLOOR,
    max_gap_sec: float = GPU_STATE_STALE_SEC,
) -> tuple[Optional[float], Optional[int], Optional[float]]:
    """Saved GPU pool snapshots (ascending ``(generated_at, queue_depth)``) -> S2 evidence.

    Returns ``(age_sec, depth, sustained_sec)``: the newest snapshot's age, its summed queue
    depth, and how long the depth has held at or above ``floor`` back from the newest snapshot
    with no gap longer than ``max_gap_sec`` (a gap is missing evidence, so it breaks the run).
    All None when there is no snapshot. A malformed depth map counts as no snapshot.
    """
    now = _utc(now)
    rows: list[tuple[datetime, int]] = []
    for ts, depth in snapshots:
        if not isinstance(depth, dict) or any(not isinstance(v, int) or v < 0 for v in depth.values()):
            continue
        rows.append((_utc(ts), sum(depth.values())))
    if not rows:
        return None, None, None
    rows.sort(key=lambda r: r[0])
    newest_ts, newest_depth = rows[-1]
    age = max(0.0, (now - newest_ts).total_seconds())
    if newest_depth < floor:
        return age, newest_depth, 0.0
    start = newest_ts
    for (ts, depth), (later_ts, _) in zip(reversed(rows[:-1]), reversed(rows[1:])):
        if depth < floor or (later_ts - ts).total_seconds() > max_gap_sec:
            break
        start = ts
    return age, newest_depth, (newest_ts - start).total_seconds()


def _usable_prev(prev: Optional[ArousalReadingV1], now: datetime, max_gap_sec: float) -> Optional[ArousalReadingV1]:
    """The previous reading carries hysteresis only across a short gap. After a long outage the
    old ``since`` and latch describe a past nobody observed, so they are dropped."""
    if prev is None:
        return None
    gap = (now - _utc(prev.observed_at)).total_seconds()
    if gap < 0 or gap > max_gap_sec:
        return None
    return prev


def classify_arousal(
    prev: Optional[ArousalReadingV1],
    inputs: ArousalInputsV1,
    *,
    enabled: bool = True,
    engaged_minutes: float = DEFAULT_ENGAGED_MINUTES,
    gpu_queue_floor: int = DEFAULT_GPU_QUEUE_FLOOR,
    gpu_sustain_sec: float = DEFAULT_GPU_SUSTAIN_SEC,
    clear_sec: float = DEFAULT_CLEAR_SEC,
    max_prev_gap_sec: float = 360.0,
) -> ArousalReadingV1:
    now = _utc(inputs.observed_at)
    evidence = dict(
        observed_at=now,
        minutes_since_juniper_turn=inputs.minutes_since_juniper_turn,
        cabinet_reflex=inputs.cabinet_reflex,
        cabinet_temp_c=inputs.cabinet_temp_c,
        gpu_queue_depth=inputs.gpu_queue_depth,
        gpu_queue_sustained_sec=inputs.gpu_queue_sustained_sec,
    )
    if not enabled:
        return ArousalReadingV1(arousal_level="unknown", since=now, reasons=["disabled"], **evidence)

    usable = _usable_prev(prev, now, max_prev_gap_sec)
    reasons: list[str] = [] if usable is not None or prev is None else ["prev_dropped:gap"]
    latched = usable.strain_latched if usable else False
    clear_since = _utc(usable.strain_clear_since) if usable else None

    # Freshness per input. S1's own module owns what a missing reading means (unknown past its
    # 300 s grace); S2 is 3x the pool's 5 s cadence; E1 is fresh whenever its query succeeded.
    s1_fresh = (inputs.cabinet_read_ok and inputs.cabinet_reflex != REFLEX_CABINET_UNKNOWN
                and inputs.cabinet_thermal_state not in (None, "unknown"))
    s1_hot = s1_fresh and inputs.cabinet_reflex == REFLEX_CABINET_HOT
    s2_fresh = (inputs.gpu_state_age_sec is not None and inputs.gpu_state_age_sec <= GPU_STATE_STALE_SEC
                and inputs.gpu_queue_depth is not None)
    s2_strain = s2_fresh and (inputs.gpu_queue_sustained_sec or 0.0) >= gpu_sustain_sec
    s2_clear = s2_fresh and inputs.gpu_queue_depth < gpu_queue_floor
    e1_fresh = inputs.juniper_turn_read_ok
    stale = [name for name, ok in (("cabinet", s1_fresh), ("gpu_state", s2_fresh), ("juniper_turns", e1_fresh)) if not ok]

    level: Optional[str] = None
    if s1_hot or s2_strain:
        level, latched, clear_since = "strained", True, None
        reasons += (["cabinet_hot"] if s1_hot else []) + (["gpu_queue_sustained"] if s2_strain else [])
    elif latched:
        if s1_fresh and s2_clear:
            clear_since = clear_since or now
            if (now - clear_since).total_seconds() >= clear_sec:
                latched, clear_since = False, None
                reasons.append("strain_cleared")
            else:
                level = "strained"
                reasons.append("strain_clearing")
        elif s1_fresh and s2_fresh:
            # Queue still at/above the floor but not yet sustained: not clear, still strained.
            level, clear_since = "strained", None
            reasons.append("strain_holding")
        else:
            level, clear_since = "unknown", None
            reasons += [f"stale:{s}" for s in stale if s != "juniper_turns"]

    if level is None:
        if stale:
            level = "unknown"
            reasons += [f"stale:{s}" for s in stale]
        else:
            minutes = inputs.minutes_since_juniper_turn
            if minutes is not None and minutes < engaged_minutes:
                level = "engaged"
                reasons.append("juniper_turn_recent")
            else:
                level = "idle"
                reasons.append(f"no_juniper_turn_{engaged_minutes:g}m")

    since = _utc(usable.since) if usable is not None and usable.arousal_level == level else now
    return ArousalReadingV1(
        arousal_level=level, since=since, strain_latched=latched, strain_clear_since=clear_since,
        reasons=reasons, **evidence,
    )


def level_seconds(readings: Iterable[ArousalReadingV1], end: datetime) -> dict[str, float]:
    """Seconds spent per level across consecutive readings (each holds until the next)."""
    out = {"engaged": 0.0, "idle": 0.0, "strained": 0.0, "unknown": 0.0}
    seq = list(readings)
    for cur, nxt in zip(seq, seq[1:] + [None]):
        stop = _utc(nxt.observed_at) if nxt is not None else _utc(end)
        out[cur.arousal_level] += max(0.0, (stop - _utc(cur.observed_at)).total_seconds())
    return out


__all__ = [
    "DEFAULT_CLEAR_SEC",
    "DEFAULT_ENGAGED_MINUTES",
    "DEFAULT_GPU_QUEUE_FLOOR",
    "DEFAULT_GPU_SUSTAIN_SEC",
    "GPU_STATE_STALE_SEC",
    "classify_arousal",
    "gpu_queue_evidence",
    "level_seconds",
]
