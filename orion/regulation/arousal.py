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
3. Any stale input (S1 past its own 300 s grace, S2 past 30 s, E1 query failed) -> unknown. Unknown is never idle.
   A GPU queue at or above its floor while the pool's lane controller is refusing to act
   (``gpu_fault``, see ``gpu_fault_from_events``) is not evidence of load: S2 reads like a stale
   input (``gpu_fault:<kind>:<seat>``), so it can neither strain nor clear. Live 2026-10-09: the
   controller refused every swap for 26 h (``config_unloadable``) and the jammed queue alone would
   have read strained for 20 h and held every optional decision.
   The same holds for queue entries a heat-reflex shed is holding back (``cabinet_hot``,
   ``cabinet_unknown``, ``cooling_incident``: ``orion.gpu_pool.shed.REFLEX_REASONS``; sql-writer
   counts only these): they wait on heat, which S1 already reads, not on GPU load. Orion's own
   ``orion_self_shed`` is NOT excluded: it fires below the reflex, where S1 does not read hot. Only the queue left after them is load; when they alone put the queue at or above the floor,
   S2 reads like a stale input (``gpu_shed:<reason>``), and a run of shed-held snapshots never counts
   toward ``gpu_sustain_sec``. Live 2026-10-18 -> 10-20: the shed held every queued background and
   system lease for ~26 of 54 h, and arousal read that queue as independent GPU strain.
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
from orion.gpu_pool.controller_health import classify as classify_controller_refusal
from orion.schemas.regulation import ArousalInputsV1, ArousalReadingV1

# GPU pool publishes GpuPoolStateV1 every 5 s; sql-writer lands each in ~10 ms (live 10-10).
# The spec's generic rule is 3x cadence (15 s), but live gaps between saved snapshots reach
# 20-40 s a few times a day (max 39.8 s over 6 h, 10-10), and a step landing in one would read
# unknown and restart a latched strain's clear clock. 30 s absorbs those gaps and is still
# 10x shorter than the 300 s sustain window (review finding).
GPU_STATE_CADENCE_SEC = 5.0
GPU_STATE_STALE_SEC = 30.0

DEFAULT_ENGAGED_MINUTES = 45.0
DEFAULT_GPU_QUEUE_FLOOR = 2
DEFAULT_GPU_SUSTAIN_SEC = 300.0
DEFAULT_CLEAR_SEC = 600.0

# The pool re-asks a refusing controller every swap_cooldown_sec (600 s); live 2026-10-09/10 the
# 155 refusals came a median 601 s apart, never more than 604 s. Three cadences: one missed or
# late refusal does not end a fault that is still there.
GPU_FAULT_LOOKBACK_SEC = 1800.0
# gpu_pool_events rows that say whether a seat's lane controller can act. A refusal on the pool's
# own action is ``actuate_refused``; a refusal on a status probe is ``swap_failed`` with reason
# ``status_refused:<reason>``. Only ``swapped`` (the controller did what was asked) ends a fault:
# other ``swap_failed`` rows (ack timeout, ambiguous reconcile, a failed action) do not prove the
# controller can read its config, so they neither trip nor clear.
GPU_FAULT_EVENTS = ("actuate_refused", "swap_failed", "swapped")
_STATUS_REFUSED = "status_refused:"


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _depth_sum(depth: object) -> Optional[int]:
    if not isinstance(depth, dict) or any(not isinstance(v, int) or v < 0 for v in depth.values()):
        return None
    return sum(depth.values())


def gpu_queue_evidence(
    snapshots: Sequence[tuple],
    now: datetime,
    *,
    floor: int = DEFAULT_GPU_QUEUE_FLOOR,
    max_gap_sec: float = GPU_STATE_STALE_SEC,
) -> tuple[Optional[float], Optional[int], Optional[float], int, Optional[str]]:
    """Saved GPU pool snapshots (ascending) -> S2 evidence.

    Each snapshot is ``(generated_at, queue_depth)`` or ``(generated_at, queue_depth,
    shed_held_depth, shed_reason)`` (``orion.gpu_pool.shed.shed_held_queue``, saved by sql-writer;
    None = not recorded, read as nothing held -- the reading before the shed columns existed).

    Returns ``(age_sec, depth, sustained_sec, held, shed_reason)``: the newest snapshot's age, its
    summed queue depth, how long the LOAD depth (queue minus shed-held) has held at or above
    ``floor`` back from the newest snapshot with no gap longer than ``max_gap_sec`` (a gap is
    missing evidence, so it breaks the run), and the newest snapshot's shed-held count and reason.
    All None (held 0) when there is no snapshot. A malformed depth map counts as no snapshot; a
    malformed held map counts as nothing held, and held never exceeds the queue.
    """
    now = _utc(now)
    rows: list[tuple[datetime, int, int, Optional[str]]] = []
    for snap in snapshots:
        ts, depth = snap[0], snap[1]
        total = _depth_sum(depth)
        if total is None:
            continue
        held = _depth_sum(snap[2]) if len(snap) > 2 and snap[2] is not None else None
        held = min(held or 0, total)
        reason = (snap[3] if len(snap) > 3 else None) if held else None
        rows.append((_utc(ts), total, held, reason))
    if not rows:
        return None, None, None, 0, None
    rows.sort(key=lambda r: r[0])
    newest_ts, newest_depth, newest_held, newest_reason = rows[-1]
    age = max(0.0, (now - newest_ts).total_seconds())
    if newest_depth - newest_held < floor:
        return age, newest_depth, 0.0, newest_held, newest_reason
    start = newest_ts
    for (ts, depth, held, _), (later_ts, *_rest) in zip(reversed(rows[:-1]), reversed(rows[1:])):
        if depth - held < floor or (later_ts - ts).total_seconds() > max_gap_sec:
            break
        start = ts
    return age, newest_depth, (newest_ts - start).total_seconds(), newest_held, newest_reason


def _fault_event(event: str, reason: Optional[str]) -> tuple[bool, Optional[str]]:
    """``(decides, kind)``: whether this row says anything about the controller, and the fault
    kind it reports (None = the controller answered)."""
    r = reason or ""
    if event == "swapped":
        # fault_cleared:discovery / fault_cleared:operator clear a CARD fault without the
        # controller saying anything (runtime._clear_faults, operator clear_fault): neutral.
        return (False, None) if r.startswith("fault_cleared:") else (True, None)
    if event == "actuate_refused":
        kind = classify_controller_refusal(r)
        return kind is not None, kind
    if event == "swap_failed" and r.startswith(_STATUS_REFUSED):
        kind = classify_controller_refusal(r[len(_STATUS_REFUSED):])
        return kind is not None, kind
    return False, None


def gpu_fault_from_events(
    events: Iterable[Tuple[datetime, str, Optional[str], Optional[str]]],
    now: datetime,
    *,
    lookback_sec: float = GPU_FAULT_LOOKBACK_SEC,
) -> Optional[str]:
    """Saved pool events ``(generated_at, event, reason, seat)`` -> ``"<kind>:<seat>"`` while a
    seat's lane controller is refusing to act, else None.

    Per seat, the newest deciding row inside ``lookback_sec`` wins: a refusal the pool's own
    ``orion.gpu_pool.controller_health.classify`` calls unfixable by retry (``config_unloadable``,
    ``launch_digest_mismatch``, ...) is a fault; a later ``swapped`` ends it (except
    ``fault_cleared:*``, which clears a card without a controller answer). Refusals a retry
    fixes (``busy``, ``upstream_not_idle``) and unknown reasons are neutral, as the pool treats
    them. Several faulted seats join with ``+``. Rows stamped after ``now`` are ignored.
    """
    now = _utc(now)
    newest: dict[str, tuple[datetime, Optional[str]]] = {}
    for ts, event, reason, seat in events:
        ts = _utc(ts)
        if ts is None or not seat:
            continue
        age = (now - ts).total_seconds()
        if age < 0 or age > lookback_sec:
            continue
        decides, kind = _fault_event(event, reason)
        if not decides:
            continue
        cur = newest.get(seat)
        # Same instant: the refusal wins, so the answer is independent of row order.
        if cur is None or ts > cur[0] or (ts == cur[0] and kind is not None):
            newest[seat] = (ts, kind)
    faults = sorted(f"{kind}:{seat}" for seat, (_, kind) in newest.items() if kind)
    return "+".join(faults) or None


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
    # 300 s grace); S2 is GPU_STATE_STALE_SEC (30 s); E1 is fresh whenever its query succeeded.
    s1_fresh = (inputs.cabinet_read_ok and inputs.cabinet_reflex != REFLEX_CABINET_UNKNOWN
                and inputs.cabinet_thermal_state not in (None, "unknown"))
    s1_hot = s1_fresh and inputs.cabinet_reflex == REFLEX_CABINET_HOT
    s2_seen = (inputs.gpu_state_age_sec is not None and inputs.gpu_state_age_sec <= GPU_STATE_STALE_SEC
               and inputs.gpu_queue_depth is not None)
    # A queue at/above the floor while a lane controller refuses to act measures the fault, not
    # load: it can neither strain nor clear. Below the floor nothing is waiting, which stays true.
    s2_faulted = s2_seen and bool(inputs.gpu_fault) and inputs.gpu_queue_depth >= gpu_queue_floor
    # Queue entries a shed (heat) holds back are not load either (S1 already reads the heat). Only
    # the rest is load; when the held entries alone lift the queue to the floor, S2 is no evidence.
    held = min(inputs.gpu_queue_shed_held or 0, inputs.gpu_queue_depth or 0) if s2_seen else 0
    load = (inputs.gpu_queue_depth - held) if s2_seen else None
    s2_shed = (s2_seen and not s2_faulted and held > 0 and inputs.gpu_queue_depth >= gpu_queue_floor
               and load < gpu_queue_floor)
    s2_fresh = s2_seen and not s2_faulted and not s2_shed
    s2_strain = s2_fresh and (inputs.gpu_queue_sustained_sec or 0.0) >= gpu_sustain_sec
    s2_clear = s2_fresh and load < gpu_queue_floor
    e1_fresh = inputs.juniper_turn_read_ok
    s2_stale = (f"gpu_fault:{inputs.gpu_fault}" if s2_faulted
                else f"gpu_shed:{inputs.gpu_shed_reason or 'unknown'}" if s2_shed else "stale:gpu_state")
    stale = [why for why, ok in (("stale:cabinet", s1_fresh), (s2_stale, s2_fresh),
                                 ("stale:juniper_turns", e1_fresh)) if not ok]

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
            reasons += [s for s in stale if s != "stale:juniper_turns"]

    if level is None:
        if stale:
            level = "unknown"
            reasons += stale
        else:
            minutes = inputs.minutes_since_juniper_turn
            if minutes is not None and minutes < engaged_minutes:
                level = "engaged"
                reasons.append("juniper_turn_recent")
            else:
                level = "idle"
                reasons.append(f"no_juniper_turn_{engaged_minutes:g}m")

    if held and not s2_shed and not s2_faulted:
        # Annotate only: part of the queue is shed-held, the rest still counted as load.
        reasons.append(f"gpu_shed_partial:{inputs.gpu_shed_reason or 'unknown'}:{held}")
    if inputs.gpu_fault and not s2_faulted:
        # Annotate only: the fault is real, but this step did not lean on the queue reading.
        reasons.append(f"gpu_fault:{inputs.gpu_fault}")
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
    "GPU_FAULT_EVENTS",
    "GPU_FAULT_LOOKBACK_SEC",
    "GPU_STATE_STALE_SEC",
    "classify_arousal",
    "gpu_fault_from_events",
    "gpu_queue_evidence",
    "level_seconds",
]
