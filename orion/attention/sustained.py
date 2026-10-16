"""Has a body signal stayed unusual for longer than one event can explain? Pure, stdlib only.

Spec: docs/superpowers/specs/2026-10-16-attention-continuous-signal-quota-design.md, option (b).

A signal that reads every tick (``node:substrate.biometrics``, ``node:substrate.bus_synaptic``)
is in the top decile of its own week ~10% of the time whatever happens, so "high now" alone is a
quota, not a verdict. The rule here: such a signal counts only when it has been hot for MORE than
``SUSTAIN_DWELL_SEC`` within the trailing ``SUSTAIN_WINDOW_SEC`` AND is hot now.

One home for that rule, three readers:

- the attention contests (``orion.attention.world_first.judge_candidate`` via the candidate's
  ``hot_dwell_sec`` / ``sustain_window_sec``),
- self-readings (``orion.attention.self_readings``, same judge on the same history),
- the metacog episode gate (``orion.metacog.self_signal_episode``: its open dwell is this
  module's threshold and comparison, and a continuous signal's dwell is this module's
  ``hot_dwell_sec`` over the same readings).

Kept free of every other ``orion`` import so the metacog episode module can import it without
pulling in the attention seam.
"""

from __future__ import annotations

from typing import Iterable

# How long one event's reading stays news. Derivation (225 isolated onsets over 7 days of
# substrate_node_prediction_error_history: one event's effect on the body is gone by 300 s) is
# recorded with its first user, orion.attention.world_first, which re-exports this value.
EVENT_ORIENTING_WINDOW_SEC: float = 300.0

# Unusual for longer than one event can explain: hot for MORE than this ...
SUSTAIN_DWELL_SEC: float = EVENT_ORIENTING_WINDOW_SEC
# ... within this trailing window (the metacog episode gate's open window: twice the orienting
# window, so a sustained run must cover more than half of it).
SUSTAIN_WINDOW_SEC: float = 2.0 * EVENT_ORIENTING_WINDOW_SEC


def hot_dwell_sec(
    readings: Iterable[tuple[float, bool | None]],
    *,
    now: float,
    window_sec: float = SUSTAIN_WINDOW_SEC,
    max_hold_sec: float | None = None,
) -> float:
    """Seconds of ``[now - window_sec, now]`` covered by hot readings.

    ``readings`` are ``(epoch seconds, hot)`` in any order. Each reading speaks for the body from
    its own time until the next reading (or ``now``), and for at most ``max_hold_sec``. ``hot``
    None (could not be judged) counts as not hot. A reading stamped after ``now`` (clock skew) is
    treated as taken at ``now``. Readings, not frames or polls: the dwell is a property of the
    signal's history, so every reader of the same history gets the same number.
    """
    pts = sorted((min(float(t), float(now)), bool(h)) for t, h in readings)
    lo = float(now) - float(window_sec)
    total = 0.0
    for i, (t, hot) in enumerate(pts):
        if not hot:
            continue
        end = pts[i + 1][0] if i + 1 < len(pts) else float(now)
        if max_hold_sec is not None:
            end = min(end, t + float(max_hold_sec))
        a, b = max(t, lo), min(end, float(now))
        if b > a:
            total += b - a
    return total


def is_sustained(
    dwell_sec: float | None, *, hot_now: bool, dwell_threshold_sec: float = SUSTAIN_DWELL_SEC
) -> bool:
    """Hot now AND hot for MORE than ``dwell_threshold_sec`` of the trailing window."""
    return bool(hot_now) and dwell_sec is not None and float(dwell_sec) > float(dwell_threshold_sec)
