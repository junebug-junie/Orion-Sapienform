"""Deterministic 7-day prediction-error histories shaped like the live table (2026-10-14).

Live shapes (substrate_node_prediction_error_history, 7 days to 2026-10-14 03:26 UTC):
biometrics 0% zeros, p50 0.022, p90 0.077; bus_synaptic p50 0.032, p90 0.065; execution 73%
zeros, p90 0.53; chat 70% zeros; route 87% zeros, values in quarters; codebase 89% zeros;
cabinet 98.4% zeros (a trigger); strain news one 1.0 row per event (a trigger); harness_closure a constant 0.65 placeholder. Steps are coarser
than live so the tests stay fast; every node still clears the 200-reading history floor.

2026-11: bus_synaptic and route were retired (PE suite rebuild step 5) and left
this fixture; chat followed 2026-11-13 (step 6), replaced by node:prediction.chat_turn. The continuous body source that must be SUSTAINED to win
(orion/attention/sustained.py) is now biometrics, the remaining per-tick body
node; it was bus_synaptic before.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

NOW = datetime(2026, 10, 14, 3, 30, tzinfo=timezone.utc)
CONTINUOUS_BODY = "node:substrate.biometrics"
Row = tuple[str, datetime, float]


def _series(node: str, step_min: float, fn, *, now: datetime = NOW, days: float = 7.0) -> list[Row]:
    n = int(days * 24 * 60 / step_min)
    return [(node, now - timedelta(minutes=step_min * k), float(fn(k))) for k in range(n, 0, -1)]


# Six historical warming episodes (onsets) in the week, each 30 minutes of "on" rows.
_CABINET_BURSTS = (300, 600, 900, 1200, 1500, 1800)
# Eight historical strain news events in the week (one 1.0 row each).
_STRAIN_EVENTS = (150, 400, 650, 900, 1150, 1400, 1650, 1900)


def calm_rows(now: datetime = NOW) -> list[Row]:
    """Every node in its usual range; event-written nodes' last event long past."""
    rows: list[Row] = []
    rows += _series("node:substrate.biometrics", 5, lambda k: 0.005 + (k * 37 % 100) / 1400.0, now=now)
    rows += _series("node:substrate.execution", 10, lambda k: 0.0 if k % 4 else 0.2 + (k % 7) / 10.0, now=now)
    # chat-turn surprise (node:prediction.chat_turn, 2026-11-13; replaced node:substrate.chat): one
    # write per settled chat turn, ~80% exact 0.0 (live replay shape).
    rows += _series("node:prediction.chat_turn", 15, lambda k: 0.0 if k % 5 else (k * 23 % 100) / 120.0, now=now)
    rows += _series("node:substrate.codebase", 15, lambda k: 0.0 if k % 9 else (k % 10) / 10.0, now=now)
    # cabinet: a trigger. A few historical bursts (k in a 12-reading run), zero otherwise.
    rows += _series("node:substrate.cabinet", 5, lambda k: 0.7 if any(b <= k < b + 6 for b in _CABINET_BURSTS) else 0.0, now=now)
    rows += _series("node:substrate.harness_closure", 60, lambda k: 0.65, now=now)
    # strain news: a trigger. A 0.0 every tick, one 1.0 row per news event (budgeted, rare).
    rows += _series("node:substrate.strain", 5, lambda k: 1.0 if k in _STRAIN_EVENTS else 0.0, now=now)
    # forecast surprise (node:prediction.*, 2026-11-11): one write per scored outcome (~5 min),
    # ~60% exact 0.0, otherwise clip(z / 3) mostly below 0.55 (live replay shape).
    rows += _series("node:prediction.room", 5, lambda k: 0.0 if k % 5 < 3 else (k * 29 % 100) / 180.0, now=now)
    rows += _series("node:prediction.body", 5, lambda k: 0.0 if k % 5 < 3 else (k * 31 % 100) / 200.0, now=now)
    # perception is the world, not the body: never a self-reading.
    rows += _series("node:substrate.perception", 5, lambda k: 0.0 if k % 2 else 0.9, now=now)
    return rows


def busy_body_rows(now: datetime = NOW) -> list[Row]:
    """Calm history, then: an execution spike 30 s ago, biometrics at its week's top for the
    last ~9 minutes (biometrics reads every tick, so one hot reading is not enough: it must be
    sustained, orion/attention/sustained.py), and the cabinet warming alarm going off twice in
    the last hour (30 s cadence, like live)."""
    hour_ago = now - timedelta(hours=1)
    ten_min_ago = now - timedelta(minutes=10)
    # The cabinet's real cadence in the last hour: a row every 30 s, "on" (>0) while warming.
    rows = [
        r for r in calm_rows(now)
        if not (r[0] == "node:substrate.cabinet" and r[1] > hour_ago)
        and not (r[0] == CONTINUOUS_BODY and r[1] > ten_min_ago)
    ]
    rows.append(("node:substrate.execution", now - timedelta(seconds=30), 1.0))
    # biometrics at a ~60 s cadence: hot from 560 s ago to now (> 300 s of the last 600).
    for s in range(560, 0, -60):
        rows.append((CONTINUOUS_BODY, now - timedelta(seconds=s - 20), 0.2))
    for i in range(1, 120):  # 119 checks, two warming episodes -> 2 onsets, not 36 "firings"
        t = hour_ago + timedelta(seconds=30 * i)
        on = (10 * 60 <= 30 * i < 20 * 60) or (40 * 60 <= 30 * i < 44 * 60)
        rows.append(("node:substrate.cabinet", t, 0.8 if on else 0.0))
    return rows


def body_spike_rows(now: datetime = NOW) -> list[Row]:
    """Calm history at a 60 s cadence for the last 10 minutes on the continuous body node
    (biometrics), then ONE reading at its week's top 20 s ago: high for itself now, but a
    one-reading flicker."""
    ten_min_ago = now - timedelta(minutes=10)
    rows = [r for r in calm_rows(now) if not (r[0] == CONTINUOUS_BODY and r[1] > ten_min_ago)]
    for s in range(620, 80, -60):
        rows.append((CONTINUOUS_BODY, now - timedelta(seconds=s), 0.02))
    rows.append((CONTINUOUS_BODY, now - timedelta(seconds=20), 0.2))
    return rows
