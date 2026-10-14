"""Deterministic 7-day prediction-error histories shaped like the live table (2026-10-14).

Live shapes (substrate_node_prediction_error_history, 7 days to 2026-10-14 03:26 UTC):
biometrics 0% zeros, p50 0.022, p90 0.077; bus_synaptic p50 0.032, p90 0.065; execution 73%
zeros, p90 0.53; chat 70% zeros; route 87% zeros, values in quarters; codebase 89% zeros;
cabinet 98.4% zeros (a trigger); harness_closure a constant 0.65 placeholder. Steps are coarser
than live so the tests stay fast; every node still clears the 200-reading history floor.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

NOW = datetime(2026, 10, 14, 3, 30, tzinfo=timezone.utc)
Row = tuple[str, datetime, float]


def _series(node: str, step_min: float, fn, *, now: datetime = NOW, days: float = 7.0) -> list[Row]:
    n = int(days * 24 * 60 / step_min)
    return [(node, now - timedelta(minutes=step_min * k), float(fn(k))) for k in range(n, 0, -1)]


def calm_rows(now: datetime = NOW) -> list[Row]:
    """Every node in its usual range; event-written nodes' last event long past."""
    rows: list[Row] = []
    rows += _series("node:substrate.biometrics", 5, lambda k: 0.005 + (k * 37 % 100) / 1400.0, now=now)
    rows += _series("node:substrate.bus_synaptic", 5, lambda k: 0.01 + (k * 53 % 100) / 1700.0, now=now)
    rows += _series("node:substrate.execution", 10, lambda k: 0.0 if k % 4 else 0.2 + (k % 7) / 10.0, now=now)
    rows += _series("node:substrate.chat", 15, lambda k: 0.0 if k % 3 else 0.1 + (k % 9) / 10.0, now=now)
    rows += _series("node:substrate.route", 15, lambda k: 0.0 if k % 8 else 0.25 * (1 + k % 3), now=now)
    rows += _series("node:substrate.codebase", 15, lambda k: 0.0 if k % 9 else (k % 10) / 10.0, now=now)
    # cabinet: a trigger. A few historical bursts (k in a 12-reading run), zero otherwise.
    rows += _series("node:substrate.cabinet", 5, lambda k: 0.7 if 600 <= k < 612 or 1500 <= k < 1508 else 0.0, now=now)
    rows += _series("node:substrate.harness_closure", 60, lambda k: 0.65, now=now)
    # perception is the world, not the body: never a self-reading.
    rows += _series("node:substrate.perception", 5, lambda k: 0.0 if k % 2 else 0.9, now=now)
    return rows


def busy_body_rows(now: datetime = NOW) -> list[Row]:
    """Calm history, then: an execution spike 30 s ago, bus anomalies at their week's top, and
    the cabinet warming alarm firing through the last hour."""
    rows = calm_rows(now)
    rows.append(("node:substrate.execution", now - timedelta(seconds=30), 1.0))
    rows.append(("node:substrate.bus_synaptic", now - timedelta(seconds=20), 0.2))
    for m in range(55, 0, -5):
        rows.append(("node:substrate.cabinet", now - timedelta(minutes=m, seconds=-1), 0.8))
    return rows
