"""The camera surprise eval's pure parts, on real and synthetic data."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (HERE, REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_camera_surprise_input_eval import (  # noqa: E402
    distribution,
    evaluate,
    frame_scores,
    gate,
    tick_series,
)

T0 = datetime(2026, 10, 20, 6, 44, 20, tzinfo=timezone.utc)


def test_reconstruction_matches_the_live_log() -> None:
    """Five real cam0 rows from substrate_perception_embedding_baseline
    (2026-10-20 06:44:25..46, verbatim) against the scores substrate-runtime
    logged for the same frames (substrate_perception_prediction_error_scored):
    06:44:31 score=0.151 raw=0.00213, then 0.000, 0.000, 0.000."""
    raw_rows = [
        (1792478665.951621, 0.001485940442347584, 2.695045766082508e-07, 419361),
        (1792478671.266253, 0.0016141996169485003, 2.9785574063314236e-07, 419362),
        (1792478676.041558, 0.0015240569911179704, 2.789130574646285e-07, 419363),
        (1792478681.197959, 0.0014261222024389473, 2.710865601397277e-07, 419364),
        (1792478686.147050, 0.0014046514942947768, 2.1917420465284291e-07, 419365),
    ]
    rows = [(datetime.fromtimestamp(e, tz=timezone.utc), m, v, n) for e, m, v, n in raw_rows]
    scores = [s for _, s in frame_scores(rows)]
    assert scores[0] is None  # needs the previous row
    assert round(scores[1], 3) == 0.151
    assert scores[2:] == [0.0, 0.0, 0.0]


def _frames(values, start=T0, step=6.0):
    return [(start + timedelta(seconds=step * i), v) for i, v in enumerate(values)]


def test_latest_vs_peak_and_missing_ticks() -> None:
    frames = _frames([0.0, 1.0, 0.1, 0.0, 0.0])  # 0..24 s
    ticks = [T0 + timedelta(seconds=25), T0 + timedelta(seconds=55), T0 + timedelta(seconds=200)]
    latest, missing = tick_series(ticks, frames, mode="latest")
    peak, _ = tick_series(ticks, frames, mode="peak")
    assert latest == [(ticks[0], 0.0)]
    assert peak == [(ticks[0], 1.0)]
    # 55 s: newest frame 31 s old but none since the last tick; 200 s: stale.
    assert missing == {"stale": 1, "no_scored_frame": 1}


def test_gate_flags_a_ceiling_bound_series() -> None:
    assert all(gate(distribution([0.0] * 60 + [0.2] * 39 + [1.0])).values())
    bound = gate(distribution([0.2] * 85 + [1.0] * 15))
    assert not bound["can_rest"] and not bound["not_ceiling_bound"]


def test_evaluate_reports_alarms_and_gates_latest_only() -> None:
    rows = []
    n = 10
    t = T0
    for i in range(400):
        t = T0 + timedelta(seconds=6 * i) + (timedelta(hours=2) if i >= 200 else timedelta(0))
        rows.append((t, 0.0015, 3e-7, n + i))
    ticks = [T0 + timedelta(seconds=30 * k) for k in range(1, 330)]
    report = evaluate({"cam0": rows}, {"node:substrate.perception": ticks})
    entry = report["streams"]["cam0"]
    assert entry["latest"]["missing"]["stale"] > 0
    states = [a[0] for a in entry["alarms"]]
    assert states[:2] == ["red", "green"]
    assert report["ok"]
