"""Acceptance check 1 (unit) for the reverie PE-magnitude spec, step 1:
docs/superpowers/specs/2026-10-02-reverie-prediction-error-magnitude-proposal.md
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion.schemas.attention_frame import OpenLoopV1, PredictionErrorMagnitudeV1
from orion.schemas.registry import resolve
from orion.substrate import prediction_error_magnitude as pem
from orion.substrate.attention_broadcast import (
    broadcast_projection_from_frame,
    build_substrate_attention_frame,
)
from orion.substrate.prediction_error_magnitude import compute_prediction_error_magnitude

NOW = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)


def _series(values, *, step=timedelta(minutes=1), end=NOW):
    """Evenly spaced readings ending at `end`, oldest first."""
    n = len(values)
    return [(end - step * (n - 1 - i), float(v)) for i, v in enumerate(values)]


def test_insufficient_history_below_200_readings():
    hist = _series([0.1] * 199)
    mag = compute_prediction_error_magnitude(
        value=0.1, observed_at=NOW, history=hist, now=NOW
    )
    assert mag.n_readings_7d == 199
    assert mag.band == "insufficient_history"
    assert mag.trend == "insufficient_history"
    # Numbers are still reported; only the verdicts are withheld.
    assert mag.p50_7d == pytest.approx(0.1)


def test_exactly_200_readings_emits_a_band():
    mag = compute_prediction_error_magnitude(
        value=0.0, observed_at=NOW, history=_series([0.0] * 200), now=NOW
    )
    assert mag.band != "insufficient_history"


def test_all_zero_history_with_current_zero_reads_percentile_zero():
    mag = compute_prediction_error_magnitude(
        value=0.0, observed_at=NOW, history=_series([0.0] * 500), now=NOW
    )
    assert mag.percentile_now == 0.0
    assert mag.band == "quiet"
    assert mag.p50_7d == 0.0 and mag.p90_7d == 0.0
    assert mag.trend == "flat"


def test_percentile_is_share_strictly_below():
    # 75% zeros, 25% at 0.4: a 0.4 reading has 75% of history strictly below.
    hist = _series([0.0] * 300 + [0.4] * 100)
    mag = compute_prediction_error_magnitude(
        value=0.4, observed_at=NOW, history=hist, now=NOW
    )
    assert mag.percentile_now == pytest.approx(0.75)
    assert mag.band == "usual"


@pytest.mark.parametrize(
    "value,band",
    [(0.0, "quiet"), (0.6, "usual"), (0.95, "high"), (2.0, "unusual")],
)
def test_band_cut_points(value, band):
    # 1000 readings 0.000..0.999 -> percentile_now ~= value
    hist = _series([i / 1000 for i in range(1000)], step=timedelta(seconds=30))
    mag = compute_prediction_error_magnitude(
        value=value, observed_at=NOW, history=hist, now=NOW
    )
    assert mag.band == band


def test_falling_series_reads_settling():
    # Prior 24h sat around 0.5; the last hour has fallen to ~0.05.
    prior = _series([0.5] * 288, step=timedelta(minutes=5), end=NOW - timedelta(hours=1, minutes=5))
    recent = _series([0.05] * 12, step=timedelta(minutes=5), end=NOW)
    mag = compute_prediction_error_magnitude(
        value=0.05, observed_at=NOW, history=prior + recent, now=NOW
    )
    assert mag.median_prior_24h == pytest.approx(0.5)
    assert mag.median_1h == pytest.approx(0.05)
    assert mag.trend == "settling"


def test_rising_series_reads_rising():
    prior = _series([0.05] * 288, step=timedelta(minutes=5), end=NOW - timedelta(hours=1, minutes=5))
    recent = _series([0.5] * 12, step=timedelta(minutes=5), end=NOW)
    mag = compute_prediction_error_magnitude(
        value=0.5, observed_at=NOW, history=prior + recent, now=NOW
    )
    assert mag.trend == "rising"


def test_small_wobble_is_flat_under_spread_threshold():
    # Wide 7d spread (p90-p50 large) means a 0.05 move is not "rising".
    older = _series([0.0] * 400 + [0.8] * 200, step=timedelta(minutes=5), end=NOW - timedelta(hours=30))
    prior = _series([0.10] * 288, step=timedelta(minutes=5), end=NOW - timedelta(hours=1, minutes=5))
    recent = _series([0.15] * 12, step=timedelta(minutes=5), end=NOW)
    mag = compute_prediction_error_magnitude(
        value=0.15, observed_at=NOW, history=older + prior + recent, now=NOW
    )
    assert mag.trend == "flat"


def test_trend_min_delta_is_a_floor_on_the_threshold():
    prior = _series([0.10] * 288, step=timedelta(minutes=5), end=NOW - timedelta(hours=1, minutes=5))
    recent = _series([0.13] * 12, step=timedelta(minutes=5), end=NOW)
    hist = prior + recent
    rising = compute_prediction_error_magnitude(
        value=0.13, observed_at=NOW, history=hist, now=NOW, trend_min_delta=0.01
    )
    flat = compute_prediction_error_magnitude(
        value=0.13, observed_at=NOW, history=hist, now=NOW, trend_min_delta=0.05
    )
    assert rising.trend == "rising"
    assert flat.trend == "flat"


def test_stale_node_with_no_recent_readings_has_no_trend_and_real_age():
    observed = NOW - timedelta(hours=14)
    hist = _series([0.65] * 300, step=timedelta(minutes=1), end=observed)
    mag = compute_prediction_error_magnitude(
        value=0.65, observed_at=observed, history=hist, now=NOW
    )
    assert mag.age_sec == pytest.approx(14 * 3600)
    assert mag.median_1h is None
    assert mag.trend == "insufficient_history"


def test_24h_and_7d_ranges_are_separate():
    old = _series([0.9] * 300, step=timedelta(minutes=5), end=NOW - timedelta(days=3))
    today = _series([0.1] * 200, step=timedelta(minutes=5), end=NOW)
    mag = compute_prediction_error_magnitude(
        value=0.1, observed_at=NOW, history=old + today, now=NOW
    )
    assert mag.p50_24h == pytest.approx(0.1)
    assert mag.p90_24h == pytest.approx(0.1)
    assert mag.p50_7d == pytest.approx(0.9)
    assert mag.n_readings_7d == 500


def test_readings_older_than_7_days_are_ignored():
    stale = _series([0.9] * 300, end=NOW - timedelta(days=8))
    mag = compute_prediction_error_magnitude(
        value=0.1, observed_at=NOW, history=stale, now=NOW
    )
    assert mag.n_readings_7d == 0
    assert mag.percentile_now is None
    assert mag.p50_7d is None
    assert mag.band == "insufficient_history"


def test_does_not_import_the_reversion_trend_forecaster():
    """compute_prediction_error_trend flips the sign on purpose to forecast
    reversion; reusing it would describe a falling error as rising."""
    source = Path(pem.__file__).read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "orion.substrate.prediction_error_trend" not in imported
    assert "compute_prediction_error_trend" not in imported
    assert not hasattr(pem, "compute_prediction_error_trend")


def test_schema_registered_and_open_loop_field_defaults_none():
    assert resolve("PredictionErrorMagnitudeV1") is PredictionErrorMagnitudeV1
    loop = OpenLoopV1(id="l1", description="x")
    assert loop.magnitude is None
    # Old payloads without the field still validate.
    assert OpenLoopV1.model_validate({"id": "l1", "description": "x"}).magnitude is None


def test_open_loop_round_trips_a_magnitude():
    mag = compute_prediction_error_magnitude(
        value=0.2, observed_at=NOW, history=_series([0.1] * 250), now=NOW
    )
    loop = OpenLoopV1(id="l1", description="x", magnitude=mag)
    again = OpenLoopV1.model_validate(loop.model_dump(mode="json"))
    assert again.magnitude == mag


def _node(node_id, label, **metadata):
    return SimpleNamespace(
        node_id=node_id,
        label=label,
        metadata=metadata,
        signals=SimpleNamespace(confidence=0.8),
    )


def test_broadcast_attaches_magnitude_only_when_supplied():
    nodes = [
        _node(
            "node:substrate.chat",
            "Chat prediction error",
            dynamic_pressure=0.4,
            prediction_error=0.14,
            dynamic_pressure_reason="prediction_error_seed",
        )
    ]
    mag = compute_prediction_error_magnitude(
        value=0.14, observed_at=NOW, history=_series([0.0] * 250), now=NOW
    )
    without = build_substrate_attention_frame(nodes=nodes, min_salience=0.05, now=NOW)
    assert without.open_loops and all(l.magnitude is None for l in without.open_loops)

    with_mag = build_substrate_attention_frame(
        nodes=nodes,
        min_salience=0.05,
        now=NOW,
        magnitude_by_node_id={"node:substrate.chat": mag},
    )
    assert with_mag.open_loops[0].magnitude == mag
    # Same winner either way: magnitude is descriptive, not a ranking input.
    assert (
        with_mag.selected_action.open_loop_id == without.selected_action.open_loop_id
    )
    # The persisted projection (broadcast log = the trace) carries it.
    projection = broadcast_projection_from_frame(with_mag)
    dumped = projection.model_dump(mode="json")
    assert dumped["frame"]["open_loops"][0]["magnitude"]["band"] == mag.band


def test_magnitude_never_changes_ranking_with_multiple_competitors():
    """Give the LOWEST-pressure loop an 'unusual' magnitude and the highest a
    'quiet' one: loop order, salience and the winner must be identical."""
    nodes = [
        _node(f"node:substrate.{name}", f"{name} prediction error",
              dynamic_pressure=p, prediction_error=p,
              dynamic_pressure_reason="prediction_error_seed",
              contributing_turn_ids=[f"turn-{name}-{i}" for i in range(k)])
        for name, p, k in (("chat", 0.9, 3), ("execution", 0.5, 2), ("route", 0.1, 1))
    ]
    hist = _series([0.0] * 300)
    unusual = compute_prediction_error_magnitude(value=0.9, observed_at=NOW, history=hist, now=NOW)
    quiet = compute_prediction_error_magnitude(value=0.0, observed_at=NOW, history=hist, now=NOW)
    assert unusual.band == "unusual" and quiet.band == "quiet"
    without = build_substrate_attention_frame(nodes=nodes, min_salience=0.05, now=NOW)
    with_mag = build_substrate_attention_frame(
        nodes=nodes, min_salience=0.05, now=NOW,
        magnitude_by_node_id={
            "node:substrate.route": unusual,
            "node:substrate.execution": quiet,
            "node:substrate.chat": quiet,
        },
    )
    assert len(without.open_loops) == 3
    assert [l.id for l in with_mag.open_loops] == [l.id for l in without.open_loops]
    assert [l.salience for l in with_mag.open_loops] == [l.salience for l in without.open_loops]
    assert with_mag.selected_action.open_loop_id == without.selected_action.open_loop_id
    assert [a.model_dump() for a in with_mag.candidate_actions] == [
        a.model_dump() for a in without.candidate_actions
    ]
    by_ref = {l.source_refs[0]: l.magnitude for l in with_mag.open_loops}
    assert by_ref["node:substrate.route"] == unusual
