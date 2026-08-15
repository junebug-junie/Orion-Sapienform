"""Tests for the feedback-credit integrity watch.

The load-bearing one is `test_rolling_window_catches_what_whole_series_misses`:
if that fails, this module has no reason to exist, because
`classify_producer_liveness` alone would do.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.attention.tension.liveness import classify_producer_liveness  # noqa: E402
from orion.field.credit_integrity import (  # noqa: E402
    SILENT,
    analyse_credit_integrity,
    load_credited_dimensions,
)
from orion.schemas.field_state import FieldStateV1  # noqa: E402

T0 = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
TICK = 2.0  # seconds; the live median is 2.04


def _state(i: int, *, pressure: float, backed: bool = True) -> FieldStateV1:
    """One tick. `backed=False` clears diffusion provenance for `pressure`, so
    collect_field_channel_pressures falls back to naming the capability."""
    cap = "capability:llm_inference"
    return FieldStateV1.model_validate(
        {
            "schema_version": "field.state.v1",
            "generated_at": (T0 + timedelta(seconds=i * TICK)).isoformat(),
            "tick_id": f"t{i}",
            "node_vectors": {"node:athena": {"availability": 1.0}},
            "capability_vectors": {cap: {"pressure": pressure}},
            "capability_provenance": ({cap: {"pressure": "node:athena"}} if backed else {}),
            "edges": [],
            "recent_perturbations": [],
        }
    )


DIMS = {"resource_pressure": ["pressure"]}


def test_policy_dimensions_map_to_real_channels():
    """The bug this module shipped with: the policy names DIMENSIONS and the
    merge returns CHANNELS. `resource_pressure` is fed by `pressure`, and
    looking it up directly in the channel dict yielded 0 samples while the
    report printed clean."""
    dims, window = load_credited_dimensions()
    assert dims["resource_pressure"] == ["pressure"]
    assert window > 0
    # Every credited dimension must resolve to at least one channel, or the
    # watch is blind to it.
    unmapped = [d for d, chans in dims.items() if not chans]
    assert not unmapped, unmapped


def test_unmappable_dimension_is_a_finding_not_silence():
    states = [_state(i, pressure=0.5) for i in range(40)]
    report = analyse_credit_integrity(states, {"nonexistent_dim": []}, 30.0)
    assert [f.kind for f in report.findings] == ["unmappable_dimension"]


def test_healthy_series_produces_no_findings():
    states = [_state(i, pressure=0.5 + (i % 3) * 0.01) for i in range(60)]
    report = analyse_credit_integrity(states, DIMS, 30.0)
    assert report.findings == []
    assert report.longest_unbacked_run["resource_pressure"] == 0


def test_rolling_window_catches_what_whole_series_misses():
    """The reason this module exists.

    A 60-second silent stretch inside a 4-minute series. The reused classifier
    called on the WHOLE series says `refreshed`, because one real write
    anywhere breaks monotonicity -- so a whole-history check cannot see an
    outage that spans the feedback loop's entire 30s sampling window.
    """
    states: list[FieldStateV1] = []
    value = 0.9
    for i in range(30):  # 60s of real writes
        value = 0.9 if i % 2 else 0.85
        states.append(_state(i, pressure=value))
    value = 0.9
    for i in range(30, 60):  # 60s of decay only
        value *= 0.92
        states.append(_state(i, pressure=value))
    for i in range(60, 120):  # writes resume
        states.append(_state(i, pressure=0.9 if i % 2 else 0.85))

    whole = classify_producer_liveness([0.9 if i % 2 else 0.85 for i in range(30)]
                                       + [0.9 * 0.92 ** (k + 1) for k in range(30)]
                                       + [0.9 if i % 2 else 0.85 for i in range(60)])
    assert whole == "refreshed", "precondition: the whole series looks healthy"

    report = analyse_credit_integrity(states, DIMS, 30.0)
    silent = [f for f in report.findings if f.kind == SILENT]
    assert silent, "the 60s silent stretch must be found"
    # Run at the classifier's supportable span, not the policy's 30s -- and the
    # report says so, so a clean result is not read as 30s coverage.
    assert report.shape_span_seconds >= 56.0
    assert silent[0].span_seconds >= report.shape_span_seconds - TICK
    assert report.whole_series_verdict["resource_pressure"] == "refreshed"


def test_a_short_dip_below_the_window_is_not_a_finding():
    """Live today: 55 unbacked events, all isolated single ticks against a 30s
    window. Those must not fire, or the watch cries wolf hourly."""
    states = []
    for i in range(120):
        states.append(_state(i, pressure=0.5, backed=(i % 40 != 0)))
    report = analyse_credit_integrity(states, DIMS, 30.0)
    assert [f for f in report.findings if f.kind == "unbacked_run"] == []
    assert report.longest_unbacked_run["resource_pressure"] == 1


def test_sustained_unbacked_run_is_a_finding():
    """Provenance covers the classifier's declared `at_floor` blind spot: a
    value pinned flat at the floor has no shape to read, but cleared provenance
    still says nobody contributed."""
    states = []
    for i in range(120):
        backed = not (30 <= i < 60)  # 60 seconds with nothing contributing
        states.append(_state(i, pressure=1e-320, backed=backed))
    report = analyse_credit_integrity(states, DIMS, 30.0)
    runs = [f for f in report.findings if f.kind == "unbacked_run"]
    assert runs, "a 60s unbacked stretch must be found"
    assert runs[0].span_seconds >= 30.0
    assert runs[0].tick_count == 30


def test_window_size_comes_from_the_policy_not_a_constant():
    """A hardcoded 30 would silently stop matching if the policy widened."""
    states = []
    value = 0.9
    for i in range(60):
        value = value * 0.92 if i >= 10 else 0.9
        states.append(_state(i, pressure=value))
    # The decay stretch (50 ticks = 100s) spans the classifier's floor, so it
    # fires even though the requested window is narrower than that floor.
    assert analyse_credit_integrity(states, DIMS, 20.0).findings
    # Window wider than the whole series: nothing can span it.
    assert not analyse_credit_integrity(states, DIMS, 10_000.0).findings


def test_findings_report_one_row_per_channel_not_per_overlapping_window():
    """Overlapping rolling windows would otherwise emit hundreds of near
    -identical rows for one outage and bury everything else."""
    states = []
    value = 0.9
    for i in range(200):
        value = value * 0.92 if i >= 10 else 0.9
        states.append(_state(i, pressure=value))
    report = analyse_credit_integrity(states, DIMS, 30.0)
    assert len([f for f in report.findings if f.kind == SILENT]) == 1


@pytest.mark.parametrize("n", [0, 1, 5])
def test_short_histories_do_not_raise(n):
    states = [_state(i, pressure=0.5) for i in range(n)]
    report = analyse_credit_integrity(states, DIMS, 30.0)
    assert report.findings == [] or all(f.kind == "unmappable_dimension" for f in report.findings)
