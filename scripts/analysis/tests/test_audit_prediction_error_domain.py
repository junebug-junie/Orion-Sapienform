"""Unit tests for audit_prediction_error_domain.py's pure layer (no DB access).

Both regression cases mirror the two real 2026-07-26 bugs this script exists to
mechanize detection of: bus_synaptic's calm-floor bias (distribution never reads
near-zero) and route's decay artifact (an exact geometric ratio sustained for
48+ hours)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "audit_prediction_error_domain.py"
_spec = importlib.util.spec_from_file_location("audit_prediction_error_domain", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["audit_prediction_error_domain"] = mod
_spec.loader.exec_module(mod)

DomainSample = mod.DomainSample
compute_distribution = mod.compute_distribution
detect_decay_artifact = mod.detect_decay_artifact
render_report = mod.render_report

BASE = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _samples(values: list[float]) -> list[DomainSample]:
    return [
        DomainSample(observed_at=BASE + timedelta(seconds=2 * i), value=v)
        for i, v in enumerate(values)
    ]


def test_compute_distribution_empty_returns_none() -> None:
    assert compute_distribution([]) is None


def test_compute_distribution_basic_stats() -> None:
    report = compute_distribution(_samples([0.0, 0.5, 1.0]))
    assert report is not None
    assert report.n == 3
    assert report.mean == 0.5
    assert report.min_value == 0.0
    assert report.max_value == 1.0
    assert report.frac_exact_zero == 1 / 3
    assert report.frac_exact_one == 1 / 3
    assert report.distinct_values == 3


def test_zero_variance_detected() -> None:
    report = compute_distribution(_samples([0.27, 0.27, 0.27, 0.27]))
    assert report is not None
    assert report.stdev == 0.0
    assert report.distinct_values == 1


def test_always_zero_detected() -> None:
    report = compute_distribution(_samples([0.0] * 100))
    assert report is not None
    assert report.frac_exact_zero == 1.0


def test_no_decay_artifact_in_real_looking_noise() -> None:
    # Independent-looking values -- no two consecutive ratios should match to
    # 9 decimal places, so no suspicious run should be detected.
    values = [0.1, 0.34, 0.02, 0.5, 0.19, 0.61, 0.03, 0.44, 0.12, 0.29]
    report = detect_decay_artifact(_samples(values))
    assert report.longest_suspicious_run < 20


def test_decay_artifact_detected_matching_route_bug_shape() -> None:
    """Reproduces route's real bug shape: a starting value multiplied by an
    exact 0.92 ratio every tick, unbroken, for far longer than the detection
    threshold -- this is the case the script exists to catch."""
    values = [0.5]
    for _ in range(49):
        values.append(values[-1] * 0.92)
    report = detect_decay_artifact(_samples(values))
    assert report.longest_suspicious_run >= 20
    assert report.suspicious_ratio is not None
    assert abs(report.suspicious_ratio - 0.92) < 1e-9


def test_decay_run_resets_when_a_real_value_breaks_it() -> None:
    """A genuine fresh measurement in the middle of a decay run must reset the
    run counter -- otherwise a single real event masks an ongoing decay bug
    instead of correctly splitting it into two shorter, sub-threshold runs."""
    decaying = [0.5]
    for _ in range(10):
        decaying.append(decaying[-1] * 0.92)
    # A real event resets the value, then decay resumes with a fresh ratio.
    decaying.append(0.9)
    for _ in range(10):
        decaying.append(decaying[-1] * 0.92)
    report = detect_decay_artifact(_samples(decaying))
    # Each sub-run is only 10 long -- below the 20-tick suspicious threshold,
    # even though 20+ total decay-like transitions occurred across the series.
    assert report.longest_suspicious_run < 20


def test_zero_values_do_not_count_as_a_ratio_step() -> None:
    """A transition into or out of exact 0.0 must not be treated as a ratio
    (division by/into zero is meaningless here, not a decay signal)."""
    values = [0.5, 0.0, 0.5, 0.0, 0.5]
    report = detect_decay_artifact(_samples(values))
    assert report.longest_suspicious_run == 0
    assert report.suspicious_ratio is None


def test_held_flat_value_is_not_a_decay_artifact() -> None:
    """Regression test for a real false positive caught by this script's own
    first live smoke test: a long run of ratio==1.0 (value genuinely
    unchanged, e.g. a hold-until-stale channel sitting calm) must NOT be
    flagged -- that is the correct post-decay.py-fix behavior, not decay."""
    report = detect_decay_artifact(_samples([0.264] * 249))
    assert report.longest_suspicious_run == 0
    assert report.suspicious_ratio is None


def test_a_single_held_flat_tick_does_not_fragment_a_real_decay_run() -> None:
    """Regression test for a review-caught false negative: a single ratio==1.0
    tick landing in the middle of an otherwise-continuous 0.92 decay run must
    NOT reset run continuity. Before this fix, one flat blip could fragment
    a genuinely long decay run into two shorter sub-runs, each potentially
    falling below the detection threshold and hiding a real bug."""
    values = [0.5]
    for _ in range(19):
        values.append(values[-1] * 0.92)
    values.append(values[-1])  # one flat (ratio == 1.0) tick, mid-run
    for _ in range(19):
        values.append(values[-1] * 0.92)
    report = detect_decay_artifact(_samples(values))
    # 38 real 0.92 transitions total, uninterrupted by the flat blip -- must
    # still cross the 20-tick threshold as a single run, not two 19-tick ones.
    assert report.longest_suspicious_run >= 20
    assert report.suspicious_ratio is not None
    assert abs(report.suspicious_ratio - 0.92) < 1e-9


def test_small_sample_caveat_shown_before_degenerate_flags() -> None:
    """Review-caught gap: a single 0.0 sample trivially reads 'ALWAYS ZERO'
    (100% of 1 tick), which would be misleading without a caveat that the
    sample size is too small to trust. A newly-added domain with sparse
    history should not be reported as confidently degenerate off one tick."""
    samples = _samples([0.0])
    dist = compute_distribution(samples)
    decay = detect_decay_artifact(samples)
    report = render_report("new_domain", "node:substrate.new_domain", dist, decay, window_hours=1.0)
    assert "only 1 sample" in report
    assert "unreliable" in report


def test_representative_sample_size_has_no_caveat() -> None:
    samples = _samples([0.0] * 50)
    dist = compute_distribution(samples)
    decay = detect_decay_artifact(samples)
    report = render_report("calm_domain", "node:substrate.calm_domain", dist, decay, window_hours=1.0)
    assert "unreliable" not in report
