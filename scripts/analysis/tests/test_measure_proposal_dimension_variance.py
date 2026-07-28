"""Deterministic unit tests for measure_proposal_dimension_variance.py.

No DB, no network. Everything here is pure: population mean/variance, the
decay-artifact streak detector, the producer-liveness (real-event) detector,
and the classify_dimension bands all operate on plain synthetic data. Same
sibling-module-by-file-path loading pattern as
test_measure_attention_salience_normalization.py.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_proposal_dimension_variance.py"
_spec = importlib.util.spec_from_file_location("measure_proposal_dimension_variance", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_proposal_dimension_variance"] = mod
_spec.loader.exec_module(mod)

BASE = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)


def _ts(seconds: list[float]) -> list[datetime]:
    return [BASE + timedelta(seconds=s) for s in seconds]


# ===========================================================================
# population_mean_variance
# ===========================================================================


def test_population_mean_variance_normal_series() -> None:
    mean, variance = mod.population_mean_variance([0.1, 0.2, 0.3, 0.4, 0.5])
    assert abs(mean - 0.3) < 1e-9
    assert variance > 0.0


def test_population_mean_variance_empty_is_zero() -> None:
    assert mod.population_mean_variance([]) == (0.0, 0.0)


def test_population_mean_variance_flat_series_is_zero_variance() -> None:
    mean, variance = mod.population_mean_variance([1.0, 1.0, 1.0])
    assert abs(mean - 1.0) < 1e-9
    assert variance == 0.0


# ===========================================================================
# detect_decay_artifact -- the node:substrate.route class of signature.
# ===========================================================================


def test_detect_decay_artifact_finds_real_geometric_decay_streak() -> None:
    """A real ratio=0.92 decay streak (BIOMETRICS_FIELD_DECAY_RATE's live
    default) must be detected and its ratio reported accurately."""
    values = [1.0]
    for _ in range(30):
        values.append(values[-1] * 0.92)
    finding = mod.detect_decay_artifact(values)
    assert finding.suspected is True
    assert finding.streak_len >= mod.DECAY_ARTIFACT_MIN_STREAK
    assert abs(finding.ratio_mean - 0.92) < 1e-6
    assert finding.matches_known_decay_rate is True


def test_detect_decay_artifact_short_streak_not_suspected() -> None:
    # Only a few decreasing ticks -- well below DECAY_ARTIFACT_MIN_STREAK.
    values = [1.0, 0.9, 0.8, 0.85, 0.9, 0.7, 0.75]
    finding = mod.detect_decay_artifact(values)
    assert finding.suspected is False


def test_detect_decay_artifact_noisy_real_signal_not_flagged() -> None:
    """A genuinely noisy real signal (varying ratios, not a tight geometric
    sequence) must NOT be flagged, even across a long series -- this is the
    key false-positive check for this detector."""
    import random

    rng = random.Random(42)
    values = [0.5]
    for _ in range(200):
        values.append(max(0.01, values[-1] + rng.uniform(-0.05, 0.05)))
    finding = mod.detect_decay_artifact(values)
    # A real random walk should not hold a tight, near-constant decreasing
    # ratio for DECAY_ARTIFACT_MIN_STREAK+ consecutive ticks.
    assert finding.suspected is False


def test_detect_decay_artifact_ratio_not_matching_known_rate_still_reported() -> None:
    values = [1.0]
    for _ in range(30):
        values.append(values[-1] * 0.5)  # a real decay, but not the 0.92 rate
    finding = mod.detect_decay_artifact(values)
    assert finding.suspected is True
    assert finding.matches_known_decay_rate is False


# ===========================================================================
# detect_producer_liveness -- distinguishes real-but-sparse events from the
# actual node:substrate.route "zero real events ever" disease.
# ===========================================================================


def test_detect_producer_liveness_no_events_at_all() -> None:
    values = [1.0, 0.92, 0.92**2, 0.92**3]
    timestamps = _ts([0, 2, 4, 6])
    finding = mod.detect_producer_liveness(timestamps, values, timestamps[-1])
    assert finding.no_real_events_in_window is True
    assert finding.n_upward_transitions == 0


def test_detect_producer_liveness_real_events_present_even_if_sparse() -> None:
    """Regression guard for the real false-positive this script's first
    version produced: a dimension with real, sparse (not frequent) upward
    transitions must NOT be flagged as having no real events, no matter how
    long ago the most recent one was relative to some external threshold --
    only a literal absence of any real event anywhere in the window
    disqualifies."""
    values = [0.5, 0.46, 0.7, 0.64, 0.9, 0.83]
    timestamps = _ts([0, 100, 200, 2000, 4000, 6000])
    finding = mod.detect_producer_liveness(timestamps, values, timestamps[-1])
    assert finding.no_real_events_in_window is False
    assert finding.n_upward_transitions == 2  # 0.46->0.7 and 0.64->0.9
    assert finding.mean_inter_event_gap_sec is not None
    assert finding.mean_inter_event_gap_sec > 0


def test_detect_producer_liveness_reports_recency_and_mean_gap() -> None:
    values = [0.5, 0.9, 0.85, 0.95]
    timestamps = _ts([0, 10, 20, 30])
    window_end = timestamps[-1] + timedelta(seconds=5)
    finding = mod.detect_producer_liveness(timestamps, values, window_end)
    assert finding.n_upward_transitions == 2  # 0->10 and 20->30
    assert finding.seconds_since_last_upward_transition == 5.0
    assert finding.mean_inter_event_gap_sec == 20.0  # gap between the two events


# ===========================================================================
# classify_dimension -- explicit documented bands.
# ===========================================================================


def _stats(
    *,
    n_present: int = 100,
    variance: float = 0.01,
    decay_suspected: bool = False,
    no_real_events: bool = False,
) -> "mod.DimensionStats":
    decay = mod.DecayArtifactFinding(suspected=decay_suspected, streak_len=25 if decay_suspected else 0)
    liveness = mod.LivenessFinding(
        n_upward_transitions=0 if no_real_events else 5,
        last_upward_transition_ts=None,
        seconds_since_last_upward_transition=None,
        mean_inter_event_gap_sec=None,
        no_real_events_in_window=no_real_events,
    )
    return mod.DimensionStats(
        n_present=n_present,
        n_window=n_present,
        mean=0.5,
        variance=variance,
        stddev=variance**0.5,
        min_value=0.0,
        max_value=1.0,
        degenerate=variance <= mod.DEGENERATE_VARIANCE_EPS,
        decay_artifact=decay,
        liveness=liveness,
    )


def test_classify_dimension_insufficient_data() -> None:
    stats = _stats(n_present=3)
    assert mod.classify_dimension(stats) == "INSUFFICIENT_DATA"


def test_classify_dimension_degenerate_flat() -> None:
    stats = _stats(variance=0.0)
    assert mod.classify_dimension(stats) == "DEGENERATE_FLAT"


def test_classify_dimension_no_real_events_disqualifies_even_with_variance() -> None:
    """Real variance alone is not enough -- if the ONLY movement is decay
    (zero real upward events anywhere), this must still be disqualified."""
    stats = _stats(variance=0.01, no_real_events=True)
    assert mod.classify_dimension(stats) == "NO_REAL_EVENTS_IN_WINDOW"


def test_classify_dimension_active_with_decay_mechanic_when_events_are_real() -> None:
    """The key correction this script went through: a decay-ratio streak is
    NOT disqualifying on its own when real events genuinely occur in the
    window."""
    stats = _stats(variance=0.01, decay_suspected=True, no_real_events=False)
    assert mod.classify_dimension(stats) == "ACTIVE_WITH_DECAY_MECHANIC"


def test_classify_dimension_non_degenerate_usable_no_decay_streak() -> None:
    stats = _stats(variance=0.01, decay_suspected=False, no_real_events=False)
    assert mod.classify_dimension(stats) == "NON_DEGENERATE_USABLE"
