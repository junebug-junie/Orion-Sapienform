"""Power-intent prior pins.

The design claim is that a MEDIAN over a BOUNDED window is the right estimator
because the live dataset provably contains a superseded measurement regime.
These tests assert that claim against the real numbers rather than invented
ones -- see docs/superpowers/specs/2026-08-30-power-intent-prior-design.md.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.power_prior import PowerPrior

KEY = dict(workload_kind="reverie_diffusion", node="circe", gpu_index=2)

# The real 24 full-window peaks (mean 252.7, median 254.3, sd 8.0) are
# represented by their observed range; the two truncated-window rows are the
# actual values recorded before the window was recalibrated.
CLEAN = [238.2, 243.0, 243.7, 254.8, 254.3, 255.2, 257.9, 268.0]
CONTAMINATED = [48.5, 144.5]


def _feed(prior: PowerPrior, values, outcome: str = "settled") -> None:
    for v in values:
        prior.observe(outcome=outcome, actual_peak_watts=v, **KEY)


def test_unmeasured_workload_declares_none_not_a_guess():
    """PowerIntentV1's docstring: None means UNKNOWN and a fabricated constant
    would be baked into a dataset that is later fitted against."""
    prior = PowerPrior()
    assert prior.expected_watts(**KEY) is None


def test_below_min_samples_still_declares_none():
    prior = PowerPrior(min_samples=3)
    _feed(prior, CLEAN[:2])
    assert prior.sample_count(**KEY) == 2
    assert prior.expected_watts(**KEY) is None


def test_declares_once_min_samples_reached():
    prior = PowerPrior(min_samples=3)
    _feed(prior, CLEAN[:3])
    assert prior.expected_watts(**KEY) == pytest.approx(statistics.median(CLEAN[:3]))


def test_median_survives_the_real_contaminated_regime():
    """The load-bearing test. Two truncated-window rows (48.5, 144.5) are a
    different instrument, not outliers of the same process. A mean would be
    dragged well below the true centre; the median must not be."""
    prior = PowerPrior(window=20, min_samples=3)
    _feed(prior, CLEAN + CONTAMINATED)

    got = prior.expected_watts(**KEY)
    contaminated_mean = statistics.mean(CLEAN + CONTAMINATED)

    # Hand-computed, not asserted by feel:
    #   clean   n=8  mean 251.9  median 254.6  min 238.2
    #   pooled  n=10 mean 220.8  median 249.0
    # The mean is dragged 31.1W and lands BELOW the clean minimum -- i.e. two
    # bad points in ten put the estimate entirely outside the range the real
    # process ever produced. The median moves 5.6W and stays inside it.
    assert contaminated_mean < min(CLEAN), (
        f"mean {contaminated_mean:.1f} should fall outside the clean range"
    )
    assert min(CLEAN) <= got <= max(CLEAN), f"median {got} left the clean range"
    assert got == pytest.approx(statistics.median(CLEAN), abs=6.0)


def test_the_window_forgets_a_superseded_regime():
    """A bounded window is what makes this self-correcting. If the workload
    genuinely changes cost, old samples must age out rather than anchor it."""
    prior = PowerPrior(window=5, min_samples=3)
    _feed(prior, [50.0] * 5)
    assert prior.expected_watts(**KEY) == pytest.approx(50.0)

    _feed(prior, [250.0] * 5)  # regime change, exactly one window of new data
    assert prior.expected_watts(**KEY) == pytest.approx(250.0)
    assert prior.sample_count(**KEY) == 5


def test_unsettled_outcomes_never_contribute():
    """"We did not see" and "we saw nothing drawn" are opposite claims. The
    settlement schema keeps them distinct; this consumer must not collapse
    them by coercing a missing peak to a number."""
    prior = PowerPrior(min_samples=1)

    # Isolate the OUTCOME guard: a non-settled outcome carrying a real number
    # must still be refused. Passing None here instead would let the separate
    # missing-peak guard do all the work and this test would pass even with
    # the outcome check deleted -- confirmed by mutation, which is how this
    # weaker earlier version was caught.
    assert prior.observe(outcome="deadline_expired", actual_peak_watts=250.0, **KEY) is False
    assert prior.observe(outcome="no_samples", actual_peak_watts=250.0, **KEY) is False
    assert prior.observe(outcome="", actual_peak_watts=250.0, **KEY) is False
    assert prior.sample_count(**KEY) == 0

    # And independently, the missing-peak guard.
    assert prior.observe(outcome="settled", actual_peak_watts=None, **KEY) is False
    assert prior.sample_count(**KEY) == 0
    assert prior.expected_watts(**KEY) is None

    # Control: the same value WITH a settled outcome is accepted, so the
    # assertions above are refusals rather than a prior that never counts.
    assert prior.observe(outcome="settled", actual_peak_watts=250.0, **KEY) is True
    assert prior.expected_watts(**KEY) == pytest.approx(250.0)


def test_a_nonpositive_peak_is_a_broken_reading_not_an_idle_card():
    prior = PowerPrior(min_samples=1)
    assert prior.observe(outcome="settled", actual_peak_watts=0.0, **KEY) is False
    assert prior.observe(outcome="settled", actual_peak_watts=-5.0, **KEY) is False
    assert prior.expected_watts(**KEY) is None


def test_priors_do_not_leak_across_workload_or_card():
    """gpu_index is in the key because an index is only meaningful per host
    and different cards draw differently."""
    prior = PowerPrior(min_samples=3)
    _feed(prior, CLEAN[:3])

    assert prior.expected_watts(workload_kind="other_job", node="circe", gpu_index=2) is None
    assert prior.expected_watts(workload_kind="reverie_diffusion", node="athena", gpu_index=2) is None
    assert prior.expected_watts(workload_kind="reverie_diffusion", node="circe", gpu_index=0) is None
    assert prior.expected_watts(**KEY) is not None


def test_degenerate_config_is_rejected_at_construction():
    with pytest.raises(ValueError):
        PowerPrior(window=0)
    with pytest.raises(ValueError):
        PowerPrior(min_samples=0)
    with pytest.raises(ValueError):
        PowerPrior(window=3, min_samples=5)
