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


# --- persistence (2026-11-03) ------------------------------------------------------------------
# The GPU pool relaunches this container for almost every generation. With an
# in-memory-only prior, each relaunch started empty and the intent carried
# expected_watts=None (0 of 9 intents on 2026-11-02, ~1,000 settlements on file).

from app.power_prior import load_prior_file, save_prior_file  # noqa: E402

DAY = 86400.0
T0 = 1_762_000_000.0


def test_window_survives_a_save_and_load(tmp_path):
    path = str(tmp_path / "state" / "power_prior.json")
    first = PowerPrior(min_samples=3, max_age_sec=7 * DAY)
    for i, v in enumerate(CLEAN[:3]):
        first.observe(outcome="settled", actual_peak_watts=v, at=T0 + i, **KEY)
    assert save_prior_file(first, path)

    relaunched = PowerPrior(min_samples=3, max_age_sec=7 * DAY)
    assert relaunched.expected_watts(**KEY, now=T0 + 10) is None  # empty before loading
    assert load_prior_file(relaunched, path, now=T0 + 10) == 3
    assert relaunched.expected_watts(**KEY, now=T0 + 10) == pytest.approx(statistics.median(CLEAN[:3]))


def test_samples_older_than_max_age_do_not_count(tmp_path):
    prior = PowerPrior(min_samples=3, max_age_sec=7 * DAY)
    for i, v in enumerate(CLEAN[:3]):
        prior.observe(outcome="settled", actual_peak_watts=v, at=T0 + i, **KEY)
    assert prior.expected_watts(**KEY, now=T0 + 6 * DAY) is not None
    # After a gap longer than the age limit: unknown again, never a stale guess.
    assert prior.expected_watts(**KEY, now=T0 + 8 * DAY) is None
    path = str(tmp_path / "p.json")
    save_prior_file(prior, path)
    later = PowerPrior(min_samples=3, max_age_sec=7 * DAY)
    assert load_prior_file(later, path, now=T0 + 8 * DAY) == 0


def test_loaded_window_stays_bounded_and_keeps_the_newest(tmp_path):
    big = PowerPrior(window=20, min_samples=3)
    for i in range(20):
        big.observe(outcome="settled", actual_peak_watts=100.0 + i, at=T0 + i, **KEY)
    path = str(tmp_path / "p.json")
    save_prior_file(big, path)
    small = PowerPrior(window=5, min_samples=3)
    load_prior_file(small, path, now=T0 + 30)
    assert small.sample_count(**KEY) == 5
    assert small.expected_watts(**KEY) == pytest.approx(117.0)  # median of 115..119


@pytest.mark.parametrize("content", ["", "not json", "[]", '{"version": 99, "buckets": []}',
                                     '{"version": 1, "buckets": [{"workload_kind": "x", "node": "circe", '
                                     '"gpu_index": "2", "samples": [[200, 1]]}]}',
                                     '{"version": 1, "buckets": [{"samples": [["a", 1], [-5, 1], [200]]}]}'])
def test_a_bad_state_file_is_an_empty_prior_not_a_crash(tmp_path, content):
    path = tmp_path / "p.json"
    path.write_text(content, encoding="utf-8")
    prior = PowerPrior(min_samples=1)
    assert load_prior_file(prior, str(path), now=T0) == 0
    assert prior.expected_watts(**KEY) is None


def test_missing_file_and_disabled_path_are_quiet(tmp_path):
    prior = PowerPrior()
    assert load_prior_file(prior, str(tmp_path / "absent.json"), now=T0) == 0
    assert load_prior_file(prior, "", now=T0) == 0
    assert save_prior_file(prior, "") is False
