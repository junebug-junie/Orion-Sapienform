"""Producer-liveness classification, including the defect it exists to catch."""
from __future__ import annotations

import pytest

from orion.attention.tension.liveness import (
    MIN_WINDOW_SAMPLES,
    classify_producer_liveness,
)


def _decay(start: float, n: int, ratio: float = 0.92) -> list[float]:
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * ratio)
    return out


def test_the_biometrics_outage_shape_is_detected():
    """The defect this module exists for: `orion-biometrics` goes quiet, every
    `resource_pressure` input decays untouched, the dimension reads calm, and
    `feedback_policy.v1.yaml` credits the in-flight action with a positive
    outcome for an outage. Nothing writes the channel; only the 0.92/tick decay
    loop touches it."""
    assert classify_producer_liveness(_decay(0.7, 60)) == "silent_producer"


def test_a_single_write_anywhere_proves_the_producer_is_alive():
    series = _decay(0.7, 60)
    series[30] = series[29] * 1.4  # one real perturbation mid-window
    assert classify_producer_liveness(series) == "refreshed"


def test_a_live_but_quiet_channel_is_not_a_silent_producer():
    """A producer writing the same low rest value repeatedly: non-increasing,
    but not decaying. This is genuine calm and must NOT read as an outage --
    the false positive that would make this instrument useless."""
    assert classify_producer_liveness([0.02] * 60) == "refreshed"


def test_noisy_live_channel_is_refreshed():
    series = [0.10, 0.12, 0.11, 0.13, 0.12, 0.11, 0.14, 0.10] * 8
    assert classify_producer_liveness(series) == "refreshed"


def test_bottomed_out_channel_reports_at_floor_not_a_confident_verdict():
    """Once a channel has finished decaying its shape carries no information.
    Saying "refreshed" here would be a false all-clear, and saying
    "silent_producer" would be unfounded -- both are worse than admitting the
    series cannot answer the question."""
    assert classify_producer_liveness([3e-323] * 60) == "at_floor"


def test_short_window_refuses_to_guess():
    """Judged over a short tail an ordinary quiet stretch is indistinguishable
    from an outage, so the honest answer is no answer."""
    assert classify_producer_liveness(_decay(0.7, MIN_WINDOW_SAMPLES - 1)) == "insufficient_data"


def test_partial_decay_that_has_not_shrunk_enough_is_not_yet_a_finding():
    """Monotonic but barely moved -- consistent with rest, not with an outage."""
    series = _decay(0.7, 40, ratio=0.999)
    assert classify_producer_liveness(series) == "refreshed"


@pytest.mark.parametrize("series", [[], [0.0] * 60, [-1.0] * 60])
def test_degenerate_input_makes_no_claim(series):
    assert classify_producer_liveness(series) in {"insufficient_data", "at_floor"}


def test_verdict_is_scale_free():
    """Same shape at three magnitudes must give the same verdict -- the absolute
    -threshold mistake `deviation_gate.py`'s original sigma_floor made."""
    for scale in (1.0, 1e-3, 1e3):
        assert classify_producer_liveness(_decay(0.7 * scale, 60)) == "silent_producer"
        assert classify_producer_liveness([0.02 * scale] * 60) == "refreshed"
