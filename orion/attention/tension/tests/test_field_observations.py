"""Observation extraction and the decay-artifact detector."""
from __future__ import annotations

import pytest

from orion.attention.tension.field_observations import (
    geometric_decay_ratio,
    iter_observations,
    subnormal_pinned,
)


def test_extracts_every_node_channel_pair():
    field_json = {
        "node_vectors": {
            "node:a": {"cpu_pressure": 0.5, "availability": 1.0},
            "node:b": {"cpu_pressure": 0.1},
        }
    }
    obs = list(iter_observations(field_json))
    assert {(o.node_id, o.channel) for o in obs} == {
        ("node:a", "cpu_pressure"),
        ("node:a", "availability"),
        ("node:b", "cpu_pressure"),
    }


def test_subnormal_is_coerced_to_clean_zero_and_flagged():
    """Live 2026-08-14: every `node:circe` channel reads at this magnitude."""
    obs = list(iter_observations({"node_vectors": {"node:circe": {"cpu_pressure": 3e-321}}}))
    assert len(obs) == 1
    assert obs[0].value == 0.0
    assert obs[0].coerced_subnormal is True


def test_genuine_zero_is_not_flagged_as_coerced():
    obs = list(iter_observations({"node_vectors": {"node:a": {"cpu_pressure": 0.0}}}))
    assert obs[0].value == 0.0
    assert obs[0].raw_value == 0.0
    assert obs[0].coerced_subnormal is False


def test_raw_value_survives_coercion_so_the_decay_probe_can_still_see_it():
    """Regression, self-review 2026-08-14. `value` is collapsed to 0.0 so a
    subnormal cannot pump variance into a baseline -- but `geometric_decay_ratio`
    rejects any series containing a non-positive value, so if the probe were fed
    `value` it could never fire on the decay artifact it exists to detect. It
    would report 'none' for a structural reason indistinguishable from clean data.
    """
    obs = list(iter_observations({"node_vectors": {"node:circe": {"cpu_pressure": 3e-321}}}))[0]
    assert obs.value == 0.0
    assert obs.raw_value == 3e-321

    # End to end at the magnitude actually named above: 3e-321, a SUBNORMAL.
    #
    # The first version of this test built its series from `1e-300` -- a normal
    # float -- while claiming to prove the property for the 3e-321 case two lines
    # up. It passed for the wrong reason and hid a real defect: subnormals carry
    # only ~10 significant bits, so a perfect 0.92 decay down here has a
    # successive-ratio spread of ~1.7e-3, far outside the 1e-6 ABSOLUTE tolerance
    # the probe originally used. Caught in review 2026-08-14; the tolerance is
    # now relative. Do not weaken this fixture back to a normal float.
    series = [3e-321 * (0.92**i) for i in range(10)]
    assert any(0.0 < abs(v) < 1e-308 for v in series), "fixture must exercise subnormals"
    raws = [
        list(iter_observations({"node_vectors": {"node:circe": {"cpu_pressure": v}}}))[0].raw_value
        for v in series
    ]
    assert geometric_decay_ratio(raws) == pytest.approx(0.92, rel=1e-2)


@pytest.mark.parametrize("bad", ["abc", None, float("nan"), float("inf")])
def test_non_numeric_and_non_finite_are_skipped(bad):
    assert list(iter_observations({"node_vectors": {"node:a": {"cpu_pressure": bad}}})) == []


@pytest.mark.parametrize("payload", [{}, {"node_vectors": None}, {"node_vectors": {"a": 5}}])
def test_malformed_payloads_yield_nothing(payload):
    assert list(iter_observations(payload)) == []


def test_detects_the_exact_geometric_decay_artifact():
    """The 0.92-per-tick generic staleness decay CLAUDE.md documents for
    `node:substrate.route`: a value heading to zero that looks identical to
    genuinely-calm-at-zero unless the successive ratio is checked."""
    series = [1.0]
    for _ in range(9):
        series.append(series[-1] * 0.92)
    assert geometric_decay_ratio(series) == pytest.approx(0.92)


def test_live_varying_channel_is_not_flagged_as_decay():
    assert geometric_decay_ratio([0.1, 0.2, 0.15, 0.3, 0.25, 0.4]) is None


def test_decay_detector_rejects_short_series_and_zeros():
    assert geometric_decay_ratio([1.0, 0.92, 0.85]) is None
    assert geometric_decay_ratio([1.0, 0.92, 0.0, 0.0, 0.0]) is None


def test_growth_is_not_decay():
    assert geometric_decay_ratio([1.0, 1.1, 1.21, 1.331, 1.4641]) is None


# ---------------------------------------------------------------------------
# Regression: the bottomed-out end state, invisible to the ratio probe (review)
# ---------------------------------------------------------------------------


def test_pinned_subnormal_is_reported_even_though_the_ratio_probe_cannot_see_it():
    """A series that has finished decaying has successive ratios of exactly 1.0,
    so `geometric_decay_ratio` rejects it (`mean >= 1.0`). That is the state most
    dead channels are actually in at any given moment -- live review found 15
    series in the `0 < v < 1e-300` band in a 30-minute window, none flagged."""
    pinned = [3e-323] * 8
    assert geometric_decay_ratio(pinned) is None, "precondition: ratio probe is blind here"
    assert subnormal_pinned(pinned) is True


def test_real_resting_channel_at_zero_is_not_reported_as_pinned():
    """Deliberate limit: a channel genuinely at rest reads exactly 0.0 too, and
    the two are indistinguishable without history predating the decay. Flagging
    zeros would trade a false-clean for a false-alarm across most of the channel
    set."""
    assert subnormal_pinned([0.0] * 8) is False


def test_live_channel_is_not_reported_as_pinned():
    assert subnormal_pinned([0.11, 0.12, 0.11, 0.13, 0.12, 0.11]) is False
    assert subnormal_pinned([3e-323, 3e-323, 0.5, 3e-323]) is False


def test_pinned_detector_needs_a_real_window():
    assert subnormal_pinned([3e-323, 3e-323]) is False
