"""Observation extraction and the decay-artifact detector."""
from __future__ import annotations

import pytest

from orion.attention.tension.field_observations import (
    geometric_decay_ratio,
    iter_observations,
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

    # And end to end: a decaying series read through iter_observations is still
    # detectable, which is the property the measurement script depends on.
    series = [1e-300 * (0.92**i) for i in range(10)]
    raws = [
        list(iter_observations({"node_vectors": {"node:circe": {"cpu_pressure": v}}}))[0].raw_value
        for v in series
    ]
    assert geometric_decay_ratio(raws) == pytest.approx(0.92)


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
