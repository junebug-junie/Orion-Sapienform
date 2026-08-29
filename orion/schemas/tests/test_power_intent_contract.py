"""Contract tests for PowerIntentV1 / PowerIntentSettledV1.

These pin the distinctions that make a settlement honest. Every one of them is a
failure this repo has actually paid for somewhere else: a blind producer reporting a
full tank, a decayed metric reading as calm, an unmeasured value defaulting to zero and
becoming indistinguishable from a measured zero.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.power import PowerIntentSettledV1, PowerIntentV1
from orion.schemas.registry import SCHEMA_REGISTRY, resolve


def _now() -> datetime:
    return datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def test_both_schemas_are_registered_in_both_maps() -> None:
    """`SCHEMA_REGISTRY` is a kind lookup; `_REGISTRY` behind `resolve()` is what the
    bus actually validates against. A schema in only one is half-registered, and the
    half that is missing is silent."""
    for name, kind in (
        ("PowerIntentV1", "power.intent.v1"),
        ("PowerIntentSettledV1", "power.intent.settled.v1"),
    ):
        assert resolve(name).__name__ == name
        assert SCHEMA_REGISTRY[name].kind == kind
        assert SCHEMA_REGISTRY[name].model is resolve(name)


def test_expected_watts_defaults_to_unknown_not_zero() -> None:
    """A new workload has no measured expectation. None says so; 0.0 would claim it
    expects to draw nothing, and would then be fitted against as if it were data."""
    intent = PowerIntentV1(
        intent_id="i1",
        workload_kind="reverie_diffusion",
        node="circe",
        gpu_index=2,
        expected_duration_sec=8.0,
        deadline=_now() + timedelta(seconds=60),
    )
    assert intent.expected_watts is None
    assert intent.expected_watts != 0.0


def test_deadline_is_required() -> None:
    """Without a hard stop, a crashed workload leaves the sampler pinned at 1 Hz
    forever and its window never closes."""
    with pytest.raises(Exception):
        PowerIntentV1(
            intent_id="i1",
            workload_kind="reverie_diffusion",
            node="circe",
            expected_duration_sec=8.0,
        )


def test_no_samples_settles_as_unmeasured_not_as_zero_watts() -> None:
    """'We did not see' and 'we saw nothing drawn' are opposite claims. Collapsing them
    is how a blind instrument comes to read as a calm one."""
    settled = PowerIntentSettledV1(
        intent_id="i1",
        workload_kind="reverie_diffusion",
        node="circe",
        gpu_index=2,
        outcome="no_samples",
        window_start=_now(),
        window_end=_now() + timedelta(seconds=8),
        sample_count=0,
    )
    assert settled.outcome == "no_samples"
    assert settled.actual_peak_watts is None
    assert settled.actual_mean_watts is None
    assert settled.energy_joules is None
    assert settled.residual_watts is None


def test_residual_stays_none_when_the_expectation_was_unknown() -> None:
    """A residual against an unknown expectation is not a small error, it is a
    meaningless one. Zero here would make an unmeasured workload look perfectly
    predicted -- the most dangerous possible default."""
    settled = PowerIntentSettledV1(
        intent_id="i1",
        workload_kind="reverie_diffusion",
        node="circe",
        gpu_index=2,
        outcome="settled",
        window_start=_now(),
        window_end=_now() + timedelta(seconds=8),
        sample_count=8,
        achieved_sample_hz=1.0,
        actual_peak_watts=220.0,
        actual_mean_watts=140.0,
        baseline_watts=42.0,
        expected_watts=None,
    )
    assert settled.expected_watts is None
    assert settled.residual_watts is None


def test_a_settlement_reports_the_resolution_it_actually_achieved() -> None:
    """Measured live 2026-08-28: the standing 31s GPU sampler caught 4 of 332 real
    diffusion jobs. A settlement built from too few samples is arithmetic, not
    measurement, and these two fields are what let a reader tell which one they have."""
    settled = PowerIntentSettledV1(
        intent_id="i1",
        workload_kind="reverie_diffusion",
        node="circe",
        gpu_index=2,
        outcome="settled",
        window_start=_now(),
        window_end=_now() + timedelta(seconds=8),
        sample_count=2,
        achieved_sample_hz=0.25,
        actual_peak_watts=220.0,
        actual_mean_watts=131.0,
        baseline_watts=42.0,
    )
    assert settled.sample_count == 2
    assert settled.achieved_sample_hz == 0.25


def test_baseline_is_carried_so_the_delta_is_recoverable() -> None:
    """A 220 W peak on a card idling at 42 W is a different event from the same peak on
    a card idling at 200 W. The workload caused the delta, not the absolute."""
    settled = PowerIntentSettledV1(
        intent_id="i1",
        workload_kind="reverie_diffusion",
        node="circe",
        gpu_index=2,
        outcome="settled",
        window_start=_now(),
        window_end=_now() + timedelta(seconds=8),
        sample_count=8,
        achieved_sample_hz=1.0,
        actual_peak_watts=220.0,
        actual_mean_watts=140.0,
        baseline_watts=42.0,
        expected_watts=180.0,
        residual_watts=40.0,
    )
    assert settled.baseline_watts == 42.0
    assert settled.actual_peak_watts - settled.baseline_watts == pytest.approx(178.0)


def test_gpu_index_is_optional_so_a_whole_node_can_be_the_meter() -> None:
    """Not every workload is one card. None scopes the intent to the node's wall draw."""
    intent = PowerIntentV1(
        intent_id="i1",
        workload_kind="fleet_job",
        node="circe",
        expected_duration_sec=30.0,
        deadline=_now() + timedelta(seconds=90),
    )
    assert intent.gpu_index is None
