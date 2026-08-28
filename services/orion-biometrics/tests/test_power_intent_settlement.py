"""Settlement arithmetic and the honesty properties it has to preserve."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, SERVICE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.power_intent import settle, summarize  # noqa: E402

from orion.schemas.power import PowerIntentV1  # noqa: E402

T0 = datetime(2026, 8, 28, 5, 0, tzinfo=timezone.utc)


def _intent(**kw) -> PowerIntentV1:
    base = dict(
        intent_id="i1",
        workload_kind="reverie_diffusion",
        node="circe",
        gpu_index=2,
        expected_duration_sec=4.0,
        deadline=T0 + timedelta(seconds=60),
    )
    base.update(kw)
    return PowerIntentV1(**base)


class _Clock:
    """Deterministic clock so a 1 Hz window costs no wall time in tests."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _run(coro):
    return asyncio.run(coro)


def test_a_real_window_settles_with_peak_mean_and_energy() -> None:
    clock = _Clock(T0)
    readings = iter([42.0, 45.0, 210.0, 220.0, 180.0, 44.0])
    out = _run(
        settle(
            _intent(),
            sampler=lambda _g: next(readings, 44.0),
            now_fn=clock,
            sleep_fn=clock.sleep,
        )
    )
    assert out.outcome == "settled"
    assert out.baseline_watts == 42.0  # taken BEFORE the window opened
    assert out.actual_peak_watts == 220.0
    assert out.sample_count == 4
    assert out.actual_mean_watts == pytest.approx((45.0 + 210.0 + 220.0 + 180.0) / 4)
    assert out.energy_joules == pytest.approx(out.actual_mean_watts * 4.0)


def test_the_achieved_rate_is_reported_not_the_configured_one() -> None:
    """The measurement this whole design exists for: 4 of 332 real jobs were caught by
    the 31s sampler, so a reader must be able to see how well a window was resolved."""
    clock = _Clock(T0)
    out = _run(
        settle(
            _intent(expected_duration_sec=4.0),
            sampler=lambda _g: 100.0,
            sample_interval_sec=2.0,
            now_fn=clock,
            sleep_fn=clock.sleep,
        )
    )
    assert out.sample_count == 2
    assert out.achieved_sample_hz == pytest.approx(0.5)


def test_an_unreadable_gpu_settles_as_no_samples_not_as_zero_watts() -> None:
    """A sampler returning None means 'could not read'. Recording 0.0 W would claim the
    card drew nothing, which is the opposite assertion."""
    clock = _Clock(T0)
    out = _run(
        settle(_intent(), sampler=lambda _g: None, now_fn=clock, sleep_fn=clock.sleep)
    )
    assert out.outcome == "no_samples"
    assert out.sample_count == 0
    assert out.actual_peak_watts is None
    assert out.actual_mean_watts is None
    assert out.energy_joules is None
    assert out.residual_watts is None


def test_residual_is_none_when_nothing_was_expected() -> None:
    """The first intents a workload declares carry expected_watts=None deliberately.
    A residual against an unknown expectation would make them look perfectly predicted."""
    clock = _Clock(T0)
    out = _run(
        settle(_intent(expected_watts=None), sampler=lambda _g: 200.0,
               now_fn=clock, sleep_fn=clock.sleep)
    )
    assert out.expected_watts is None
    assert out.residual_watts is None


def test_residual_is_computed_once_an_expectation_exists() -> None:
    clock = _Clock(T0)
    out = _run(
        settle(_intent(expected_watts=180.0), sampler=lambda _g: 220.0,
               now_fn=clock, sleep_fn=clock.sleep)
    )
    assert out.residual_watts == pytest.approx(40.0)


def test_a_passed_deadline_closes_the_window_immediately() -> None:
    """A crashed workload must not leave the sampler running."""
    clock = _Clock(T0)
    out = _run(
        settle(
            _intent(deadline=T0 - timedelta(seconds=1)),
            sampler=lambda _g: 200.0,
            now_fn=clock,
            sleep_fn=clock.sleep,
        )
    )
    assert out.outcome == "deadline_expired"
    assert out.sample_count == 0


def test_the_deadline_bounds_a_long_declared_duration() -> None:
    """Declared duration is a request, not an authority."""
    clock = _Clock(T0)
    out = _run(
        settle(
            _intent(expected_duration_sec=600.0, deadline=T0 + timedelta(seconds=3)),
            sampler=lambda _g: 150.0,
            now_fn=clock,
            sleep_fn=clock.sleep,
        )
    )
    assert out.sample_count <= 3
    assert out.outcome in {"settled", "deadline_expired"}


def test_a_node_scoped_intent_is_reported_honestly_not_invented() -> None:
    """gpu_index=None means wall power, which this module does not measure. It must say
    so rather than fabricate a GPU window."""
    clock = _Clock(T0)
    out = _run(
        settle(_intent(gpu_index=None), sampler=lambda _g: 999.0,
               now_fn=clock, sleep_fn=clock.sleep)
    )
    assert out.outcome == "no_samples"
    assert out.actual_peak_watts is None


def test_summarize_is_pure_and_needs_no_clock() -> None:
    out = summarize(
        _intent(expected_watts=100.0),
        [110.0, 130.0],
        baseline=40.0,
        window_start=T0,
        window_end=T0 + timedelta(seconds=2),
        hit_deadline=False,
    )
    assert out.actual_peak_watts == 130.0
    assert out.actual_mean_watts == pytest.approx(120.0)
    assert out.energy_joules == pytest.approx(240.0)
    assert out.residual_watts == pytest.approx(30.0)
    assert out.achieved_sample_hz == pytest.approx(1.0)
