"""Service-layer behavior of the two generative gates.

Review finding S3 (2026-07-30): the de-dupe / staleness / cooldown-retry logic in
`_evaluate_insight_gate` / `_evaluate_flow_gate` / `_generative_samples_are_fresh`
is the load-bearing correctness claim of this feature and originally had no test
at all -- every test was pure-detector or pure-builder. These are the tests that
would have caught the double-fire bug (M2) and the stale-window bug (M1).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import EquilibriumService, settings
from orion.substrate.metacog_trigger_signals import ConfidenceSample

TICK = timedelta(seconds=30)


def _service() -> EquilibriumService:
    svc = EquilibriumService()
    svc.bus = MagicMock()
    svc.bus.publish = AsyncMock()
    # Keep the cooldown out of the way unless a test is specifically about it.
    return svc


def _now_series(values: list[float]) -> list[ConfidenceSample]:
    """Samples ending 'now', so the freshness guard passes."""
    end = datetime.now(timezone.utc)
    n = len(values)
    return [
        ConfidenceSample(generated_at=end - (n - 1 - i) * TICK, value=v)
        for i, v in enumerate(values)
    ]


# A real-shaped recovery: drops into the low band, climbs, holds high.
_RECOVERY_VALUES = [0.95, 0.66, 0.80, 0.91, 0.92, 0.93]
# 20 ticks of tight calm at/above the 0.90 floor.
_FLOW_VALUES = [0.93, 0.94, 0.93, 0.95, 0.94] * 4


@pytest.fixture(autouse=True)
def _open_cooldowns(monkeypatch):
    for attr in (
        "metacog_cooldown_sec",
        "metacog_insight_cooldown_sec",
        "metacog_flow_cooldown_sec",
    ):
        monkeypatch.setattr(settings, attr, 0.0)


# ===========================================================================
# insight de-dupe (M2 regression at the service layer)
# ===========================================================================


@pytest.mark.asyncio
async def test_insight_fires_once_then_dedupes_on_repeat_polls() -> None:
    svc = _service()
    samples = _now_series(_RECOVERY_VALUES)

    await svc._evaluate_insight_gate(samples, zen_state="zen", pressure=0.1)
    await svc._evaluate_insight_gate(samples, zen_state="zen", pressure=0.1)
    await svc._evaluate_insight_gate(samples, zen_state="zen", pressure=0.1)

    assert svc.bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_insight_does_not_refire_when_the_high_run_breaks_and_reforms() -> None:
    """The exact double-fire the review reproduced: a single sub-threshold tick
    re-anchors `high_at`, so keying on it published one recovery twice. Keying on
    `low_at` must publish once."""
    svc = _service()
    series = [0.95, 0.66, 0.80, 0.91, 0.92, 0.93, 0.89, 0.91, 0.92, 0.93]
    all_samples = _now_series(series)

    # Every sliding window a real poll loop would see.
    for end in range(3, len(all_samples) + 1):
        window = all_samples[max(0, end - 20) : end]
        await svc._evaluate_insight_gate(window, zen_state="zen", pressure=0.1)

    assert svc.bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_a_genuinely_new_recovery_fires_again() -> None:
    """De-dupe must not be so strong that a second real episode is swallowed."""
    svc = _service()

    await svc._evaluate_insight_gate(
        _now_series(_RECOVERY_VALUES), zen_state="zen", pressure=0.1
    )
    assert svc.bus.publish.call_count == 1

    # A new low crossing => a new low_at => a new episode.
    await svc._evaluate_insight_gate(
        _now_series([0.92, 0.64, 0.85, 0.93, 0.94]), zen_state="zen", pressure=0.1
    )
    assert svc.bus.publish.call_count == 2


@pytest.mark.asyncio
async def test_cooldown_suppressed_insight_stays_retryable(monkeypatch) -> None:
    """Review finding S1: the de-dupe key must be recorded only on a real
    publish. Otherwise a cooldown-suppressed episode is marked seen and, because
    the key is stable, never retried."""
    monkeypatch.setattr(settings, "metacog_insight_cooldown_sec", 9999.0)
    svc = _service()
    samples = _now_series(_RECOVERY_VALUES)

    # Burn the lane so the next fire is cooldown-suppressed.
    svc._last_trigger_ts_by_kind["insight"] = datetime.now().timestamp()
    await svc._evaluate_insight_gate(samples, zen_state="zen", pressure=0.1)
    assert svc.bus.publish.call_count == 0
    assert svc._last_insight_low_at is None, "suppressed fire must not be recorded"

    # Cooldown elapses; the same episode is still eligible.
    monkeypatch.setattr(settings, "metacog_insight_cooldown_sec", 0.0)
    await svc._evaluate_insight_gate(samples, zen_state="zen", pressure=0.1)
    assert svc.bus.publish.call_count == 1
    assert svc._last_insight_low_at is not None


@pytest.mark.asyncio
async def test_insight_does_not_fire_on_a_calm_window() -> None:
    svc = _service()
    await svc._evaluate_insight_gate(
        _now_series([0.93] * 10), zen_state="zen", pressure=0.1
    )
    assert svc.bus.publish.call_count == 0


# ===========================================================================
# flow
# ===========================================================================


@pytest.mark.asyncio
async def test_flow_fires_on_a_sustained_plateau() -> None:
    svc = _service()
    await svc._evaluate_flow_gate(
        _now_series(_FLOW_VALUES), zen_state="zen", pressure=0.1
    )
    assert svc.bus.publish.call_count == 1
    published = svc.bus.publish.call_args[0][1]
    assert published.payload["trigger_kind"] == "flow"


@pytest.mark.asyncio
async def test_flow_dedupes_the_same_newest_row() -> None:
    """Poll cadence and write cadence drift, so the same newest row is polled
    twice; that must not double-publish."""
    svc = _service()
    samples = _now_series(_FLOW_VALUES)

    await svc._evaluate_flow_gate(samples, zen_state="zen", pressure=0.1)
    await svc._evaluate_flow_gate(samples, zen_state="zen", pressure=0.1)

    assert svc.bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_cooldown_suppressed_flow_stays_retryable(monkeypatch) -> None:
    monkeypatch.setattr(settings, "metacog_flow_cooldown_sec", 9999.0)
    svc = _service()
    samples = _now_series(_FLOW_VALUES)

    svc._last_trigger_ts_by_kind["flow"] = datetime.now().timestamp()
    await svc._evaluate_flow_gate(samples, zen_state="zen", pressure=0.1)
    assert svc.bus.publish.call_count == 0
    assert svc._last_flow_ended_at is None

    monkeypatch.setattr(settings, "metacog_flow_cooldown_sec", 0.0)
    await svc._evaluate_flow_gate(samples, zen_state="zen", pressure=0.1)
    assert svc.bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_flow_does_not_fire_when_a_tick_dips_below_the_floor() -> None:
    svc = _service()
    values = list(_FLOW_VALUES)
    values[7] = 0.85
    await svc._evaluate_flow_gate(_now_series(values), zen_state="zen", pressure=0.1)
    assert svc.bus.publish.call_count == 0


# ===========================================================================
# Freshness guard (M1 regression at the service layer)
# ===========================================================================


def test_stale_window_is_rejected() -> None:
    """A frozen window keeps satisfying both conditions forever, so the poll
    loop must reject it before either gate sees it."""
    svc = _service()
    old_end = datetime.now(timezone.utc) - timedelta(days=3)
    stale = [
        ConfidenceSample(generated_at=old_end - (19 - i) * TICK, value=0.93)
        for i in range(20)
    ]
    assert svc._generative_samples_are_fresh(stale) is False


def test_fresh_window_is_accepted() -> None:
    svc = _service()
    assert svc._generative_samples_are_fresh(_now_series(_FLOW_VALUES)) is True


def test_empty_window_is_rejected() -> None:
    svc = _service()
    assert svc._generative_samples_are_fresh([]) is False


def test_stale_window_warning_is_rate_limited(caplog) -> None:
    """Verification finding 2026-07-30: this warning had the same unbounded-repeat
    shape the reader's own failure logging was fixed for -- ~2880 identical lines a
    day while the writer is down. Rate-limited, but still re-logged periodically so
    a stopped writer does not go silent."""
    import logging

    svc = _service()
    old_end = datetime.now(timezone.utc) - timedelta(days=3)
    stale = [
        ConfidenceSample(generated_at=old_end - (19 - i) * TICK, value=0.93)
        for i in range(20)
    ]

    with caplog.at_level(logging.WARNING, logger="orion-equilibrium"):
        for _ in range(10):
            assert svc._generative_samples_are_fresh(stale) is False

    warnings = [r for r in caplog.records if "window_stale" in r.getMessage()]
    assert len(warnings) == 1, f"expected 1 warning across 10 stale polls, got {len(warnings)}"
    assert svc._stale_window_polls == 10


def test_stale_counter_resets_when_the_window_recovers() -> None:
    """Otherwise a single earlier outage would permanently offset the re-log
    cadence for every later one."""
    svc = _service()
    old_end = datetime.now(timezone.utc) - timedelta(days=3)
    stale = [
        ConfidenceSample(generated_at=old_end - (19 - i) * TICK, value=0.93)
        for i in range(20)
    ]

    for _ in range(3):
        svc._generative_samples_are_fresh(stale)
    assert svc._stale_window_polls == 3

    assert svc._generative_samples_are_fresh(_now_series(_FLOW_VALUES)) is True
    assert svc._stale_window_polls == 0


def test_window_just_past_max_age_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "metacog_generative_max_age_sec", 60.0)
    svc = _service()
    samples = [
        ConfidenceSample(
            generated_at=datetime.now(timezone.utc) - timedelta(seconds=90), value=0.93
        )
    ]
    assert svc._generative_samples_are_fresh(samples) is False


# ===========================================================================
# Fetch-limit invariant (review finding S2)
# ===========================================================================


def test_fetch_limit_covers_flow_min_ticks_even_if_window_is_set_lower(monkeypatch) -> None:
    """Setting WINDOW_TICKS below FLOW_MIN_TICKS would otherwise make the flow
    gate a silent permanent no-op."""
    monkeypatch.setattr(settings, "metacog_generative_window_ticks", 5)
    monkeypatch.setattr(settings, "metacog_flow_min_ticks", 40)
    svc = _service()
    assert svc._generative_fetch_limit() >= 40


def test_fetch_limit_covers_insight_cross_plus_confirm(monkeypatch) -> None:
    monkeypatch.setattr(settings, "metacog_generative_window_ticks", 5)
    monkeypatch.setattr(settings, "metacog_flow_min_ticks", 5)
    monkeypatch.setattr(settings, "metacog_insight_max_ticks_to_cross", 30)
    monkeypatch.setattr(settings, "metacog_insight_confirm_ticks", 3)
    svc = _service()
    assert svc._generative_fetch_limit() >= 33


# ===========================================================================
# Shipped-disabled acceptance criterion (review finding N1)
# ===========================================================================


def test_both_generative_flags_default_to_disabled() -> None:
    """Hard acceptance criterion, not a preference: these gates dispatch real
    MetacogTriggerV1 events into orion_metacog and are flipped on by a human
    after a post-merge live-data check. A deterministic gate, so an accidental
    default flip fails CI instead of shipping quietly."""
    from app.settings import Settings

    fresh = Settings(_env_file=None)
    assert fresh.metacog_insight_trigger_enable is False
    assert fresh.metacog_flow_trigger_enable is False
