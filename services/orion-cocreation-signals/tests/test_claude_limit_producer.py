"""Unit tests for app/producers/claude_limit.py.

Two things carry the weight here. First, `_to_event` must COPY the
observation's derived properties rather than recompute them: `state` encodes
two independent ways a limit lifts (a stated reset time passing, and real
activity after the event) and a second implementation would drift from the one
that was debugged live. Second, an unobserved window must publish
`observed=False` rather than nothing, so a consumer can tell a full pool from
a vanished mount.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SERVICE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.producers import claude_limit as mod  # noqa: E402
from app.settings import _parse_window_hours  # noqa: E402

from orion.dev_economics.rate_limit_events import LimitObservation, RateLimitEvent  # noqa: E402

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)


# Sentinel, not None: an explicit `latest_activity=None` is the unobserved
# case this suite has to be able to construct, and a `x if x is not None else
# default` helper silently substitutes the default for it instead.
_UNSET = object()


def _obs(*, hours=5.0, events=(), latest_activity=_UNSET, latest_success=None,
         message_count=3, file_count=7) -> LimitObservation:
    return LimitObservation(
        window_hours=hours,
        window_start=NOW - timedelta(hours=hours),
        window_end=NOW,
        events=tuple(events),
        latest_activity_at=(NOW - timedelta(seconds=5)) if latest_activity is _UNSET else latest_activity,
        latest_success_at=latest_success,
        observed_message_count=message_count,
        scanned_file_count=file_count,
    )


def test_to_event_copies_the_observations_own_derived_values():
    obs = _obs()
    ev = mod._to_event(obs, observed_at=NOW)
    # Every derived field comes from the property, not a local recomputation.
    assert ev.state == obs.state
    assert ev.observed == obs.observed
    assert ev.event_count == obs.event_count
    assert ev.resets_at == obs.resets_at
    assert ev.seconds_until_reset == obs.seconds_until_reset
    assert ev.staleness_sec == obs.staleness_sec
    assert (ev.window_hours, ev.window_start, ev.window_end) == (
        obs.window_hours, obs.window_start, obs.window_end
    )
    assert (ev.observed_message_count, ev.scanned_file_count) == (
        obs.observed_message_count, obs.scanned_file_count
    )


def test_a_live_limit_carries_its_reset_time_through():
    resets = NOW + timedelta(minutes=30)
    obs = _obs(events=[RateLimitEvent(at=NOW - timedelta(minutes=5), kind="session", resets_at=resets, source="t.jsonl")])
    ev = mod._to_event(obs, observed_at=NOW)
    assert ev.state == "limited"
    assert ev.resets_at == resets
    # The number a consumer actually gates on, not something it has to derive.
    assert ev.seconds_until_reset == pytest.approx(1800.0)


def test_an_unobserved_window_publishes_unknown_rather_than_nothing():
    obs = _obs(message_count=0, latest_activity=None)
    ev = mod._to_event(obs, observed_at=NOW)
    assert ev.state == "unknown"
    assert ev.observed is False
    assert ev.staleness_sec is None


@pytest.mark.asyncio
async def test_loop_refuses_to_start_on_a_missing_mount(fake_bus, source, tmp_path, caplog):
    stop = asyncio.Event()
    await mod.claude_limit_loop(
        bus=fake_bus, channel="c", source=source,
        claude_projects_path=str(tmp_path / "does-not-exist"),
        window_hours=(5.0,), poll_interval_sec=0.01, stop=stop,
    )
    # Returns rather than publishing a stream of empty ticks that would
    # misreport a broken mount as a quiet window.
    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_loop_refuses_to_start_with_no_windows_configured(fake_bus, source, tmp_path):
    stop = asyncio.Event()
    await mod.claude_limit_loop(
        bus=fake_bus, channel="c", source=source,
        claude_projects_path=str(tmp_path), window_hours=(),
        poll_interval_sec=0.01, stop=stop,
    )
    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_loop_publishes_one_event_per_window_each_tick(fake_bus, source, tmp_path, monkeypatch):
    stop = asyncio.Event()
    monkeypatch.setattr(mod, "observe", lambda *, window_hours, root: _obs(hours=window_hours))

    async def _run():
        await mod.claude_limit_loop(
            bus=fake_bus, channel="orion:substrate:claude_limit", source=source,
            claude_projects_path=str(tmp_path), window_hours=(5.0, 168.0),
            poll_interval_sec=0.01, stop=stop,
        )

    async def _watch():
        while len(fake_bus.published) < 2:
            await asyncio.sleep(0.001)
        stop.set()

    await asyncio.wait_for(asyncio.gather(_run(), _watch()), timeout=5.0)
    windows = [e.payload["window_hours"] for _, e in fake_bus.published[:2]]
    assert windows == [5.0, 168.0]
    assert all(e.kind == "substrate.claude_limit.v1" for _, e in fake_bus.published)


@pytest.mark.asyncio
async def test_one_failing_window_does_not_cost_the_tick_its_other_observation(
    fake_bus, source, tmp_path, monkeypatch
):
    stop = asyncio.Event()

    def _flaky(*, window_hours, root):
        if window_hours == 5.0:
            raise RuntimeError("unparseable window")
        return _obs(hours=window_hours)

    monkeypatch.setattr(mod, "observe", _flaky)

    async def _run():
        await mod.claude_limit_loop(
            bus=fake_bus, channel="c", source=source,
            claude_projects_path=str(tmp_path), window_hours=(5.0, 168.0),
            poll_interval_sec=0.01, stop=stop,
        )

    async def _watch():
        while not fake_bus.published:
            await asyncio.sleep(0.001)
        stop.set()

    await asyncio.wait_for(asyncio.gather(_run(), _watch()), timeout=5.0)
    assert fake_bus.published[0][1].payload["window_hours"] == 168.0


@pytest.mark.asyncio
async def test_publish_is_skipped_cleanly_when_the_bus_is_disabled(fake_bus, source, tmp_path, monkeypatch):
    fake_bus.enabled = False
    monkeypatch.setattr(mod, "observe", lambda *, window_hours, root: _obs(hours=window_hours))
    ev = mod._to_event(_obs(), observed_at=NOW)
    await mod._publish(fake_bus, "c", source, ev)
    assert fake_bus.published == []


# -- the settings helper both the validator and the property depend on ---


@pytest.mark.parametrize("raw,expected", [
    ("5,168", (5.0, 168.0)),
    # Deduplicated and sorted ascending, so the tightest window publishes first.
    ("168,5,5", (5.0, 168.0)),
    ("0.5, 5", (0.5, 5.0)),
    ("", ()),
    ("abc", ()),
    ("-1", ()),
    ("0", ()),
    ("5,abc", ()),
])
def test_parse_window_hours(raw, expected):
    assert _parse_window_hours(raw) == expected
