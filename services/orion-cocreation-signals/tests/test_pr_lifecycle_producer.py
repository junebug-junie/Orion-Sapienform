"""Unit tests for app/producers/pr_lifecycle.py's scheduling logic --
publish-every-tick (unlike git_delta/graph_delta), half-open window tiling,
and possibly_truncated pass-through. The pure fetch/scoring functions are
already covered by orion/structural_mass/tests/.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SERVICE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.producers import pr_lifecycle as pr_lifecycle_module  # noqa: E402

from orion.structural_mass.pr_lifecycle import PrLifecycleDelta  # noqa: E402


async def _run_until_calls(loop_coro_factory, stop: asyncio.Event, target_calls: int, call_list: list) -> None:
    async def _watcher() -> None:
        while len(call_list) < target_calls:
            await asyncio.sleep(0.001)
        stop.set()

    await asyncio.wait_for(asyncio.gather(loop_coro_factory(), _watcher()), timeout=5.0)


@pytest.mark.asyncio
async def test_publishes_every_tick_even_with_zero_counts(fake_bus, source, stop_event, monkeypatch):
    """Unlike git_delta/graph_delta, a real all-zero window is a genuine
    observation ('checked, nothing happened'), not a no-op to skip."""
    calls: list[str] = []

    def _fake_fetch(owner, repo, *, limit):
        calls.append(owner)
        return []

    def _fake_score(prs, since, until, *, fetch_limit):
        return PrLifecycleDelta(
            since=since, until=until, submitted_count=0, merged_count=0, closed_without_merge_count=0,
        )

    monkeypatch.setattr(pr_lifecycle_module, "fetch_recent_prs", _fake_fetch)
    monkeypatch.setattr(pr_lifecycle_module, "pr_lifecycle_delta", _fake_score)

    async def _loop():
        await pr_lifecycle_module.pr_lifecycle_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            owner="o", repo="r", fetch_limit=200, poll_interval_sec=0.01, cold_start_lookback_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    assert len(fake_bus.published) == 3
    for _, envelope in fake_bus.published:
        assert envelope.payload["domain"] == "pr_lifecycle"
        assert envelope.payload["pr_lifecycle"]["submitted_count"] == 0
        assert envelope.payload["git"] is None
        assert envelope.payload["graph"] is None


@pytest.mark.asyncio
async def test_windows_tile_contiguously_tick_to_tick(fake_bus, source, stop_event, monkeypatch):
    """This tick's `until` must become the next tick's `since` -- the
    half-open-interval dedup mechanism named in pr_lifecycle_delta's own
    docstring."""
    seen_windows: list[tuple] = []

    def _fake_fetch(owner, repo, *, limit):
        return []

    def _fake_score(prs, since, until, *, fetch_limit):
        seen_windows.append((since, until))
        return PrLifecycleDelta(since=since, until=until, submitted_count=0, merged_count=0, closed_without_merge_count=0)

    monkeypatch.setattr(pr_lifecycle_module, "fetch_recent_prs", _fake_fetch)
    monkeypatch.setattr(pr_lifecycle_module, "pr_lifecycle_delta", _fake_score)

    async def _loop():
        await pr_lifecycle_module.pr_lifecycle_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            owner="o", repo="r", fetch_limit=200, poll_interval_sec=0.01, cold_start_lookback_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=seen_windows)

    assert seen_windows[0][1] == seen_windows[1][0]  # tick1.until == tick2.since
    assert seen_windows[1][1] == seen_windows[2][0]  # tick2.until == tick3.since


@pytest.mark.asyncio
async def test_possibly_truncated_passed_through_to_event(fake_bus, source, stop_event, monkeypatch):
    calls: list[str] = []

    def _fake_fetch(owner, repo, *, limit):
        calls.append(owner)
        return []

    def _fake_score(prs, since, until, *, fetch_limit):
        return PrLifecycleDelta(
            since=since, until=until, submitted_count=1, merged_count=0, closed_without_merge_count=0,
            possibly_truncated=True,
        )

    monkeypatch.setattr(pr_lifecycle_module, "fetch_recent_prs", _fake_fetch)
    monkeypatch.setattr(pr_lifecycle_module, "pr_lifecycle_delta", _fake_score)

    async def _loop():
        await pr_lifecycle_module.pr_lifecycle_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            owner="o", repo="r", fetch_limit=200, poll_interval_sec=0.01, cold_start_lookback_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=1, call_list=calls)

    assert fake_bus.published[0][1].payload["pr_lifecycle"]["possibly_truncated"] is True


@pytest.mark.asyncio
async def test_failed_publish_does_not_advance_window(fake_bus, source, stop_event, monkeypatch):
    """Regression guard (code review 2026-07-30): if the bus publish itself
    raises, last_until must NOT advance -- the next tick's window should
    extend to cover the failed range too, instead of silently dropping it."""

    async def _raising_publish(channel, envelope):
        raise ConnectionError("transient redis failure")

    fake_bus.publish = _raising_publish

    seen_windows: list[tuple] = []

    def _fake_fetch(owner, repo, *, limit):
        return []

    def _fake_score(prs, since, until, *, fetch_limit):
        seen_windows.append((since, until))
        return PrLifecycleDelta(since=since, until=until, submitted_count=0, merged_count=0, closed_without_merge_count=0)

    monkeypatch.setattr(pr_lifecycle_module, "fetch_recent_prs", _fake_fetch)
    monkeypatch.setattr(pr_lifecycle_module, "pr_lifecycle_delta", _fake_score)

    async def _loop():
        await pr_lifecycle_module.pr_lifecycle_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            owner="o", repo="r", fetch_limit=200, poll_interval_sec=0.01, cold_start_lookback_sec=0.01,
            stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=seen_windows)

    # Every window's `since` must be identical -- proof last_until never
    # advanced past a failed publish (windows only ever grow, never shift).
    first_since = seen_windows[0][0]
    assert all(since == first_since for since, _ in seen_windows)
    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_cold_start_window_uses_dedicated_lookback_not_poll_interval(fake_bus, source, stop_event, monkeypatch):
    """The first window's span must come from cold_start_lookback_sec, not
    poll_interval_sec -- deliberately set to different values here so a
    regression that swaps them (or falls back to the wrong one) fails this
    test instead of passing by coincidence."""
    windows: list[tuple] = []

    def _fake_fetch(owner, repo, *, limit):
        return []

    def _fake_score(prs, since, until, *, fetch_limit):
        windows.append((since, until))
        return PrLifecycleDelta(since=since, until=until, submitted_count=0, merged_count=0, closed_without_merge_count=0)

    monkeypatch.setattr(pr_lifecycle_module, "fetch_recent_prs", _fake_fetch)
    monkeypatch.setattr(pr_lifecycle_module, "pr_lifecycle_delta", _fake_score)

    async def _loop():
        await pr_lifecycle_module.pr_lifecycle_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            owner="o", repo="r", fetch_limit=200, poll_interval_sec=5.0, cold_start_lookback_sec=60.0,
            stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=1, call_list=windows)

    since, until = windows[0]
    span_sec = (until - since).total_seconds()
    assert 59.0 <= span_sec <= 61.0  # ~cold_start_lookback_sec (60), not poll_interval_sec (5)
