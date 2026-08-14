"""Unit tests for app/producers/affective_state.py's scheduling logic --
publish-every-tick (a real all-zero window is a genuine observation, same
as pr_lifecycle), half-open window tiling, and the fail-loud-on-missing-
mount startup guard. The real end-to-end scoring path (_score_window against
real transcript files) is covered separately, not mocked away, since that
wiring -- not just the scheduling loop -- is what this producer actually adds.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SERVICE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.producers import affective_state as affective_state_module  # noqa: E402

from orion.schemas.affective_state import JuniperAffectiveStateV1  # noqa: E402


async def _run_until_calls(loop_coro_factory, stop: asyncio.Event, target_calls: int, call_list: list) -> None:
    async def _watcher() -> None:
        while len(call_list) < target_calls:
            await asyncio.sleep(0.001)
        stop.set()

    await asyncio.wait_for(asyncio.gather(loop_coro_factory(), _watcher()), timeout=5.0)


def _fake_event(since: datetime, until: datetime, **overrides) -> JuniperAffectiveStateV1:
    fields = dict(
        observed_at=datetime.now(timezone.utc),
        window_since=since,
        window_until=until,
        message_count=0,
        word_count=0,
        swear_count=0,
        swear_frequency=None,
    )
    fields.update(overrides)
    return JuniperAffectiveStateV1(**fields)


@pytest.mark.asyncio
async def test_publishes_every_tick_even_with_zero_messages(
    fake_bus, source, stop_event, monkeypatch, tmp_path
) -> None:
    """A real all-zero window ('checked, no new messages') is a genuine
    observation, not a no-op to skip -- same reasoning as pr_lifecycle."""
    calls: list[tuple] = []

    def _fake_score_window(path, since, until, *, cold_start=False):
        calls.append((since, until))
        return _fake_event(since, until)

    monkeypatch.setattr(affective_state_module, "_score_window", _fake_score_window)

    async def _loop():
        await affective_state_module.affective_state_loop(
            bus=fake_bus, channel="orion:substrate:juniper_affective_state", source=source,
            claude_projects_path=str(tmp_path), poll_interval_sec=0.01, cold_start_lookback_sec=0.01,
            stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    assert len(fake_bus.published) == 3
    for _, envelope in fake_bus.published:
        assert envelope.payload["message_count"] == 0
        assert envelope.payload["swear_frequency"] is None


@pytest.mark.asyncio
async def test_windows_tile_contiguously_tick_to_tick(
    fake_bus, source, stop_event, monkeypatch, tmp_path
) -> None:
    seen_windows: list[tuple] = []

    def _fake_score_window(path, since, until, *, cold_start=False):
        seen_windows.append((since, until))
        return _fake_event(since, until)

    monkeypatch.setattr(affective_state_module, "_score_window", _fake_score_window)

    async def _loop():
        await affective_state_module.affective_state_loop(
            bus=fake_bus, channel="orion:substrate:juniper_affective_state", source=source,
            claude_projects_path=str(tmp_path), poll_interval_sec=0.01, cold_start_lookback_sec=0.01,
            stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=seen_windows)

    assert seen_windows[0][1] == seen_windows[1][0]
    assert seen_windows[1][1] == seen_windows[2][0]


@pytest.mark.asyncio
async def test_failed_publish_does_not_advance_window(
    fake_bus, source, stop_event, monkeypatch, tmp_path
) -> None:
    async def _raising_publish(channel, envelope):
        raise ConnectionError("transient redis failure")

    fake_bus.publish = _raising_publish

    seen_windows: list[tuple] = []

    def _fake_score_window(path, since, until, *, cold_start=False):
        seen_windows.append((since, until))
        return _fake_event(since, until)

    monkeypatch.setattr(affective_state_module, "_score_window", _fake_score_window)

    async def _loop():
        await affective_state_module.affective_state_loop(
            bus=fake_bus, channel="orion:substrate:juniper_affective_state", source=source,
            claude_projects_path=str(tmp_path), poll_interval_sec=0.01, cold_start_lookback_sec=0.01,
            stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=seen_windows)

    first_since = seen_windows[0][0]
    assert all(since == first_since for since, _ in seen_windows)
    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_cold_start_window_uses_dedicated_lookback_not_poll_interval(
    fake_bus, source, stop_event, monkeypatch, tmp_path
) -> None:
    windows: list[tuple] = []

    def _fake_score_window(path, since, until, *, cold_start=False):
        windows.append((since, until))
        return _fake_event(since, until)

    monkeypatch.setattr(affective_state_module, "_score_window", _fake_score_window)

    async def _loop():
        await affective_state_module.affective_state_loop(
            bus=fake_bus, channel="orion:substrate:juniper_affective_state", source=source,
            claude_projects_path=str(tmp_path), poll_interval_sec=5.0, cold_start_lookback_sec=60.0,
            stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=1, call_list=windows)

    since, until = windows[0]
    span_sec = (until - since).total_seconds()
    assert 59.0 <= span_sec <= 61.0


@pytest.mark.asyncio
async def test_missing_claude_projects_path_fails_loud_not_silent(
    fake_bus, source, stop_event
) -> None:
    """A missing/misconfigured mount must not silently publish an empty
    window forever, claiming 'checked, found nothing' when the real fact is
    'never checked at all'."""
    await asyncio.wait_for(
        affective_state_module.affective_state_loop(
            bus=fake_bus, channel="orion:substrate:juniper_affective_state", source=source,
            claude_projects_path="/definitely/does/not/exist/anywhere",
            poll_interval_sec=0.01, cold_start_lookback_sec=0.01, stop=stop_event,
        ),
        timeout=2.0,
    )
    assert fake_bus.published == []


def test_score_window_real_end_to_end_against_real_transcript_files(tmp_path) -> None:
    """Not mocked -- exercises the real wiring between _score_window,
    orion.dev_economics.claude_code_ingest, and orion.cocreation.affective_signals
    against a real transcript file on disk."""
    now = datetime.now(timezone.utc)
    transcript = tmp_path / "session.jsonl"
    in_window = {
        "type": "user",
        "message": {"role": "user", "content": "this is fucking annoying"},
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        "sessionId": "s1",
        "promptSource": "typed",
        "origin": {"kind": "human"},
    }
    out_of_window = {
        "type": "user",
        "message": {"role": "user", "content": "old message, outside the window"},
        "timestamp": (now - timedelta(days=2)).isoformat().replace("+00:00", "Z"),
        "sessionId": "s1",
        "promptSource": "typed",
        "origin": {"kind": "human"},
    }
    transcript.write_text(
        json.dumps(out_of_window) + "\n" + json.dumps(in_window), encoding="utf-8"
    )

    event = affective_state_module._score_window(
        str(tmp_path), now - timedelta(hours=1), now + timedelta(hours=1)
    )
    assert event.message_count == 1
    assert event.swear_count == 1
    assert event.swear_frequency == 0.25


@pytest.mark.asyncio
async def test_only_the_first_tick_is_flagged_cold_start(
    fake_bus, source, stop_event, monkeypatch, tmp_path
) -> None:
    """The first tick's window is cold_start_lookback_sec wide and therefore
    OVERLAPS windows a previous run already published; every later tick tiles
    exactly and can be summed with its neighbours.

    Without the flag, the obvious `SUM(word_count)` over an hour silently
    double-counts. Confirmed on the first two rows ever persisted
    (2026-08-14): a 913s window sat entirely inside a 3600s one and the sum
    returned 7669 words against a true 5913, +34.7%.

    Asserts the whole sequence, not just the first element -- an
    implementation that flagged every tick, or that never cleared the flag,
    would pass a `flags[0] is True` check."""
    flags: list[bool] = []

    def _fake_score_window(path, since, until, *, cold_start=False):
        flags.append(cold_start)
        return _fake_event(since, until)

    monkeypatch.setattr(affective_state_module, "_score_window", _fake_score_window)

    async def _loop():
        await affective_state_module.affective_state_loop(
            bus=fake_bus, channel="orion:substrate:juniper_affective_state", source=source,
            claude_projects_path=str(tmp_path), poll_interval_sec=0.01, cold_start_lookback_sec=0.01,
            stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=flags)

    assert flags[0] is True
    assert all(flag is False for flag in flags[1:])


@pytest.mark.asyncio
async def test_cold_start_flag_survives_a_failed_publish(
    fake_bus, source, stop_event, monkeypatch, tmp_path
) -> None:
    """The flag clears on a SUCCESSFUL publish, not merely on having ticked.

    A failed publish leaves `last_until` parked, so the retry covers an even
    WIDER range that still overlaps rows a previous run published. Clearing
    the flag unconditionally would hand that retry out as an ordinary tiling
    window -- the exact row a consumer would then wrongly include in a sum."""
    async def _raising_publish(channel, envelope):
        raise ConnectionError("transient redis failure")

    fake_bus.publish = _raising_publish

    flags: list[bool] = []

    def _fake_score_window(path, since, until, *, cold_start=False):
        flags.append(cold_start)
        return _fake_event(since, until)

    monkeypatch.setattr(affective_state_module, "_score_window", _fake_score_window)

    async def _loop():
        await affective_state_module.affective_state_loop(
            bus=fake_bus, channel="orion:substrate:juniper_affective_state", source=source,
            claude_projects_path=str(tmp_path), poll_interval_sec=0.01, cold_start_lookback_sec=0.01,
            stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=flags)

    assert all(flag is True for flag in flags), flags
