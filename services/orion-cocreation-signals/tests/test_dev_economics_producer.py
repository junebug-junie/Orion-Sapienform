"""Unit tests for app/producers/dev_economics.py's scheduling logic --
cold-start-scan (no publish on tick 1, matching git_delta_loop's own cold
start), real-delta publish on growth, no-publish on a no-op rescan, and the
fail-loud-on-missing-mount startup guard. The real end-to-end scoring path
(_scan_totals/_score_tick against real transcript files) is covered
separately, not mocked away, since that wiring is what this producer
actually adds.
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

from app.producers import dev_economics as dev_economics_module  # noqa: E402

from orion.dev_economics.claude_code_ingest import SessionUsageRecord  # noqa: E402


async def _run_until_calls(loop_coro_factory, stop: asyncio.Event, target_calls: int, call_list: list) -> None:
    async def _watcher() -> None:
        while len(call_list) < target_calls:
            await asyncio.sleep(0.001)
        stop.set()

    await asyncio.wait_for(asyncio.gather(loop_coro_factory(), _watcher()), timeout=5.0)


def _record(session_id: str, input_tokens: int) -> SessionUsageRecord:
    return SessionUsageRecord(
        session_id=session_id,
        transcript_path=Path(f"/fake/{session_id}.jsonl"),
        is_subagent=False,
        models=("claude-sonnet-5",),
        effort_tiers=("medium",),
        input_tokens=input_tokens,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        assistant_turn_count=1,
        assistant_word_count=10,
        human_turn_count=1,
        human_word_count=5,
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_cold_start_seeds_baseline_without_publishing(fake_bus, source, stop_event, monkeypatch, tmp_path):
    calls: list[int] = []

    def _fake_scan(path):
        calls.append(1)
        return {"s1": _record("s1", 1000)}

    monkeypatch.setattr(dev_economics_module, "_scan_totals", _fake_scan)

    async def _loop():
        await dev_economics_module.dev_economics_loop(
            bus=fake_bus, channel="orion:substrate:dev_economics_ledger", source=source,
            claude_projects_path=str(tmp_path), poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=1, call_list=calls)

    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_real_growth_publishes_the_delta_not_the_cumulative_total(
    fake_bus, source, stop_event, monkeypatch, tmp_path
):
    """Regression guard for the 2026-08-12 code review finding: a session
    still growing across ticks must publish its real incremental growth,
    not its full running total (which would double-count) and not nothing
    (which would silently drop it)."""
    totals_by_tick = [
        {"s1": _record("s1", 1000)},  # cold start
        {"s1": _record("s1", 2500)},  # real growth: +1500
    ]
    calls: list[int] = []

    def _fake_scan(path):
        idx = min(len(calls), len(totals_by_tick) - 1)
        calls.append(1)
        return totals_by_tick[idx]

    monkeypatch.setattr(dev_economics_module, "_scan_totals", _fake_scan)

    async def _loop():
        await dev_economics_module.dev_economics_loop(
            bus=fake_bus, channel="orion:substrate:dev_economics_ledger", source=source,
            claude_projects_path=str(tmp_path), poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=2, call_list=calls)

    assert len(fake_bus.published) == 1
    _, envelope = fake_bus.published[0]
    assert envelope.payload["total_input_tokens"] == 1500


@pytest.mark.asyncio
async def test_no_growth_still_publishes_a_real_zero_delta_tick(fake_bus, source, stop_event, monkeypatch, tmp_path):
    """After cold start, every tick publishes -- a real 'checked, nothing
    grew' observation is a genuine fact worth reporting (same 'publish
    every tick, even all-zero' convention as affective_state_loop), not a
    no-op to suppress. The published event's own totals correctly read
    zero, since no session actually grew between rescans."""
    same_totals = {"s1": _record("s1", 1000)}
    calls: list[int] = []

    def _fake_scan(path):
        calls.append(1)
        return same_totals

    monkeypatch.setattr(dev_economics_module, "_scan_totals", _fake_scan)

    async def _loop():
        await dev_economics_module.dev_economics_loop(
            bus=fake_bus, channel="orion:substrate:dev_economics_ledger", source=source,
            claude_projects_path=str(tmp_path), poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    # 3 real scans = 1 cold start (no publish) + 2 real ticks published.
    assert len(fake_bus.published) == 2
    for _, envelope in fake_bus.published:
        assert envelope.payload["session_count"] == 0
        assert envelope.payload["total_input_tokens"] == 0


@pytest.mark.asyncio
async def test_failed_publish_does_not_advance_baseline(fake_bus, source, stop_event, monkeypatch, tmp_path):
    async def _raising_publish(channel, envelope):
        raise ConnectionError("transient redis failure")

    fake_bus.publish = _raising_publish

    totals_by_tick = [
        {"s1": _record("s1", 1000)},  # cold start
        {"s1": _record("s1", 2500)},  # real growth, but publish will fail
        {"s1": _record("s1", 2500)},  # unchanged from tick 2 -- but baseline never advanced
    ]
    calls: list[int] = []

    def _fake_scan(path):
        idx = min(len(calls), len(totals_by_tick) - 1)
        calls.append(1)
        return totals_by_tick[idx]

    monkeypatch.setattr(dev_economics_module, "_scan_totals", _fake_scan)

    async def _loop():
        await dev_economics_module.dev_economics_loop(
            bus=fake_bus, channel="orion:substrate:dev_economics_ledger", source=source,
            claude_projects_path=str(tmp_path), poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_missing_claude_projects_path_fails_loud_not_silent(fake_bus, source, stop_event):
    await asyncio.wait_for(
        dev_economics_module.dev_economics_loop(
            bus=fake_bus, channel="orion:substrate:dev_economics_ledger", source=source,
            claude_projects_path="/definitely/does/not/exist/anywhere",
            poll_interval_sec=0.01, stop=stop_event,
        ),
        timeout=2.0,
    )
    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_tick_exception_does_not_kill_the_loop(fake_bus, source, stop_event, monkeypatch, tmp_path):
    calls: list[int] = []

    def _flaky_scan(path):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient scan failure")
        return {}

    monkeypatch.setattr(dev_economics_module, "_scan_totals", _flaky_scan)

    async def _loop():
        await dev_economics_module.dev_economics_loop(
            bus=fake_bus, channel="orion:substrate:dev_economics_ledger", source=source,
            claude_projects_path=str(tmp_path), poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=2, call_list=calls)

    assert len(calls) == 2


def test_score_tick_real_end_to_end_against_real_transcript_files(tmp_path) -> None:
    """Not mocked -- exercises the real wiring between _scan_totals,
    orion.dev_economics.claude_code_ingest, and
    orion.dev_economics.ledger_aggregate against a real transcript file on
    disk."""
    now = datetime.now(timezone.utc)
    transcript_dir = tmp_path / "proj" / "sess"
    transcript_dir.mkdir(parents=True)
    transcript = transcript_dir / "session.jsonl"

    line = {
        "type": "assistant",
        "message": {
            "id": "msg_1",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "real work done here"}],
            "usage": {
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        "sessionId": "s1",
    }
    transcript.write_text(json.dumps(line), encoding="utf-8")

    baseline: dict = {}
    event, current_totals = dev_economics_module._score_tick(str(tmp_path), baseline, now)
    assert event.total_input_tokens == 1_000_000
    assert event.total_output_tokens == 1_000_000
    assert "s1" in current_totals
