"""HarnessStepRelay fans harness.run.step events to per-correlation queues."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from scripts.harness_step_relay import HarnessStepRelay


@pytest.mark.asyncio
async def test_harness_step_relay_dispatches_matching_correlation() -> None:
    relay = HarnessStepRelay(channel="orion:harness:run:step")
    queue: asyncio.Queue = asyncio.Queue()
    relay.register_queue("corr-1", queue)

    step_event = MagicMock()
    step_event.correlation_id = "corr-1"
    step_event.step_index = 0
    step_event.step = {"type": "assistant", "raw": {"type": "assistant"}}

    await relay._dispatch_step(step_event)

    item = queue.get_nowait()
    assert item["kind"] == "claude_step"
    assert item["mode"] == "orion"
    assert item["correlation_id"] == "corr-1"
    assert item["step_index"] == 0


@pytest.mark.asyncio
async def test_harness_step_relay_ignores_unregistered_correlation() -> None:
    relay = HarnessStepRelay(channel="orion:harness:run:step")
    step_event = MagicMock()
    step_event.correlation_id = "missing"
    step_event.step_index = 0
    step_event.step = {"type": "assistant"}
    await relay._dispatch_step(step_event)


@pytest.mark.asyncio
async def test_harness_step_relay_tracks_liveness_even_without_queue() -> None:
    """A step should mark liveness for its correlation_id even with no registered UI queue."""
    relay = HarnessStepRelay(channel="orion:harness:run:step")
    step_event = MagicMock()
    step_event.correlation_id = "corr-2"
    step_event.step_index = 0
    step_event.step = {"type": "assistant"}

    assert relay.seen_recently("corr-2", within_sec=60) is False
    await relay._dispatch_step(step_event)
    assert relay.seen_recently("corr-2", within_sec=60) is True
    assert relay.seen_recently("corr-2", within_sec=0) is False


def test_harness_step_relay_forget_clears_liveness() -> None:
    relay = HarnessStepRelay(channel="orion:harness:run:step")
    relay._last_seen["corr-3"] = 0.0
    relay.forget("corr-3")
    assert relay.seen_recently("corr-3", within_sec=1_000_000) is False


@pytest.mark.asyncio
async def test_harness_step_relay_sweeps_stale_entries_never_forgotten() -> None:
    """Regression test: _last_seen must not grow unbounded for correlation_ids this
    process observes on the shared step channel but never explicitly forget()s (e.g. a
    turn owned by a sibling hub replica). A stale entry should eventually be evicted by
    the opportunistic TTL sweep, not linger forever.
    """
    relay = HarnessStepRelay(channel="orion:harness:run:step", last_seen_ttl_sec=100.0)
    relay._SWEEP_INTERVAL_SEC = 0.0  # sweep on every dispatch for the test

    # Simulate an old entry from a correlation_id this instance will never forget().
    relay._last_seen["orphaned-corr"] = time.monotonic() - 1000.0  # well past the 100s TTL

    step_event = MagicMock()
    step_event.correlation_id = "new-corr"
    step_event.step_index = 0
    step_event.step = {"type": "assistant"}
    await relay._dispatch_step(step_event)

    assert "orphaned-corr" not in relay._last_seen
    assert "new-corr" in relay._last_seen


def test_harness_step_relay_sweep_is_rate_limited() -> None:
    """The sweep should not rescan the whole dict on every single dispatch."""
    relay = HarnessStepRelay(channel="orion:harness:run:step", last_seen_ttl_sec=100.0)
    relay._last_seen["stale"] = -1_000_000.0
    relay._last_sweep_monotonic = time.monotonic()  # sweep just ran

    relay._sweep_last_seen(now=time.monotonic())

    assert "stale" in relay._last_seen  # not swept yet, interval hasn't elapsed


@pytest.mark.asyncio
async def test_harness_step_relay_last_seen_bounded_by_max_entries() -> None:
    """Regression test: a burst of unique correlation_ids within a single sweep interval
    (before any of them individually crosses the TTL) must still be bounded by a hard
    entry-count cap, not just the time-gated TTL sweep — same cap+TTL pattern as
    CognitionTraceCache/SignalsInspectCache elsewhere in this service.
    """
    relay = HarnessStepRelay(
        channel="orion:harness:run:step",
        last_seen_ttl_sec=100.0,
        last_seen_max_entries=3,
    )

    for cid in ("c1", "c2", "c3", "c4", "c5"):
        step_event = MagicMock()
        step_event.correlation_id = cid
        step_event.step_index = 0
        step_event.step = {"type": "assistant"}
        await relay._dispatch_step(step_event)

    assert len(relay._last_seen) == 3
    # Oldest-touched entries evicted first; most recent survive.
    assert set(relay._last_seen.keys()) == {"c3", "c4", "c5"}
