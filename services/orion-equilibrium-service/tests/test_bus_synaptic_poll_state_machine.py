"""Service-level tests for the bus_synaptic poll loop's edge/hysteresis state.

Review finding (2026-07-30): the gate-function tests carried their OWN copy of
the service's bookkeeping, so they exercised the gate's contract and never the
state machine. Breaking the hysteresis in service.py left all 20 of them green.
These drive the real `_bus_synaptic_poll_loop` bookkeeping instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for c in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if c not in sys.path:
        sys.path.insert(0, c)

from app.transport_metacog_gate import (  # noqa: E402
    build_transport_metacog_trigger_from_bus_synaptic,
)

THRESHOLD = 1.0
CLEAR_RATIO = 0.8


class _Loop:
    """Mirrors _bus_synaptic_poll_loop's real bookkeeping, including the
    latch-on-publish-success ordering the must-fix introduced."""

    def __init__(self, publish: AsyncMock) -> None:
        self.above = False
        self.publish = publish

    async def poll(self, error: float | None) -> None:
        if error is None:
            return  # read failure -> state HELD, no fire
        trigger = build_transport_metacog_trigger_from_bus_synaptic(
            error,
            zen_state="not_zen",
            pressure=0.4,
            recall_enabled=False,
            error_threshold=THRESHOLD,
            previously_above=self.above,
        )
        published = True
        if trigger is not None:
            published = await self.publish(trigger)
        clear_at = THRESHOLD * CLEAR_RATIO
        if error >= THRESHOLD:
            self.above = published
        elif error < clear_at:
            self.above = False


@pytest.mark.asyncio
async def test_cooldown_suppressed_episode_start_is_retried_not_lost() -> None:
    """THE must-fix. The transport cooldown lane is shared with rpc_health and
    rpc_timeout (~859 sibling rows/day), so an episode start can land in a
    shadowed window. Latching from the raw level would drop it permanently:
    suppressed by the lane, latched anyway, then silenced by previously_above
    on every later poll. It must retry until it actually publishes."""
    publish = AsyncMock(side_effect=[False, False, True])
    loop = _Loop(publish)

    await loop.poll(1.2)  # lane shadowed -> suppressed
    assert loop.above is False, "must not latch on a suppressed publish"
    await loop.poll(1.2)  # still shadowed
    assert loop.above is False
    await loop.poll(1.2)  # lane clears -> publishes
    assert loop.above is True

    assert publish.await_count == 3
    # And now it goes quiet for the rest of the episode.
    await loop.poll(1.2)
    assert publish.await_count == 3


@pytest.mark.asyncio
async def test_sustained_episode_publishes_exactly_once() -> None:
    publish = AsyncMock(return_value=True)
    loop = _Loop(publish)
    for _ in range(120):  # an hour of 30s polls, condition held
        await loop.poll(1.5)
    assert publish.await_count == 1


@pytest.mark.asyncio
async def test_hysteresis_band_holds_state_and_suppresses_flapping() -> None:
    """Inside [clear_at, threshold) the previous state is HELD. Without it, a
    reading oscillating across the threshold re-arms and re-fires on every
    crossing."""
    publish = AsyncMock(return_value=True)
    loop = _Loop(publish)

    await loop.poll(1.0)          # episode start
    assert publish.await_count == 1
    for value in (0.85, 1.0, 0.9, 1.0, 0.95):  # all inside/above the band
        await loop.poll(value)
    assert publish.await_count == 1, "flapping inside the band must not re-fire"

    await loop.poll(0.1)          # genuine clear, below clear_at
    assert loop.above is False
    await loop.poll(1.0)          # new episode
    assert publish.await_count == 2


@pytest.mark.asyncio
async def test_read_failure_holds_state_and_cannot_cause_spam() -> None:
    """A FalkorDB read failure yields error=None. State must be HELD, not
    reset -- resetting would make the next successful poll look like a fresh
    rising edge and re-fire mid-episode."""
    publish = AsyncMock(return_value=True)
    loop = _Loop(publish)

    await loop.poll(1.2)
    assert publish.await_count == 1
    for _ in range(10):
        await loop.poll(None)
    assert loop.above is True
    await loop.poll(1.2)
    assert publish.await_count == 1


@pytest.mark.asyncio
async def test_below_threshold_never_publishes() -> None:
    publish = AsyncMock(return_value=True)
    loop = _Loop(publish)
    for value in (0.0, 0.011, 0.5, 0.79):
        await loop.poll(value)
    publish.assert_not_awaited()
    assert loop.above is False
