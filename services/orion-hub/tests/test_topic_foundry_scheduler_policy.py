"""Pins the behaviour branch 3 is actually selling.

Review finding 2026-08-28: nothing tested the sleep-ordering change. Reverting
the loop to its old unconditional `await asyncio.sleep(interval)` left all 33
hub tests green -- the headline fix was unpinned. The timing/retry decision now
lives in a pure module so it can be asserted directly.

The bug: at SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_INTERVAL_SEC=86400 the loop slept
a full day BEFORE its first tick, so it needed 24 unbroken hours of Hub uptime
to fire even once. Confirmed live: it never had.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(HUB_ROOT), str(HUB_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from topic_foundry_scheduler_policy import (  # noqa: E402
    RETRYABLE_TRIGGER_REASONS,
    STARTUP_TICK_DELAY_SEC,
    STARTUP_TICK_MAX_ATTEMPTS,
    next_wait_seconds,
    should_retry_startup_tick,
)

PROD_INTERVAL = 86400.0


def test_first_wait_is_the_startup_delay_not_a_whole_interval() -> None:
    """This single assertion is the patch. If it regresses, the scheduler is
    back to needing a day of uptime to run once."""
    wait = next_wait_seconds(startup_pending=True, interval_sec=PROD_INTERVAL)
    assert wait == STARTUP_TICK_DELAY_SEC
    assert wait < PROD_INTERVAL


def test_steady_state_still_uses_the_configured_interval() -> None:
    assert next_wait_seconds(startup_pending=False, interval_sec=PROD_INTERVAL) == PROD_INTERVAL


def test_startup_retries_back_off_but_stay_well_inside_one_interval() -> None:
    waits = [
        next_wait_seconds(startup_pending=True, interval_sec=PROD_INTERVAL, attempt=a)
        for a in range(STARTUP_TICK_MAX_ATTEMPTS)
    ]
    # Hand-computed from a 30s base: 30, 60, 120, 240.
    assert waits == [30.0, 60.0, 120.0, 240.0]
    assert sum(waits) < PROD_INTERVAL


def test_a_backed_off_retry_never_waits_longer_than_the_normal_cadence() -> None:
    """On a short configured interval the interval tick is the better deal."""
    assert next_wait_seconds(startup_pending=True, interval_sec=10.0, attempt=9) == 10.0


@pytest.mark.parametrize("reason", sorted(RETRYABLE_TRIGGER_REASONS))
def test_a_tick_that_could_not_reach_topic_foundry_is_retried(reason: str) -> None:
    """Hub and orion-topic-foundry come up together; the first tick can easily
    land before topic-foundry is answering."""
    assert should_retry_startup_tick({"triggered": False, "reason": reason}, attempt=0) is True


def test_a_raising_tick_is_retried() -> None:
    assert should_retry_startup_tick(None, attempt=0) is True


def test_a_successful_tick_is_not_retried() -> None:
    assert should_retry_startup_tick({"triggered": True, "run_id": "r"}, attempt=0) is False


def test_a_non_transient_failure_is_not_retried() -> None:
    """"Nothing to do" is not "could not reach it"."""
    assert (
        should_retry_startup_tick(
            {"triggered": False, "reason": "enrich_limit_non_positive"}, attempt=0
        )
        is False
    )


def test_retries_are_bounded() -> None:
    reason = sorted(RETRYABLE_TRIGGER_REASONS)[0]
    summary = {"triggered": False, "reason": reason}
    assert should_retry_startup_tick(summary, attempt=STARTUP_TICK_MAX_ATTEMPTS - 2) is True
    assert should_retry_startup_tick(summary, attempt=STARTUP_TICK_MAX_ATTEMPTS - 1) is False
    assert should_retry_startup_tick(None, attempt=STARTUP_TICK_MAX_ATTEMPTS) is False


def test_main_loop_waits_via_the_policy_rather_than_a_bare_interval_sleep() -> None:
    """Guards the wiring, not just the policy: a revert to the old
    unconditional `await asyncio.sleep(topic_foundry_interval_sec)` would leave
    every assertion above green while restoring the bug."""
    source = (HUB_ROOT / "scripts" / "main.py").read_text()
    body = source.split("_run_substrate_topic_foundry_scheduler")[1].split("async def ")[0]
    assert "_tf_next_wait_seconds(" in body
    assert "await asyncio.sleep(topic_foundry_interval_sec)" not in body
