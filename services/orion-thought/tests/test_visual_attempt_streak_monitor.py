"""Third watchdog check: `visual_attempt_failure_streak` (N painting attempts in a row produced
nothing).

Incident 2026-10-16 -> 10-20: gpu2 sat faulted for 4 days. 35 painting attempts in a row ended
`abandoned`. The only alert was the painting-gap card: once, 12 h in, severity "error" (a pending
card that emails once after 60 min unacked). These pin that the 10-16 state now raises a CRITICAL
card (orion-notify emails critical at once) by the 6th failed attempt. `app.*` imports stay inside
each test (conftest purges `app.*` between tests).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _settings(**over):
    from app.settings import ThoughtSettings

    return ThoughtSettings(NOTIFY_BASE_URL="http://notify.test:7140", NOTIFY_API_TOKEN="", **over)


def _reasons(client_cls):
    return [(c.kwargs["context"]["reason"], c.kwargs["severity"])
            for c in client_cls.return_value.attention_request.call_args_list]


def test_defaults():
    s = _settings()
    assert s.visual_attempt_streak_check_enabled is True
    assert s.visual_attempt_failure_streak_threshold == 6


@pytest.mark.parametrize("streak,healthy", [(0, True), (5, True), (6, False), (35, False)])
def test_streak_threshold(streak, healthy):
    from app.visual_chain_health_monitor import _attempt_streak_check

    check = _attempt_streak_check(streak=streak, threshold=6)
    assert check.key == "visual_attempt_failure_streak"
    assert check.healthy is healthy and check.severity == "critical"


def test_the_2026_10_16_state_raises_a_critical_card_by_the_sixth_failed_attempt():
    """Replays 10-16 as the watchdog saw it: attempts 04:34 .. 18:56 abandoned, gap growing.
    Before this patch the only card was ("visual_painting_gap", "error")."""
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get",
        return_value=MagicMock(status_code=200, json=lambda: []),
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        m = VisualChainHealthMonitor(settings_obj=_settings())
        # (hours since the 03:01 painting, failed attempts so far)
        for gap_h, streak in [(1.6, 1), (3.2, 2), (4.0, 3), (5.5, 4), (9.0, 5), (12.2, 5), (15.9, 6),
                              (40.0, 20), (98.0, 35)]:
            m.record_painting_gap(age_hours=gap_h)
            m.record_attempt_streak(streak=streak)
        assert ("visual_attempt_failure_streak", "critical") in _reasons(client_cls)
        assert _reasons(client_cls).count(("visual_attempt_failure_streak", "critical")) == 1
        msg = next(c.kwargs["message"] for c in client_cls.return_value.attention_request.call_args_list
                   if c.kwargs["context"]["reason"] == "visual_attempt_failure_streak")
        assert "6 painting attempts in a row" in msg and "gpu2" in msg

        m.record_attempt_streak(streak=0)              # an attempt produced an image
        assert _reasons(client_cls)[-1] == ("visual_attempt_failure_streak", "info")


def test_a_failed_read_is_no_observation():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get",
        return_value=MagicMock(status_code=200, json=lambda: []),
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        m = VisualChainHealthMonitor(settings_obj=_settings())
        m.record_attempt_streak(streak=8)
        m.record_attempt_streak(streak=None)
        m.record_attempt_streak(streak=9)
        assert _reasons(client_cls) == [("visual_attempt_failure_streak", "critical")]


@pytest.mark.asyncio
async def test_the_gap_loop_also_reads_and_checks_the_streak(monkeypatch):
    import asyncio

    from app import visual_chain_health_monitor as mon

    cfg = _settings(ORION_VISUAL_CHAIN_ENABLED=False, ORION_BUS_ENABLED=False,
                    ORION_VISUAL_PAINTING_GAP_CHECK_INTERVAL_SEC=0.01)
    stop = asyncio.Event()
    streaks = iter([3, 7])
    seen = []

    def fake_check(streak):
        seen.append(streak)
        if len(seen) >= 2:
            stop.set()

    monkeypatch.setattr("app.store.visual_last_painting_age_hours", lambda: 1.0)
    monkeypatch.setattr("app.store.visual_attempt_failure_streak", lambda: next(streaks))
    monkeypatch.setattr(mon, "check_visual_painting_gap", lambda age: None)
    monkeypatch.setattr(mon, "check_visual_attempt_streak", fake_check)
    await asyncio.wait_for(mon.run_visual_painting_gap_watchdog(stop, settings_obj=cfg), 2.0)
    assert seen == [3, 7]
