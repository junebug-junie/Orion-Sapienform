"""Eval: replay 21 days of real painting gaps through the painting-gap monitor.

Fixture `fixtures/painting_gaps_21d.json` is every gap between consecutive
produced paintings (production_receipt.produced_at) in live Postgres over the
21 days ending 2026-10-10: 175 paintings, 174 gaps. Only three were real
outages -- 27.6 h (GPU lane controller refusing swaps, ending 10-10 06:02),
17.8 h (ending 10-08 21:04), 16.0 h (ending 09-25 22:19); the next longest is
7.6 h. Each gap is replayed as watchdog ticks every 10 minutes (the default
ORION_VISUAL_CHAIN_WATCHDOG_CHECK_INTERVAL_SEC) through the real monitor, and
at the default 12 h threshold exactly those three must alert, once each, each
followed by one recovery note.

Run: pytest services/orion-thought/evals -q
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

FIXTURE = Path(__file__).parent / "fixtures" / "painting_gaps_21d.json"
TICK_H = 600 / 3600.0
EXPECTED_OUTAGES = {"2026-10-10T06:02:52Z", "2026-10-08T21:04:01Z", "2026-09-25T22:19:58Z"}


def _replay(threshold_hours: float) -> tuple[list[str], int]:
    from app.settings import ThoughtSettings

    gaps = json.loads(FIXTURE.read_text())["gaps"]
    settings_obj = ThoughtSettings(
        NOTIFY_BASE_URL="http://notify.test:7140",
        NOTIFY_API_TOKEN="",
        ORION_VISUAL_PAINTING_GAP_THRESHOLD_HOURS=threshold_hours,
    )
    alerted: list[str] = []
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get",
        return_value=MagicMock(status_code=200, json=lambda: []),
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        monitor = VisualChainHealthMonitor(settings_obj=settings_obj)
        monitor.record_painting_gap(age_hours=0.0)
        for gap in gaps:
            before = client_cls.return_value.attention_request.call_count
            t = TICK_H
            while t < gap["hours"]:
                monitor.record_painting_gap(age_hours=t)
                t += TICK_H
            if client_cls.return_value.attention_request.call_count > before:
                alerted.append(gap["ended_at"])
            monitor.record_painting_gap(age_hours=0.0)  # the painting that ended the gap
        calls = client_cls.return_value.attention_request.call_args_list
    return alerted, len(calls)


def test_fixture_is_the_calibration_series():
    data = json.loads(FIXTURE.read_text())
    assert data["painting_count"] == 175
    assert len(data["gaps"]) == 174
    longest = sorted((g["hours"] for g in data["gaps"]), reverse=True)[:4]
    assert [round(h, 1) for h in longest] == [27.6, 17.8, 16.0, 7.6]


def test_default_12h_threshold_alerts_on_exactly_the_three_real_outages():
    from app.settings import ThoughtSettings

    threshold = ThoughtSettings().visual_painting_gap_threshold_hours
    assert threshold == 12.0
    alerted, total_calls = _replay(threshold)
    assert set(alerted) == EXPECTED_OUTAGES
    assert len(alerted) == 3
    assert total_calls == 6  # 3 alerts + 3 recovery notes, no repeats


def test_threshold_sensitivity_is_documented_by_the_series():
    # Too low (6 h) starts paging on ordinary quiet stretches; too high (20 h)
    # misses two of the three real outages.
    assert len(_replay(6.0)[0]) > 3
    assert _replay(20.0)[0] == ["2026-10-10T06:02:52Z"]
