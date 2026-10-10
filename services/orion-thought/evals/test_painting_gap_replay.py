"""Eval: replay every real painting gap through the painting-gap monitor.

Fixture `fixtures/painting_gaps_2026-09-14_to_10-10.json`: every gap between
consecutive produced paintings (production_receipt.produced_at) in live
Postgres from the first receipt-bearing painting (2026-09-14 02:23Z) to the
end of the 2026-10-09/10 incident (10-10 06:02Z) -- ~26.2 days, 170
paintings, 169 gaps. (1,433 earlier paintings, 08-25 -> 09-08, predate
receipts and are not in it.)

Honesty notes:
- The 12 h threshold was picked IN-SAMPLE on this same data. This eval
  proves the monitor behaves as calibrated on it; it is not an out-of-sample
  test of the threshold.
- Only one alerting gap has a confirmed cause: 27.6 h ending 10-10 06:02
  (GPU lane controller refusing every swap). The 17.8 h (ending 10-08 21:04),
  16.0 h (ending 09-25 22:19) and 14.7 h (ending 09-14 17:07, the first day
  of receipts) gaps have unknown causes. The 247 h (~10.3 day) stretch
  09-14 17:07 -> 09-25 00:04 also alerts, cause also unknown.
- The margin is the empty band between the longest non-alerting gap (7.6 h)
  and the shortest alerting one (14.7 h).

Each gap is replayed as watchdog ticks every 10 minutes (default
ORION_VISUAL_PAINTING_GAP_CHECK_INTERVAL_SEC) through the real monitor.

Run: pytest services/orion-thought/evals -q
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

FIXTURE = Path(__file__).parent / "fixtures" / "painting_gaps_2026-09-14_to_10-10.json"
TICK_H = 600 / 3600.0
CONFIRMED_INCIDENT = "2026-10-10T06:02:52Z"
EXPECTED_ALERTS_AT_12H = {
    "2026-09-14T17:07:18Z",  # 14.7 h, cause unknown (first day of receipts)
    "2026-09-25T00:04:26Z",  # 247 h / ~10.3 days, cause unknown
    "2026-09-25T22:19:58Z",  # 16.0 h, cause unknown
    "2026-10-08T21:04:01Z",  # 17.8 h, cause unknown
    CONFIRMED_INCIDENT,  # 27.6 h, GPU lane controller refusing swaps
}


def _gaps():
    return json.loads(FIXTURE.read_text())["gaps"]


def _replay(threshold_hours: float) -> tuple[list[str], int]:
    from app.settings import ThoughtSettings

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
        for gap in _gaps():
            before = client_cls.return_value.attention_request.call_count
            t = TICK_H
            while t < gap["hours"]:
                monitor.record_painting_gap(age_hours=t)
                t += TICK_H
            if client_cls.return_value.attention_request.call_count > before:
                alerted.append(gap["ended_at"])
            monitor.record_painting_gap(age_hours=0.0)  # the painting that ended the gap
        total = client_cls.return_value.attention_request.call_count
    return alerted, total


def test_fixture_is_the_calibration_series():
    data = json.loads(FIXTURE.read_text())
    assert data["painting_count"] == 170
    assert len(data["gaps"]) == 169
    assert data["first_painting"] == "2026-09-14T02:23:26Z"
    assert data["last_painting"] == CONFIRMED_INCIDENT
    longest = sorted((g["hours"] for g in data["gaps"]), reverse=True)[:6]
    assert [round(h, 1) for h in longest] == [247.0, 27.6, 17.8, 16.0, 14.7, 7.6]


def test_default_12h_threshold_alerts_on_every_gap_over_14h_and_nothing_under_8h():
    from app.settings import ThoughtSettings

    threshold = ThoughtSettings().visual_painting_gap_threshold_hours
    assert threshold == 12.0
    alerted, total = _replay(threshold)
    assert set(alerted) == EXPECTED_ALERTS_AT_12H
    assert len(alerted) == 5
    assert CONFIRMED_INCIDENT in alerted
    assert total == 10  # one alert + one recovery per gap, no repeats


def test_threshold_sensitivity_on_the_same_series():
    # 6 h starts paging on ordinary quiet stretches (the 7.4/7.6 h gaps);
    # 20 h would still catch the confirmed incident but drop the 16-18 h ones.
    assert len(_replay(6.0)[0]) > 5
    assert set(_replay(20.0)[0]) == {"2026-09-25T00:04:26Z", CONFIRMED_INCIDENT}
