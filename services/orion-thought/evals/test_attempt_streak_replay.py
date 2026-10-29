"""Eval: replay every real painting attempt through the failed-attempt streak check.

Fixture `fixtures/visual_attempts_2026-09-14_to_10-19.json`: every row of live
`reverie_visual_attempt` (started_at, outcome), 405 attempts, 2026-09-14 02:22Z -> 10-19 08:07Z,
read 2026-10-20 (the table's whole history).

Honesty notes:
- The threshold (6) was picked IN-SAMPLE on this data: one more than the longest ordinary streak
  (5). This proves the check behaves as calibrated; it is not an out-of-sample test.
- Streaks >= 6: 35 from 10-16 04:34 (gpu2 faulted, confirmed), 16 from 10-09 03:57 (the lane
  controller refusing every swap, confirmed), 9 from 10-08 05:00 and 9 from 09-25 08:02 (causes
  unknown, both inside painting gaps the gap check also flagged).

Run: pytest services/orion-thought/evals -q
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

FIXTURE = Path(__file__).parent / "fixtures" / "visual_attempts_2026-09-14_to_10-19.json"
INCIDENT_10_16 = "2026-10-16 04:34:49.693115+00"


def _attempts():
    return json.loads(FIXTURE.read_text())["attempts"]


def _replay(threshold: int) -> list[str]:
    """Start time of the first attempt of each streak that raised a card."""
    from app.settings import ThoughtSettings

    s = ThoughtSettings(NOTIFY_BASE_URL="http://notify.test:7140", NOTIFY_API_TOKEN="",
                        ORION_VISUAL_ATTEMPT_FAILURE_STREAK_THRESHOLD=threshold)
    alerted: list[str] = []
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get",
        return_value=MagicMock(status_code=200, json=lambda: []),
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        m = VisualChainHealthMonitor(settings_obj=s)
        m.record_attempt_streak(streak=0)
        streak, start = 0, None
        for a in _attempts():
            if a["outcome"] == "produced":
                streak, start = 0, None
            else:
                start = start or a["started_at"]
                streak += 1
            before = [c.kwargs["severity"] for c in client_cls.return_value.attention_request.call_args_list]
            m.record_attempt_streak(streak=streak)
            after = [c.kwargs["severity"] for c in client_cls.return_value.attention_request.call_args_list]
            if len(after) > len(before) and after[-1] == "critical":
                alerted.append(start)
    return alerted


def test_fixture_is_the_whole_history():
    data = json.loads(FIXTURE.read_text())
    assert data["attempt_count"] == len(data["attempts"]) == 405


def test_default_threshold_flags_exactly_the_four_outage_streaks():
    from app.settings import ThoughtSettings

    assert ThoughtSettings().visual_attempt_failure_streak_threshold == 6
    assert _replay(6) == ["2026-09-25 08:02:18.586066+00", "2026-10-08 05:00:28.056841+00",
                          "2026-10-09 03:57:02.98324+00", INCIDENT_10_16]


def test_threshold_sensitivity():
    assert len(_replay(5)) > 4          # 5 starts paging on ordinary streaks
    assert INCIDENT_10_16 in _replay(12)
