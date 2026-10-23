"""Strain report replay over a fixed arousal history: flickers never report, one per episode,
one per local day, quiet hours delay to 08:00. Live run: run_strain_report_replay.py (no flags)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from run_strain_report_replay import episodes_from_rows, replay

TZ = ZoneInfo("America/Denver")


def _row(local: str, to: str, since_local: str | None = None):
    at = datetime.fromisoformat(local).replace(tzinfo=TZ).astimezone(timezone.utc)
    since = datetime.fromisoformat(since_local or local).replace(tzinfo=TZ).astimezone(timezone.utc)
    return {"occurred_at": at.isoformat(), "payload_json": {"to": to, "reading": {"since": since.isoformat()}}}


ROWS = [
    _row("2026-10-20T09:00", "strained"), _row("2026-10-20T09:30", "idle"),      # flicker: 30 min
    _row("2026-10-20T10:00", "strained"), _row("2026-10-20T12:00", "idle"),      # 2 h: report 11:00
    _row("2026-10-20T15:00", "strained"), _row("2026-10-20T17:00", "idle"),      # 2 h: day cap
    _row("2026-10-20T22:30", "strained"), _row("2026-10-21T09:00", "idle"),      # quiet: 08:00 next day
    _row("2026-10-21T10:00", "strained"),                                        # restart, same since:
    _row("2026-10-21T10:05", "strained", "2026-10-21T10:00"),                    # one episode
    _row("2026-10-21T13:00", "idle"),
]


def test_replay_counts_per_day():
    eps = episodes_from_rows(ROWS, now=datetime(2026, 10, 22, tzinfo=timezone.utc))
    assert len(eps) == 5
    out = replay(eps, min_episode_sec=3600.0, quiet=(23, 8), tz=TZ)
    d20, d21 = out["days"]["2026-10-20"], out["days"]["2026-10-21"]
    assert [r["sent_at_local"] for r in d20["reports"]] == ["2026-10-20 11:00"]
    assert d20["held_by_day_cap"] == 1
    # The overnight episode reports at 08:00; that uses 10-21's one report, so the 10:00 episode waits.
    assert [r["sent_at_local"] for r in d21["reports"]] == ["2026-10-21 08:00"]
    assert d21["reports"][0]["waited_for_quiet_hours"] is True
    assert d21["held_by_day_cap"] == 1
    assert out["reports_total"] == 2


def test_open_episode_runs_to_now():
    now = datetime(2026, 10, 20, 20, 0, tzinfo=timezone.utc)
    eps = episodes_from_rows([_row("2026-10-20T10:00", "strained")], now=now)
    assert eps[0].still_open and eps[0].end == now
    assert timedelta(hours=1) < eps[0].end - eps[0].since
