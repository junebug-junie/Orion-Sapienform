"""app/guards.py: the swap-load guards read what durable-runs' elastic path reads today, and every
way a read can fail blocks (a named reason), never passes silently."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from app.guards import GuardReader

NOW = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
CAB, VIS = "http://cab/latest", "http://thought/visual-chain/activity"


class Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.body


class Client:
    def __init__(self, routes):
        self.routes = routes

    async def get(self, url):
        r = self.routes[url]
        if isinstance(r, Exception):
            raise r
        return r


def cabinet(temp, age=5):
    return Resp({"snapshot": {"frame": {"environment": {"temp_c": temp}}}, "age_sec": age})


def activity(**kw):
    body = {"observed_at": NOW.isoformat(), "history_status": "ok",
            "last_success_at": (NOW - timedelta(minutes=10)).isoformat()}
    body.update(kw)
    return Resp(body)


def read(routes, reader=None):
    reader = reader or GuardReader(cabinet_url=CAB, visual_activity_url=VIS)
    return asyncio.run(reader.read(Client(routes), NOW))


def test_clear_room_and_fresh_baseline_pass():
    assert read({CAB: cabinet(22.0), VIS: activity()}) == {"thermal": None, "visual_baseline": None}


def test_hot_room_blocks_and_stays_blocked_until_rearm():
    reader = GuardReader(cabinet_url=CAB, visual_activity_url=VIS)
    hot = read({CAB: cabinet(40.0), VIS: activity()}, reader)["thermal"]
    assert hot and hot.startswith("hot")
    assert read({CAB: cabinet(31.0), VIS: activity()}, reader)["thermal"] is not None   # under hot, above rearm: still hot
    assert read({CAB: cabinet(22.0), VIS: activity()}, reader)["thermal"] is None


def test_unreadable_or_stale_thermal_blocks():
    assert read({CAB: RuntimeError("down"), VIS: activity()})["thermal"].startswith("unavailable")
    assert read({CAB: Resp({}, 200), VIS: activity()})["thermal"].startswith("unavailable")
    assert read({CAB: cabinet(22.0, age=10_000), VIS: activity()})["thermal"].startswith("degraded")


def test_visual_baseline_overdue_running_or_unreadable_blocks():
    overdue = activity(last_success_at=(NOW - timedelta(hours=3)).isoformat())
    assert read({CAB: cabinet(22.0), VIS: overdue})["visual_baseline"] == "visual_baseline_urgent"
    assert read({CAB: cabinet(22.0), VIS: activity(last_success_at=None)})["visual_baseline"] == "visual_baseline_urgent"
    assert read({CAB: cabinet(22.0), VIS: activity(active_attempt_id="a1")})["visual_baseline"] == "visual_attempt_running"
    stale = activity(observed_at=(NOW - timedelta(hours=1)).isoformat())
    assert read({CAB: cabinet(22.0), VIS: stale})["visual_baseline"] == "visual_activity_unavailable"
    assert read({CAB: cabinet(22.0), VIS: Resp({}, 503)})["visual_baseline"].startswith("unavailable")
