"""app/guards.py: the swap-load guard reads what durable-runs' elastic path read (the cabinet
sensor), and every way a read can fail blocks (a named reason), never passes silently. The stage-4
visual_baseline guard was deleted in stage 5.4; test_visual_baseline_guard_is_gone pins that."""
from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from app.guards import GuardReader
from app.settings import Settings

CAB = "http://cab/latest"


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
        self.gets: list[str] = []

    async def get(self, url):
        self.gets.append(url)
        r = self.routes[url]
        if isinstance(r, Exception):
            raise r
        return r


def cabinet(temp, age=5):
    return Resp({"snapshot": {"frame": {"environment": {"temp_c": temp}}}, "age_sec": age})


def read(routes, reader=None):
    reader = reader or GuardReader(cabinet_url=CAB)
    return asyncio.run(reader.read(Client(routes)))


def test_clear_room_passes():
    assert read({CAB: cabinet(22.0)}) == {"thermal": None}


def test_hot_room_blocks_and_stays_blocked_until_rearm():
    reader = GuardReader(cabinet_url=CAB)
    hot = read({CAB: cabinet(40.0)}, reader)["thermal"]
    assert hot and hot.startswith("hot")
    assert read({CAB: cabinet(31.0)}, reader)["thermal"] is not None   # under hot, above rearm: still hot
    assert read({CAB: cabinet(22.0)}, reader)["thermal"] is None


def test_unreadable_or_stale_thermal_blocks():
    assert read({CAB: RuntimeError("down")})["thermal"].startswith("unavailable")
    assert read({CAB: Resp({}, 200)})["thermal"].startswith("unavailable")
    assert read({CAB: cabinet(22.0, age=10_000)})["thermal"].startswith("degraded")


def test_visual_baseline_guard_is_gone():
    """Stage 5.4: the pool no longer reads thought's /visual-chain/activity at all, the guard name
    is not accepted in config, and its URL setting is gone (kill means kill)."""
    client = Client({CAB: cabinet(22.0)})
    asyncio.run(GuardReader(cabinet_url=CAB).read(client))
    assert client.gets == [CAB]
    assert "visual_activity_url" not in Settings.model_fields
    from orion.gpu_pool.config import SWAP_GUARDS, SwapSpec
    assert SWAP_GUARDS == ("thermal",)
    with pytest.raises(ValidationError):
        SwapSpec.model_validate({"evicts": ["diffusion"], "guards": ["visual_baseline"]})


# --- thermal controller v2, D10 / C10 / C13 -------------------------------------------------------

def test_starts_unknown_not_hot():
    reader = GuardReader(cabinet_url=CAB)
    assert reader.thermal_state == "unknown"
    assert read({CAB: RuntimeError("down")}, reader)["thermal"] == "unavailable:RuntimeError"


def test_one_failed_read_inside_grace_holds_the_last_state():
    from datetime import datetime, timedelta, timezone
    t = [datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)]
    reader = GuardReader(cabinet_url=CAB, clock=lambda: t[0])
    assert read({CAB: cabinet(22.0, age=5)}, reader)["thermal"] is None
    t[0] += timedelta(seconds=60)
    assert read({CAB: RuntimeError("blip")}, reader)["thermal"] is None        # C10: one blip is not unknown
    t[0] += timedelta(seconds=300)
    assert read({CAB: RuntimeError("down")}, reader)["thermal"].startswith("unavailable")   # past grace


def test_elevated_does_not_block_swap_loads_hot_does():
    """The 32 C hot line is the guard's (Decisions 2026-10-06); 34 C is the reflex shed's."""
    assert read({CAB: cabinet(30.0)})["thermal"] is None
    assert read({CAB: cabinet(32.0)})["thermal"].startswith("hot")
