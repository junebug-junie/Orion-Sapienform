"""Hub's chronicle read for the kickoff's "today so far" line degrades to no line."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx

import scripts.curiosity_investigation as ci
from orion.core.bus.bus_schemas import ServiceRef
from orion.curiosity.worldview import RecentRun, WorldviewSnapshot
from scripts.curiosity_investigation import CuriosityInvestigation

SOURCE = ServiceRef(name="orion-hub", version="0.1.0", node="athena")
NOW = datetime(2026, 10, 14, 3, 0, tzinfo=timezone.utc)
VIEW = WorldviewSnapshot(recent_runs=[RecentRun(run_id="r1"), RecentRun(run_id="r9")])
FRAME = {"day_id": "2026-10-13", "as_of": (NOW - timedelta(seconds=30)).isoformat(),
         "arcs_today": [], "arcs_today_total": 300}
ARCS = {"day_id": "2026-10-13", "count": 1, "arcs": [{
    "arc_id": "a1", "kind": "curiosity", "day_id": "2026-10-13", "subject_ref": "r1",
    "began_at": "2026-10-13T08:00:00+00:00", "cumulative_dwell_sec": 600.0}]}


def _loop(*, enabled=True, url="http://durable.test") -> CuriosityInvestigation:
    return CuriosityInvestigation(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=60.0, daily_cap=3,
        timeout_sec=60.0, session_id="s", pool_provider=lambda: None, source_ref=SOURCE,
        reader=object(), durable_runs_url=url, temporal_self_fact_enabled=enabled,
    )


def _patch_client(monkeypatch, handler):
    real = httpx.AsyncClient
    seen = {}

    def factory(*args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return real(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(ci.httpx, "AsyncClient", factory)
    return seen


def test_renders_the_fact_from_frame_and_arcs(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/temporal-self/frame":
            return httpx.Response(200, json=FRAME)
        return httpx.Response(200, json=ARCS)

    seen = _patch_client(monkeypatch, handler)
    line = asyncio.run(_loop()._read_today_fact(VIEW, NOW))
    assert line == ("TODAY SO FAR (day record as of 20:59 local): 1 curiosity run started today "
                    "and finished, 10 min in total. 1 of the earlier runs listed above is among "
                    "them, 10 min of that. [arc:a1]")
    assert seen["timeout"] is not None and seen["timeout"] <= 1.5
    assert sorted(calls) == ["http://durable.test/temporal-self/arcs?kind=curiosity",
                             "http://durable.test/temporal-self/frame"]


def test_fetch_failure_is_no_line(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    _patch_client(monkeypatch, handler)
    assert asyncio.run(_loop()._read_today_fact(VIEW, NOW)) is None


def test_404_frame_is_no_line(monkeypatch):
    _patch_client(monkeypatch, lambda request: httpx.Response(404, json={"detail": "none"}))
    assert asyncio.run(_loop()._read_today_fact(VIEW, NOW)) is None


def test_bad_json_is_no_line(monkeypatch):
    _patch_client(monkeypatch, lambda request: httpx.Response(200, content=b"<html>"))
    assert asyncio.run(_loop()._read_today_fact(VIEW, NOW)) is None


def test_arcs_failure_falls_back_to_capped_frame_and_says_nothing(monkeypatch):
    def handler(request):
        if request.url.path == "/temporal-self/frame":
            return httpx.Response(200, json=FRAME)
        raise httpx.ReadTimeout("slow", request=request)

    _patch_client(monkeypatch, handler)
    assert asyncio.run(_loop()._read_today_fact(VIEW, NOW)) is None


def test_slow_service_hits_the_deadline(monkeypatch):
    monkeypatch.setattr(ci, "TEMPORAL_SELF_FACT_TIMEOUT_SEC", 0.05)

    async def handler(request):
        await asyncio.sleep(1.0)
        return httpx.Response(200, json=FRAME)

    _patch_client(monkeypatch, handler)
    started = asyncio.run(_timed(_loop()._read_today_fact(VIEW, NOW)))
    assert started[0] is None and started[1] < 0.5


async def _timed(coro):
    t = asyncio.get_running_loop().time()
    out = await coro
    return out, asyncio.get_running_loop().time() - t


def test_flag_off_or_no_url_makes_no_call(monkeypatch):
    def handler(request):
        raise AssertionError("no call expected")

    _patch_client(monkeypatch, handler)
    assert asyncio.run(_loop(enabled=False)._read_today_fact(VIEW, NOW)) is None
    assert asyncio.run(_loop(url="")._read_today_fact(VIEW, NOW)) is None


def test_settings_ship_the_flag_on():
    from app.settings import Settings

    assert Settings.model_fields["HUB_CURIOSITY_TEMPORAL_SELF_FACT_ENABLED"].default is True


def test_investigate_hands_the_fact_to_the_kickoff_prompt():
    """Removing the kwarg from `_investigate`'s build_kickoff_prompt call must fail a test."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(CuriosityInvestigation._investigate)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "build_kickoff_prompt"]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kw.get("today_fact"), ast.Name) and kw["today_fact"].id == "today_fact"
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "_read_today_fact"]
    assert reads, "_investigate no longer reads the fact"
