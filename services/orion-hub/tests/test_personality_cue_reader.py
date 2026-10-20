"""Hub's personality frame read degrades to an empty cue, never an error (personality graph rev 2)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx

import scripts.personality_cue_reader as pcr
from orion.personality.cue import HEADER
from scripts.personality_cue_reader import cue_status, read_personality_cue

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
BASE = "http://durable.test"
REAL_CLIENT = httpx.AsyncClient  # captured once: re-patching must not wrap an earlier patch


def frame(generated_at: datetime = NOW - timedelta(minutes=5), nodes: int = 2) -> dict:
    return {
        "generated_at": generated_at.isoformat(),
        "dispositions": [{
            "disposition_id": f"d{i}",
            "label": f"the walkway camera sees presence without a person {i}",
            "origin": "recurrence",
            "status": "active",
            "weight": 10.0 - i,
            "evidence_count": 7 + i,
            "lane_counts": {"reverie": 4, "dream": 2},
            "first_seen_at": "2026-10-16T08:00:00+00:00",
            "last_reinforced_at": "2026-10-20T11:00:00+00:00",
        } for i in range(nodes)],
    }


def patch_client(monkeypatch, handler) -> list:
    real = REAL_CLIENT
    calls: list = []

    def factory(*args, **kwargs):
        calls.append(kwargs.get("timeout"))
        return real(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(pcr.httpx, "AsyncClient", factory)
    return calls


def test_valid_frame_renders_the_cue(monkeypatch):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=frame())

    timeouts = patch_client(monkeypatch, handler)
    cue = asyncio.run(read_personality_cue(BASE + "/", 1.5, now=NOW))
    assert seen == ["http://durable.test/personality/frame"]
    assert timeouts == [1.5]
    assert cue.skip_reason is None and cue.text.startswith(HEADER)
    assert cue.disposition_ids == ["d0", "d1"]
    assert cue_status(cue) == "rendered=True ids=d0,d1 skip=-"


def test_disabled_makes_no_call(monkeypatch):
    def handler(request):
        raise AssertionError("no call expected")

    patch_client(monkeypatch, handler)
    cue = asyncio.run(read_personality_cue(BASE, 1.5, now=NOW, enabled=False))
    assert (cue.text, cue.skip_reason) == ("", "disabled")
    assert cue_status(cue) == "rendered=False ids=- skip=disabled"


def test_no_url_is_fetch_failed_without_a_call(monkeypatch):
    def handler(request):
        raise AssertionError("no call expected")

    patch_client(monkeypatch, handler)
    cue = asyncio.run(read_personality_cue("", 1.5, now=NOW))
    assert (cue.text, cue.skip_reason) == ("", "fetch_failed")


def test_connection_error_is_fetch_failed(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    patch_client(monkeypatch, handler)
    cue = asyncio.run(read_personality_cue(BASE, 1.5, now=NOW))
    assert (cue.text, cue.skip_reason) == ("", "fetch_failed")


def test_404_and_bad_json_are_fetch_failed(monkeypatch):
    patch_client(monkeypatch, lambda r: httpx.Response(404, json={"detail": "no frame yet"}))
    assert asyncio.run(read_personality_cue(BASE, 1.5, now=NOW)).skip_reason == "fetch_failed"
    patch_client(monkeypatch, lambda r: httpx.Response(200, content=b"<html>"))
    assert asyncio.run(read_personality_cue(BASE, 1.5, now=NOW)).skip_reason == "fetch_failed"


def test_invalid_stale_and_empty_frames_render_nothing(monkeypatch):
    patch_client(monkeypatch, lambda r: httpx.Response(200, json={"generated_at": "nope"}))
    assert asyncio.run(read_personality_cue(BASE, 1.5, now=NOW)).skip_reason == "invalid_frame"
    patch_client(monkeypatch, lambda r: httpx.Response(200, json=frame()))
    stale = asyncio.run(read_personality_cue(BASE, 1.5, now=NOW + timedelta(hours=2)))
    assert (stale.text, stale.skip_reason) == ("", "stale")
    patch_client(monkeypatch, lambda r: httpx.Response(200, json=frame(nodes=0)))
    assert asyncio.run(read_personality_cue(BASE, 1.5, now=NOW)).skip_reason == "empty"


def test_slow_service_hits_the_deadline(monkeypatch):
    async def handler(request):
        await asyncio.sleep(1.0)
        return httpx.Response(200, json=frame())

    patch_client(monkeypatch, handler)

    async def timed():
        loop = asyncio.get_running_loop()
        t = loop.time()
        cue = await read_personality_cue(BASE, 0.05, now=NOW)
        return cue, loop.time() - t

    cue, elapsed = asyncio.run(timed())
    assert cue.skip_reason == "fetch_failed" and elapsed < 0.5


def test_renderer_crash_is_contained(monkeypatch):
    patch_client(monkeypatch, lambda r: httpx.Response(200, json=frame()))

    def boom(*a, **k):
        raise RuntimeError("renderer broke its promise")

    monkeypatch.setattr(pcr, "project_personality_cue", boom)
    cue = asyncio.run(read_personality_cue(BASE, 1.5, now=NOW))
    assert (cue.text, cue.skip_reason) == ("", "render_failed")


def test_naive_now_is_treated_as_utc(monkeypatch):
    patch_client(monkeypatch, lambda r: httpx.Response(200, json=frame()))
    cue = asyncio.run(read_personality_cue(BASE, 1.5, now=NOW.replace(tzinfo=None)))
    assert cue.skip_reason is None and cue.text


def test_settings_ship_the_flag_on():
    from app.settings import Settings

    assert Settings.model_fields["HUB_PERSONALITY_CUE_ENABLED"].default is True
