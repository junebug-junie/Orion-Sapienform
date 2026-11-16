"""The shared now client (integration patch 3): never raises, a real run reads without since, no cites no call."""

from __future__ import annotations

import asyncio
import time
import urllib.error
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

from orion.integration import now_client as nc

T = datetime(2026, 11, 16, 1, 0, tzinfo=timezone.utc)
NOW = {"shown_id": "s1", "reader_organ": "prime", "since": "2026-11-16T00:00:00Z", "as_of": "2026-11-16T01:00:00Z",
       "items": [{"event_id": "vision_percept:a", "organ": "camera", "source_kind": "vision_percept",
                  "label": "saw a dog", "occurred_at": "2026-11-16T00:50:00Z", "salience": None}]}


def test_a_real_read_sends_no_since_and_carries_the_run_ref():
    seen = {}

    def get(url, timeout):
        seen["url"], seen["timeout"] = url, timeout
        return NOW

    got = nc.read_now("http://dr:8121/", "prime", run_ref="prime_thought:abc", timeout_sec=2.0, get=get)
    q = parse_qs(urlparse(seen["url"]).query)
    assert urlparse(seen["url"]).path == "/temporal-self/now"
    assert q == {"organ": ["prime"], "run_ref": ["prime_thought:abc"]}
    assert seen["timeout"] == 2.0
    assert got.error is None and got.shown_ids == ["vision_percept:a"]


def test_an_inspection_read_passes_since():
    seen = {}
    nc.read_now("http://dr", "reverie", since=T, get=lambda u, t: seen.setdefault("url", u) and NOW)
    assert parse_qs(urlparse(seen["url"]).query)["since"] == [T.isoformat()]


def test_a_failed_read_is_no_now_with_a_reason_never_a_raise():
    def boom(exc):
        def get(url, timeout):
            raise exc
        return get

    assert nc.read_now("http://dr", "prime", get=boom(TimeoutError())).error == "timeout"
    assert nc.read_now("http://dr", "prime", get=boom(urllib.error.URLError(TimeoutError()))).error == "timeout"
    off = urllib.error.HTTPError("u", 404, "off", {}, None)
    assert nc.read_now("http://dr", "prime", get=boom(off)).error == "http 404"
    assert nc.read_now("http://dr", "prime", get=lambda u, t: {"nope": 1}).error.startswith("not a TemporalNowV1")
    assert nc.read_now("", "prime").error == "no durable-runs URL configured"
    assert nc.read_now("http://dr", "prime", get=boom(ConnectionRefusedError())).now is None


def test_no_cites_means_no_call_and_cites_are_posted_once_each():
    calls = []

    def post(url, body, timeout):
        calls.append((url, body))
        return {"shown_id": "s1", "citing_organ": "prime", "uses": [], "rejected_not_shown": ["x"]}

    assert nc.record_use("http://dr", "s1", [], "prime_thought:1", post=post) == (None, None)
    assert calls == []
    res, err = nc.record_use("http://dr/", "s1", ["a", "a", "x"], "prime_thought:1", post=post)
    assert err is None and res.rejected_not_shown == ["x"]
    assert calls == [("http://dr/temporal-self/now/use",
                      {"shown_id": "s1", "citing_ref": "prime_thought:1", "cites": ["a", "x"]})]
    res, err = nc.record_use("http://dr", "s1", ["a"], "r", post=lambda *a: (_ for _ in ()).throw(TimeoutError()))
    assert res is None and err == "timeout"


def test_async_wrappers_are_bounded_even_when_the_call_hangs():
    def slow(url, timeout):
        time.sleep(1.0)
        return NOW

    async def main():
        t0 = time.monotonic()
        got = await nc.aread_now("http://dr", "prime", timeout_sec=0.1, get=slow)
        res = await nc.arecord_use("http://dr", "s1", ["a"], "r", timeout_sec=0.1,
                                   post=lambda u, b, t: time.sleep(1.0))
        return got, res, time.monotonic() - t0

    got, (res, err), waited = asyncio.run(main())
    assert got.now is None and got.error == "timeout"
    assert res is None and err == "timeout"
    assert waited < 1.5  # each call gave up at ~0.6 s; the hung threads finish on their own
