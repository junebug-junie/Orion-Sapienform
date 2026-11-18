"""advertise_at_startup: the process-start advertisement the readiness gate depends on.

Regression, 2026-10-06: readers advertised only when they first built a substrate store,
which many do lazily or never, so after a full redeploy only 3 of 9 keys existed and the
referent/assertion projectors never wrote anything.
"""

from __future__ import annotations

import threading
import time

from orion.substrate import reader_capability as rc


class TtlRedis:
    """A Redis stand-in with real expiry against an injectable clock."""

    def __init__(self, clock):
        self.clock = clock
        self.data = {}
        self.sets = 0

    def set(self, key, value, ex=None):
        self.sets += 1
        self.data[key] = (value, None if ex is None else self.clock() + ex)

    def get(self, key):
        value, expires = self.data.get(key, (None, None))
        if expires is not None and self.clock() >= expires:
            self.data.pop(key, None)
            return None
        return value


def _clock():
    now = [1000.0]
    return now, (lambda: now[0])


def test_advertise_sets_a_ttl_and_an_expired_key_closes_the_gate():
    now, clock = _clock()
    r = TtlRedis(clock)
    assert rc.advertise("redis://x", name="hub", client=r, ttl_s=1800)
    assert rc.readiness("redis://x", ("hub",), client=r).ready
    now[0] += 1799
    assert rc.readiness("redis://x", ("hub",), client=r).ready
    now[0] += 2
    state = rc.readiness("redis://x", ("hub",), client=r)
    assert not state.ready and state.missing == ("hub",)


def test_the_refresher_keeps_the_key_alive_and_a_dead_reader_drops_out():
    now, clock = _clock()
    r = TtlRedis(clock)
    stop = threading.Event()
    thread = rc.advertise_at_startup("redis://refresh", name="recall-refresh", interval_s=0.01,
                                     ttl_s=1800, stop=stop, client=r)
    assert thread is not None
    deadline = time.monotonic() + 2.0
    while r.sets < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert r.sets >= 3, "refresher did not re-advertise"
    now[0] += 1000  # a refresh lands after this: the key is renewed, not left to expire
    seen = r.sets
    while r.sets == seen and time.monotonic() < deadline:
        time.sleep(0.005)
    now[0] += 1000  # 2000s since first write, < 1800s since the last refresh
    assert rc.readiness("redis://x", ("recall-refresh",), client=r).ready
    stop.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    now[0] += 1801  # reader is gone: nothing refreshes, the key expires
    assert not rc.readiness("redis://x", ("recall-refresh",), client=r).ready


def test_startup_never_raises_or_blocks_when_falkor_is_down():
    class Hung:
        def set(self, *_a, **_k):
            time.sleep(0.5)
            raise ConnectionError("down")

    stop = threading.Event()
    started = time.monotonic()
    thread = rc.advertise_at_startup("redis://down", name="down-reader", interval_s=60, stop=stop,
                                     client=Hung())
    assert time.monotonic() - started < 0.1, "startup blocked on Redis"
    assert thread is not None and thread.daemon
    stop.set()
    thread.join(timeout=2.0)


def test_no_uri_is_a_logged_no_op(monkeypatch):
    monkeypatch.delenv("FALKORDB_URI", raising=False)
    assert rc.advertise_at_startup(name="nobody") is None


def test_startup_is_idempotent_per_reader():
    now, clock = _clock()
    r = TtlRedis(clock)
    stop = threading.Event()
    a = rc.advertise_at_startup("redis://idem", name="idem", interval_s=60, stop=stop, client=r)
    b = rc.advertise_at_startup("redis://idem", name="idem", interval_s=60, stop=stop, client=r)
    assert a is b
    stop.set()
    a.join(timeout=1.0)


def test_unexpected_internal_errors_do_not_escape(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(rc.threading, "Thread", boom)
    assert rc.advertise_at_startup("redis://boom", name="boom") is None


def test_ttl_outlives_two_missed_refreshes():
    assert rc.CAPABILITY_TTL_S >= 2 * rc.ADVERTISE_INTERVAL_S
