"""AI Town session ledger arithmetic (orion/embodiment/town_sessions.py): the cap is per local day."""
from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from orion.embodiment import town_sessions as ts
from orion.schemas.aitown_session import AITOWN_SESSION_LEDGER_REDIS_KEY, AITownSessionRowV1

DEN = ZoneInfo("America/Denver")
CAP = 7200.0


def at(y, m, d, hh, mm=0, ss=0) -> float:
    return datetime(y, m, d, hh, mm, ss, tzinfo=DEN).timestamp()


def row(start, end=None, *, kind="orion", sid=None, last_seen=None, stop=None) -> AITownSessionRowV1:
    return AITownSessionRowV1(
        session_id=sid or f"s{start}", kind=kind, started_at=start,
        last_seen_at=last_seen if last_seen is not None else (end if end is not None else start),
        ended_at=end, start_reason="want_funded" if kind == "orion" else "juniper_override",
        stop_reason=stop if end is not None else None)


def gate(rows, now, **kw):
    args = dict(enabled=True, daily_cap_sec=CAP, min_session_sec=300.0, min_gap_sec=3600.0)
    args.update(kw)
    return ts.start_block_reason(rows, now, **args)


def test_a_session_across_midnight_is_split_between_the_two_days() -> None:
    r = row(at(2026, 10, 30, 23, 30), at(2026, 10, 31, 0, 30))
    assert ts.seconds_today([r], at(2026, 10, 30, 23, 59, 59), kind="orion") == 1799.0
    assert ts.seconds_today([r], at(2026, 10, 31, 12), kind="orion") == 1800.0


def test_the_cap_resets_at_local_midnight_not_utc_midnight() -> None:
    rows = [row(at(2026, 10, 30, 10), at(2026, 10, 30, 12))]           # 2 h on the 30th
    assert gate(rows, at(2026, 10, 30, 20)) == "daily_cap"
    # 18:00 local = 00:00 UTC on the 31st: still the 30th in Denver.
    assert gate(rows, at(2026, 10, 30, 18, 30)) == "daily_cap"
    assert gate(rows, at(2026, 10, 31, 0, 0, 1)) is None


def test_retry_after_a_daily_cap_is_the_next_local_midnight() -> None:
    rows = [row(at(2026, 10, 30, 10), at(2026, 10, 30, 12))]
    now = at(2026, 10, 30, 22)
    assert ts.retry_after_sec("daily_cap", rows, now, min_gap_sec=3600.0) == 7200.0


def test_a_dst_day_is_25_hours_long() -> None:
    noon = at(2026, 11, 1, 12)   # US DST ends 2026-11-01
    assert ts.next_local_midnight(noon) - ts.local_day_start(noon) == 25 * 3600


def test_an_open_session_mid_crossing_stops_only_on_the_new_days_count() -> None:
    """Yesterday used 1 h 55 min; a session that started 23:58 is not cut at 00:03: today starts at 0."""
    prev = [row(at(2026, 10, 30, 9), at(2026, 10, 30, 10, 55))]
    live = row(at(2026, 10, 30, 23, 58), sid="live")
    now = at(2026, 10, 31, 0, 3)
    assert ts.stop_reason(live, prev + [live], now, enabled=True, session_max_sec=1800.0, daily_cap_sec=CAP) is None
    # The same session, had the day not changed, hits the cap at 23:59 + 1 min of overlap.
    before = at(2026, 10, 30, 23, 59, 59)
    used = [row(at(2026, 10, 30, 9), at(2026, 10, 30, 10, 59))]   # 1 h 59 min
    assert ts.stop_reason(live, used + [live], before, enabled=True, session_max_sec=1800.0,
                          daily_cap_sec=CAP) == "daily_cap"


def test_stop_reasons_in_order() -> None:
    live = row(at(2026, 10, 30, 10), sid="live")
    assert ts.stop_reason(live, [live], at(2026, 10, 30, 10, 5), enabled=False, session_max_sec=1800.0,
                          daily_cap_sec=CAP) == "disabled"
    assert ts.stop_reason(live, [live], at(2026, 10, 30, 10, 30), enabled=True, session_max_sec=1800.0,
                          daily_cap_sec=CAP) == "session_cap"
    assert ts.stop_reason(live, [live], at(2026, 10, 30, 10, 29), enabled=True, session_max_sec=1800.0,
                          daily_cap_sec=CAP) is None


def test_start_gate_reasons() -> None:
    now = at(2026, 10, 30, 15)
    assert gate([], now, enabled=False) == "disabled"
    assert gate(None, now) == "ledger_unknown"
    assert gate([row(at(2026, 10, 30, 14, 50))], now) == "session_open"
    assert gate([row(at(2026, 10, 30, 14, 50), kind="manual")], now) == "manual_session_open"
    assert gate([row(at(2026, 10, 30, 14), at(2026, 10, 30, 14, 30))], now) == "cooldown"
    assert gate([row(at(2026, 10, 30, 13), at(2026, 10, 30, 13, 30))], now) is None
    # Less than min_session_sec left today: refused, not a 2-minute session.
    rows = [row(at(2026, 10, 30, 8), at(2026, 10, 30, 9, 56))]
    assert gate(rows, now) == "daily_cap"


def test_manual_time_is_not_counted_toward_orions_cap() -> None:
    rows = [row(at(2026, 10, 30, 8), at(2026, 10, 30, 14), kind="manual")]
    assert ts.seconds_today(rows, at(2026, 10, 30, 20), kind="orion") == 0.0
    assert gate(rows, at(2026, 10, 30, 20)) is None


def test_a_stale_open_row_counts_only_to_its_last_tick() -> None:
    r = row(at(2026, 10, 30, 10), last_seen=at(2026, 10, 30, 10, 10))
    assert ts.row_end(r, at(2026, 10, 30, 18)) == at(2026, 10, 30, 10, 10) + ts.STALE_SEC


def test_summarize_shapes_today_and_the_window() -> None:
    rows = [row(at(2026, 10, 30, 9), at(2026, 10, 30, 9, 30), stop="session_cap"),
            row(at(2026, 10, 30, 11), kind="manual", last_seen=at(2026, 10, 30, 11, 20))]
    now = at(2026, 10, 30, 11, 20)
    out = ts.summarize(rows, now, since=now - 86400, daily_cap_sec=CAP)
    assert out["local_date"] == "2026-10-30"
    assert out["orion_seconds_today"] == 1800.0 and out["manual_seconds_today"] == 1200.0
    assert out["open_kind"] == "manual"
    assert [s["kind"] for s in out["sessions"]] == ["manual", "orion"]
    assert out["sessions"][1]["stop_reason"] == "session_cap"


class FakeRedis:
    def __init__(self):
        self.h: dict = {}
        self.kv: dict = {}

    async def hgetall(self, key):
        return {k.encode(): v.encode() for k, v in self.h.get(key, {}).items()}

    async def hset(self, key, field, value):
        self.h.setdefault(key, {})[field] = value

    async def hdel(self, key, *fields):
        for f in fields:
            self.h.get(key, {}).pop(f, None)

    async def set(self, key, value):
        self.kv[key] = value

    async def get(self, key):
        return self.kv.get(key)


def test_redis_round_trip_skips_malformed_rows_and_prunes_old_ones() -> None:
    r = FakeRedis()
    old = row(at(2026, 9, 1, 10), at(2026, 9, 1, 11))
    new = row(at(2026, 10, 30, 10), at(2026, 10, 30, 11))

    async def go():
        await ts.save_row(r, old)
        await ts.save_row(r, new)
        r.h[AITOWN_SESSION_LEDGER_REDIS_KEY]["junk"] = "{not json"
        rows = await ts.load_rows(r)
        assert {x.session_id for x in rows} == {old.session_id, new.session_id}
        assert await ts.prune(r, rows, at(2026, 10, 30, 12)) == 1
        return await ts.load_rows(r)

    assert [x.session_id for x in asyncio.run(go())] == [new.session_id]
