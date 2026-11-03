"""Self-monitor E2 in Hub (scripts/self_monitor_questions.py): one self-question per failing episode.

Fakes for Redis and the asyncpg pool; the pure decision is tested in
orion/self_monitor/tests/test_self_monitor_gate.py.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from orion.curiosity.self_question_pool import UPSERT_SEED_SQL
from orion.self_monitor.view import SELF_MONITOR_REDIS_KEY
from scripts.self_monitor_questions import SEEN_KEY, SelfMonitorQuestionMinter

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
WALKWAY = "act.actions.journal.make.walkway_grade"


def _state(status: str, *, computed_at: datetime = NOW) -> str:
    return json.dumps({"kind": "self_monitor.state.v1", "computed_at": computed_at.isoformat(), "window_days": 7.0,
                       "faculties": [{
                           "faculty": WALKWAY, "gradeability": "ungradeable:too_few",
                           "window_start": (computed_at - timedelta(days=7)).isoformat(),
                           "window_end": computed_at.isoformat(), "n_bets": 28, "n_graded": 6,
                           "n_failed_to_execute": 22, "n_expired": 0, "weight_graded": 2.9,
                           "candidate_status": status, "status": status,
                           "status_since": (NOW - timedelta(days=1)).isoformat()}]})


class _Redis:
    def __init__(self, data):
        self.data = dict(data)

    async def get(self, key):
        v = self.data.get(key)
        return v.encode() if isinstance(v, str) else v

    async def set(self, key, value):
        self.data[key] = value


class _Conn:
    def __init__(self, sink, fail):
        self.sink, self.fail = sink, fail

    async def execute(self, sql, *args):
        if self.fail:
            raise RuntimeError("db down")
        self.sink.append((sql, args))
        return "INSERT 0 1"


class _Pool:
    def __init__(self, fail=False):
        self.rows, self.fail = [], fail

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return _Conn(pool.rows, pool.fail)

            async def __aexit__(self, *a):
                return False

        return _Ctx()


def _minter(redis, pool, now=NOW):
    return SelfMonitorQuestionMinter(enabled=True, redis_provider=lambda: redis, pool_provider=lambda: pool,
                                     clock=lambda: now)


def test_broken_faculty_mints_once_as_orion():
    redis, pool = _Redis({SELF_MONITOR_REDIS_KEY: _state("broken")}), _Pool()
    out = asyncio.run(_minter(redis, pool).tick())
    assert out["minted"] == 1
    sql, args = pool.rows[0]
    assert sql == UPSERT_SEED_SQL and "DO NOTHING" in sql
    qid, text, family, pinned, minted_by = args
    assert minted_by == "orion" and family == "lived" and pinned is False
    assert text == f"My faculty {WALKWAY} didn't run 22 of its last 28 tries in the last 7 days. Why?"
    assert json.loads(redis.data[SEEN_KEY])[WALKWAY].startswith("broken@")
    # A later tick (or a restarted Hub reading the same Redis memory) asks nothing.
    again = asyncio.run(_minter(redis, pool, NOW + timedelta(hours=1)).tick())
    assert again["minted"] == 0 and len(pool.rows) == 1


def test_reliable_and_unknown_mint_nothing():
    pool = _Pool()
    assert asyncio.run(_minter(_Redis({SELF_MONITOR_REDIS_KEY: _state("reliable")}), pool).tick())["minted"] == 0
    stale = _Redis({SELF_MONITOR_REDIS_KEY: _state("broken", computed_at=NOW - timedelta(hours=4)),
                    SEEN_KEY: json.dumps({WALKWAY: "reliable@x"})})
    out = asyncio.run(_minter(stale, pool).tick())
    assert out["view"] == "stale" and pool.rows == []
    assert json.loads(stale.data[SEEN_KEY]) == {WALKWAY: "reliable@x"}
    assert asyncio.run(_minter(_Redis({}), pool).tick())["view"] == "absent"


def test_failed_insert_is_retried_next_tick():
    redis = _Redis({SELF_MONITOR_REDIS_KEY: _state("broken")})
    out = asyncio.run(_minter(redis, _Pool(fail=True)).tick())
    assert out["minted"] == 0 and out["failed"] == [WALKWAY]
    assert WALKWAY not in json.loads(redis.data.get(SEEN_KEY) or "{}")
    pool = _Pool()
    assert asyncio.run(_minter(redis, pool).tick())["minted"] == 1


def test_bus_not_connected_is_a_skip_not_a_crash():
    def not_connected():
        raise RuntimeError("OrionBusAsync not connected")

    m = SelfMonitorQuestionMinter(enabled=True, redis_provider=not_connected, pool_provider=lambda: _Pool(),
                                  clock=lambda: NOW)
    assert asyncio.run(m.tick())["skipped"] == "no_redis"
