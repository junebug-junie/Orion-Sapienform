"""A fake asyncpg connection answering power_attribution's three queries from fixture rows."""
from __future__ import annotations

from contextlib import asynccontextmanager

from orion.introspect.power_attribution import BIOMETRICS_SQL, ENERGY_SQL, POOL_BUSY_SQL


class FakePowerConn:
    def __init__(self, rows: dict[str, list[dict]], *, fail: bool = False):
        self.rows = rows
        self.fail = fail
        self.queries: list[tuple[str, tuple]] = []
        self.readonly_flags: list[bool] = []

    @asynccontextmanager
    async def transaction(self, readonly=False):
        self.readonly_flags.append(readonly)
        yield

    async def fetch(self, sql, *args):
        if self.fail:
            raise RuntimeError("password=hunter2 host=db")
        name = {ENERGY_SQL: "energy_hourly", BIOMETRICS_SQL: "biometrics_hourly", POOL_BUSY_SQL: "pool_busy_hours"}[sql]
        self.queries.append((name, args))
        start, end = args[0], args[1]
        key = "hour_start" if name == "energy_hourly" else "hour"
        return [r for r in self.rows[name] if start <= r[key] < end]


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self.conn
