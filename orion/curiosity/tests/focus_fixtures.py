"""In-memory ``curiosity_focus_request`` for the Hub tests.

Implements exactly the statements orion/curiosity/focus.py sends. ``test_focus_lifecycle.py``
runs the same lifecycle scenarios against this fake AND a real Postgres (when
ORION_MEMORY_EPISODE_TEST_DATABASE_URL is set; CI: orion-memory-episode-tests.yml), so the fake
cannot drift from the SQL silently.
"""

from __future__ import annotations

import copy
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from orion.curiosity import focus as cf


class UniqueViolation(Exception):
    sqlstate = "23505"


class FocusTable:
    def __init__(self, now: datetime | None = None) -> None:
        self.rows: list[dict] = []
        self.now = now or datetime(2026, 10, 16, 15, 0, tzinfo=timezone.utc)
        self.statements: list[str] = []
        self.fail_with: Exception | None = None

    # --- connection surface ---------------------------------------------------------------

    @asynccontextmanager
    async def transaction(self, **_: Any):
        saved = copy.deepcopy(self.rows)
        try:
            yield
        except BaseException:
            self.rows = saved
            raise

    def _eff(self, r: dict) -> dict:
        out = dict(r)
        out["effective_status"] = (
            cf.EXPIRED if r["status"] == cf.PENDING and r["expires_at"] <= self.now else r["status"]
        )
        return out

    def _check(self) -> None:
        if sum(1 for r in self.rows if r["status"] == cf.PENDING) > 1:
            raise UniqueViolation("curiosity_focus_request_one_pending")

    def _record(self, sql: str) -> None:
        self.statements.append(sql)
        if self.fail_with is not None:
            raise self.fail_with

    async def execute(self, sql: str, *args: Any):
        self._record(sql)
        if sql == cf.LOCK_SQL:
            return "SELECT 1"
        if sql == cf.EXPIRE_SQL:
            for r in self.rows:
                if r["status"] == cf.PENDING and r["expires_at"] <= self.now:
                    r["status"], r["status_at"] = cf.EXPIRED, self.now
            return "UPDATE"
        if sql == cf.STALE_SQL:
            cutoff = self.now - timedelta(seconds=args[0])
            for r in self.rows:
                if r["status"] == cf.TAKEN and r["outcome_stamped_at"] is None and r["offered_at"] < cutoff:
                    r.update(status=cf.FAILED, status_at=self.now, outcome_stamped_at=self.now)
            return "UPDATE"
        if sql == cf.REPLACE_SQL:
            old, new = args
            for r in self.rows:
                if r["focus_id"] == old and r["status"] == cf.PENDING:
                    r.update(status=cf.REPLACED, status_at=self.now, replaced_by=new)
            return "UPDATE"
        raise AssertionError(sql[:80])

    async def fetchrow(self, sql: str, *args: Any):
        self._record(sql)
        if sql == cf.BY_ID_SQL:
            found = [r for r in self.rows if r["focus_id"] == args[0]]
            return self._eff(found[0]) if found else None
        if sql == cf.PENDING_FOR_UPDATE_SQL:
            found = [r for r in self.rows if r["status"] == cf.PENDING]
            return self._eff(found[0]) if found else None
        if sql == cf.PENDING_SQL:
            found = [r for r in self.rows if r["status"] == cf.PENDING and r["expires_at"] > self.now]
            return self._eff(found[0]) if found else None
        if sql == cf.INSERT_SQL:
            focus_id, question, why, origin, chat, ttl_sec, refs = args
            if any(r["focus_id"] == focus_id for r in self.rows):
                raise UniqueViolation("pkey")
            # Each queue is its own transaction in Postgres, with its own now().
            self.now += timedelta(seconds=1)
            row = {
                "focus_id": focus_id, "question": question, "why": why, "origin": origin,
                "chat_correlation_id": chat, "created_at": self.now,
                "expires_at": self.now + timedelta(seconds=ttl_sec), "status": cf.PENDING,
                "status_at": self.now, "replaced_by": None, "offered_at": None, "offered_run_id": None,
                "outcome_stamped_at": None, "turn_ok": None, "disposition": None, "disposition_raw": None,
                "refs": refs,  # asyncpg hands jsonb back as text, as here
            }
            self.rows.append(row)
            self._check()
            return {**row, "effective_status": cf.PENDING}
        raise AssertionError(sql[:80])

    async def fetch(self, sql: str, *args: Any):
        self._record(sql)
        if sql == cf.TAKE_SQL:
            out = []
            for r in self.rows:
                if r["status"] == cf.PENDING and r["expires_at"] > self.now:
                    r.update(status=cf.TAKEN, status_at=self.now, offered_at=self.now, offered_run_id=args[0])
                    out.append({**r, "effective_status": cf.TAKEN})
            return out
        if sql == cf.RELEASE_SQL:
            pending = next((r["focus_id"] for r in self.rows if r["status"] == cf.PENDING), None)
            out = []
            for r in self.rows:
                if r["offered_run_id"] == args[0] and r["status"] == cf.TAKEN and r["outcome_stamped_at"] is None:
                    r.update(status=cf.PENDING if pending is None else cf.REPLACED, replaced_by=pending,
                             status_at=self.now, offered_at=None, offered_run_id=None)
                    out.append({"focus_id": r["focus_id"], "status": r["status"]})
            self._check()
            return out
        if sql == cf.STAMP_SQL:
            run_id, ok, disposition, raw, graph_read = args
            out = []
            for r in self.rows:
                if (r["offered_run_id"] == run_id and r["status"] in (cf.TAKEN, cf.RAN, cf.FAILED)
                        and not (r["status"] == cf.RAN and not ok)):
                    r.update(status=cf.RAN if ok else cf.FAILED, status_at=self.now, turn_ok=ok,
                             outcome_stamped_at=self.now)
                    if graph_read:
                        r.update(disposition=disposition, disposition_raw=raw)
                    out.append({"focus_id": r["focus_id"]})
            return out
        if sql == cf.FAIL_SQL:
            out = []
            for r in self.rows:
                if r["offered_run_id"] == args[0] and r["status"] == cf.TAKEN:
                    r.update(status=cf.FAILED, status_at=self.now, turn_ok=False, outcome_stamped_at=self.now)
                    out.append({"focus_id": r["focus_id"]})
            return out
        if sql == cf.CLEAR_SQL:
            out = []
            for r in self.rows:
                if r["focus_id"] == args[0] and r["status"] == cf.PENDING and r["expires_at"] > self.now:
                    r.update(status=cf.CLEARED, status_at=self.now)
                    out.append({"focus_id": r["focus_id"]})
            return out
        if sql == cf.RECENT_SQL:
            since, limit = args
            rows = [r for r in self.rows if since is None or r["created_at"] >= since]
            rows.sort(key=lambda r: (r["created_at"], str(r["focus_id"])), reverse=True)
            return [{**self._eff(r), "total": len(rows)} for r in rows[:limit]]
        if sql == cf.FOR_RUNS_SQL:
            return [self._eff(r) for r in self.rows if r["offered_run_id"] in args[0]]
        raise AssertionError(sql[:80])


class FocusPool:
    def __init__(self, table: FocusTable) -> None:
        self.table = table

    def acquire(self):
        @asynccontextmanager
        async def _cm():
            yield self.table

        return _cm()
