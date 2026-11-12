"""Prune LangGraph checkpoint history down to what a thread needs to resume.

Why: the Postgres saver keeps every checkpoint, every pending write and every channel version
forever. Its hot read (``aget_tuple``: the newest checkpoint of a thread) joins
``checkpoint_blobs`` on ``thread_id`` and hashes EVERY blob the thread ever wrote, so its cost grows
with the thread's history, not with its state. Measured 2026-11-12: one temporal_self day thread
held 5,321 blob rows to return 10; the boot resume sweep (``runner.py``'s ``alist(None)``) touched
~4M buffers and ran 16 s warm over 53.7k checkpoints of 5.3k threads.

What a thread needs to resume (langgraph-checkpoint-postgres 3.1.2, every row here ``v`` = 4):

* its NEWEST checkpoint per ``(thread_id, checkpoint_ns)`` -- ``aget_tuple`` without a
  ``checkpoint_id`` reads ``ORDER BY checkpoint_id DESC LIMIT 1``; durable-runs never reads history
  (no ``aget_state_history``, no ``checkpoint_id`` in a config; ``alist(None)`` keeps only the
  newest per thread). The parent checkpoint is read only for ``v`` < 4 (pending-sends migration);
* the pending writes of that newest checkpoint (an interrupted or half-done superstep);
* for each channel, the blob at the version the newest checkpoint names.

So this deletes, per ``(thread_id, checkpoint_ns)``: checkpoints older than the newest, writes of
those older checkpoints, and blobs of a channel whose version is LOWER than the one the newest
checkpoint names. Never deleted: anything at or above the newest (a writer racing this job only
adds newer rows -- channel versions only grow), blobs of channels the newest does not name, and any
version string that is not ``<digits>[.<anything>]`` (fail closed). Each statement re-reads the
newest checkpoint itself, so a checkpoint written mid-run only makes the rule stricter.

Dry run by default; ``scripts/prune_checkpoints.py`` is the operator entry point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
import asyncio
import logging
import time
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("orion-durable-runs.checkpoint_prune")
MODES = ("off", "dry_run", "apply")

# The newest checkpoint per (thread, ns), restricted to the threads in %(threads)s (NULL = all).
_LATEST = (
    "SELECT DISTINCT ON (thread_id, checkpoint_ns) thread_id, checkpoint_ns, checkpoint_id, "
    "checkpoint -> 'channel_versions' AS cv "
    "FROM checkpoints WHERE (%(threads)s::text[] IS NULL OR thread_id = ANY(%(threads)s::text[])) "
    "AND (%(prefix)s::text IS NULL OR thread_id LIKE %(prefix)s::text) "
    "ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC"
)

# LangGraph's Postgres versions are f"{int:032}.{random}"; compare the integer part only, and only
# when both sides have one (anything else is kept).
_VERSION_INT = r"^[0-9]+(\.|$)"
_BLOB_STALE = (
    f"l.cv ? b.channel AND b.version ~ '{_VERSION_INT}' AND (l.cv ->> b.channel) ~ '{_VERSION_INT}' "
    "AND split_part(b.version, '.', 1)::numeric < split_part(l.cv ->> b.channel, '.', 1)::numeric"
)

PLAN_SQL = f"""
WITH latest AS ({_LATEST}),
c AS (
  SELECT c.thread_id, count(*) AS total, count(*) FILTER (WHERE c.checkpoint_id < l.checkpoint_id) AS del
  FROM checkpoints c JOIN latest l USING (thread_id, checkpoint_ns) GROUP BY c.thread_id),
w AS (
  SELECT w.thread_id, count(*) AS total, count(*) FILTER (WHERE w.checkpoint_id < l.checkpoint_id) AS del
  FROM checkpoint_writes w JOIN latest l USING (thread_id, checkpoint_ns) GROUP BY w.thread_id),
b AS (
  SELECT b.thread_id, count(*) AS total, count(*) FILTER (WHERE {_BLOB_STALE}) AS del,
         coalesce(sum(length(b.blob)) FILTER (WHERE {_BLOB_STALE}), 0) AS del_bytes
  FROM checkpoint_blobs b JOIN latest l USING (thread_id, checkpoint_ns) GROUP BY b.thread_id)
SELECT c.thread_id, c.total AS ckpt_total, c.del AS ckpt_del,
       coalesce(w.total, 0) AS writes_total, coalesce(w.del, 0) AS writes_del,
       coalesce(b.total, 0) AS blobs_total, coalesce(b.del, 0) AS blobs_del,
       coalesce(b.del_bytes, 0) AS blob_bytes_del
FROM c LEFT JOIN w USING (thread_id) LEFT JOIN b USING (thread_id)
ORDER BY c.thread_id
"""

DELETE_BLOBS_SQL = f"""
WITH latest AS ({_LATEST})
DELETE FROM checkpoint_blobs b USING latest l
WHERE b.thread_id = l.thread_id AND b.checkpoint_ns = l.checkpoint_ns AND {_BLOB_STALE}
"""

DELETE_WRITES_SQL = f"""
WITH latest AS ({_LATEST})
DELETE FROM checkpoint_writes w USING latest l
WHERE w.thread_id = l.thread_id AND w.checkpoint_ns = l.checkpoint_ns AND w.checkpoint_id < l.checkpoint_id
"""

DELETE_CHECKPOINTS_SQL = f"""
WITH latest AS ({_LATEST})
DELETE FROM checkpoints c USING latest l
WHERE c.thread_id = l.thread_id AND c.checkpoint_ns = l.checkpoint_ns AND c.checkpoint_id < l.checkpoint_id
"""

# Read-only: a running pg_dump / pg_dumpall in this database (same rule as
# scripts/apply_post_pull_migrations.py BACKUP_SQL). Deleting during a dump is consistent (the dump
# has its own snapshot) but adds I/O to the window this job exists to relieve.
BACKUP_SQL = (
    "SELECT pid, application_name FROM pg_stat_activity "
    "WHERE application_name IN ('pg_dump', 'pg_dumpall') AND pid <> pg_backend_pid() "
    "AND datname = current_database() AND xact_start IS NOT NULL ORDER BY pid"
)

_ID_TAIL = re.compile(r"([:\-][0-9a-f]{8,}.*|:[0-9]{4}-[0-9]{2}-[0-9]{2}.*)$")
_BARE_ID = re.compile(r"^[0-9a-f]{8,}$")
COUNT_KEYS = ("ckpt_total", "ckpt_del", "writes_total", "writes_del", "blobs_total", "blobs_del", "blob_bytes_del")


def thread_family(thread_id: str) -> str:
    """Group key for the report: the thread id without its run id / date tail."""
    if _BARE_ID.match(thread_id or ""):
        return "<bare run id>"
    fam = _ID_TAIL.sub("", thread_id or "")
    return fam or "<bare run id>"


@dataclass
class Plan:
    rows: list[dict] = field(default_factory=list)

    @property
    def threads_to_prune(self) -> list[str]:
        return [r["thread_id"] for r in self.rows
                if r["ckpt_del"] or r["writes_del"] or r["blobs_del"]]

    def totals(self) -> dict:
        out = {k: 0 for k in COUNT_KEYS}
        for r in self.rows:
            for k in COUNT_KEYS:
                out[k] += int(r.get(k) or 0)
        out["threads"] = len(self.rows)
        out["threads_to_prune"] = len(self.threads_to_prune)
        return out

    def by_family(self, top: int = 15) -> list[dict]:
        fam: dict[str, dict] = {}
        for r in self.rows:
            f = fam.setdefault(thread_family(r["thread_id"]), {"family": thread_family(r["thread_id"]),
                                                              "threads": 0, **{k: 0 for k in COUNT_KEYS}})
            f["threads"] += 1
            for k in COUNT_KEYS:
                f[k] += int(r.get(k) or 0)
        return sorted(fam.values(), key=lambda f: (-f["ckpt_del"] - f["blobs_del"], f["family"]))[:top]

    def largest_threads(self, top: int = 5) -> list[dict]:
        return sorted(self.rows, key=lambda r: (-int(r["blobs_total"]), r["thread_id"]))[:top]


def _params(threads: Optional[list[str]] = None, prefix: Optional[str] = None) -> dict:
    return {"threads": threads, "prefix": prefix}


def _dict(cur, row) -> dict:
    if isinstance(row, dict):
        return dict(row)
    return {d.name if hasattr(d, "name") else d[0]: v for d, v in zip(cur.description, row)}


def build_plan(conn: Any, prefix: Optional[str] = None) -> Plan:
    """Read-only. One row per thread: what exists and what pruning would delete."""
    with conn.cursor() as cur:
        cur.execute(PLAN_SQL, _params(prefix=prefix))
        return Plan(rows=[_dict(cur, r) for r in cur.fetchall()])


def running_backups(conn: Any) -> list[tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(BACKUP_SQL)
        return [(int(_dict(cur, r)["pid"]), str(_dict(cur, r)["application_name"])) for r in cur.fetchall()]


def _batches(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), max(1, size)):
        yield items[i:i + max(1, size)]


def apply_plan(conn: Any, plan: Plan, *, batch_threads: int = 200, lock_timeout: str = "5s",
               statement_timeout: str = "120s", prefix: Optional[str] = None) -> dict:
    """DELETE what ``plan`` found, one transaction per batch of threads. Each statement recomputes
    the newest checkpoint itself (see module docstring). ``conn`` must not be in autocommit."""
    deleted = {"checkpoints": 0, "writes": 0, "blobs": 0, "batches": 0}
    for batch in _batches(plan.threads_to_prune, batch_threads):
        params = _params(threads=batch, prefix=prefix)
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL lock_timeout = '{_interval(lock_timeout)}'")
                cur.execute(f"SET LOCAL statement_timeout = '{_interval(statement_timeout)}'")
                cur.execute(DELETE_BLOBS_SQL, params)
                deleted["blobs"] += max(cur.rowcount, 0)
                cur.execute(DELETE_WRITES_SQL, params)
                deleted["writes"] += max(cur.rowcount, 0)
                cur.execute(DELETE_CHECKPOINTS_SQL, params)
                deleted["checkpoints"] += max(cur.rowcount, 0)
        deleted["batches"] += 1
    return deleted


def _interval(v: str) -> str:
    if not re.fullmatch(r"[0-9]+(ms|s|min)?", v or ""):
        raise ValueError(f"bad timeout {v!r}")
    return v


def run_once(dsn: str, *, apply: bool = False, prefix: Optional[str] = None, batch_threads: int = 200,
             allow_during_backup: bool = False, connect: Optional[Callable[[str], Any]] = None) -> dict:
    """One pass on its own connection (never the saver's: the saver serialises every call behind
    one lock, see README). Dry run unless ``apply``. Returns the report; ``status`` is ``ok`` or
    ``skipped_backup``."""
    if connect is None:
        import psycopg
        from psycopg.rows import dict_row

        def connect(d):  # noqa: E306
            return psycopg.connect(d, row_factory=dict_row, application_name="orion-checkpoint-prune")
    with connect(dsn) as conn:
        backups = running_backups(conn)
        conn.commit()
        mode = "apply" if apply else "dry_run"
        if backups and not allow_during_backup:
            return {"status": "skipped_backup", "mode": mode, "backups": backups}
        plan = build_plan(conn, prefix=prefix)
        conn.commit()
        report = {"status": "ok", "mode": mode, "thread_prefix": prefix, "totals": plan.totals(),
                  "by_family": plan.by_family(), "largest_threads": plan.largest_threads()}
        if apply:
            t0 = time.monotonic()
            report["deleted"] = apply_plan(conn, plan, batch_threads=batch_threads, prefix=prefix)
            report["apply_sec"] = round(time.monotonic() - t0, 3)
    return report


class PrunePoller:
    """Every ``interval_sec`` (first pass after ``first_delay_sec``): one ``run_once`` in a worker
    thread. ``mode`` ``dry_run`` (the shipped default) only logs and reports; ``apply`` deletes."""

    def __init__(self, *, dsn: str, mode: str, interval_sec: float, first_delay_sec: float = 300.0,
                 runner: Callable[..., dict] = run_once) -> None:
        if mode not in MODES:
            raise ValueError(f"checkpoint prune mode {mode!r} not in {MODES}")
        self.dsn, self.mode, self.interval_sec, self.first_delay_sec = dsn, mode, interval_sec, first_delay_sec
        self._runner = runner
        self.last: Optional[dict] = None
        self.last_error: Optional[str] = None
        self.passes = 0

    async def tick(self) -> Optional[dict]:
        if self.mode == "off":
            return None
        try:
            report = await asyncio.to_thread(self._runner, self.dsn, apply=self.mode == "apply")
        except Exception as exc:  # noqa: BLE001 - one bad pass must not stop the loop
            self.last_error = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("checkpoint_prune_failed mode=%s error=%s", self.mode, self.last_error)
            return None
        self.passes += 1
        self.last_error = None
        t = report.get("totals") or {}
        self.last = {"at": time.time(), "status": report.get("status"), "mode": report.get("mode"),
                     "totals": t, "deleted": report.get("deleted")}
        logger.info("checkpoint_prune mode=%s status=%s threads=%s to_prune=%s ckpt_del=%s/%s writes_del=%s "
                    "blobs_del=%s/%s deleted=%s", report.get("mode"), report.get("status"), t.get("threads"),
                    t.get("threads_to_prune"), t.get("ckpt_del"), t.get("ckpt_total"), t.get("writes_del"),
                    t.get("blobs_del"), t.get("blobs_total"), report.get("deleted"))
        return report

    async def run_forever(self, stop: asyncio.Event) -> None:
        delay = self.first_delay_sec
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
                return
            except asyncio.TimeoutError:
                pass
            await self.tick()
            delay = self.interval_sec

    def health(self) -> dict:
        return {"mode": self.mode, "interval_sec": self.interval_sec, "passes": self.passes,
                "last": self.last, "last_error": self.last_error}
