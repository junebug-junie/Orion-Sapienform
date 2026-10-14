"""Postgres checkpoints and readout; writes only curiosity_compaction.

A session advisory lock per cadence serializes replicas/manual sleeps. Checkpoint
transactions are short; no transaction stays open while waiting for the model.
A crash releases the connection lock and the next sleep resumes saved progress.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import cached_property

from sqlalchemy import create_engine, text

from orion.curiosity.compaction import batches, due_window, source_material, window_id

class Store:
    """Explicit shared persistence API used by sleep admission and the durable graph."""
    def __init__(self, postgres_uri, *, driver=None):
        self.postgres_uri, self.driver = postgres_uri, driver

    @cached_property
    def engine(self):
        from sqlalchemy.engine import make_url
        uri = self.postgres_uri() if callable(self.postgres_uri) else self.postgres_uri
        url = make_url(uri)
        if self.driver:
            url = url.set(drivername="postgresql+" + self.driver)
        return create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=0, pool_timeout=3,
                             connect_args={"connect_timeout": 3, "options": "-c statement_timeout=10000"})

    def load(self, compaction_id, run_id):
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM curiosity_compaction WHERE compaction_id=:id AND run_id=:run"),
                               {"id": compaction_id, "run": run_id}).mappings().one()
            return dict(row)

    def persist(self, row):
        with self.engine.connect() as conn:
            Session(conn, row["cadence"], None).save(row)
        return row

    def recent(self, limit=20):
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(text("""
                SELECT compaction_id, cadence, window_start, window_end, cycle_id, c.run_id,
                       c.status, c.updated_at, d.terminal AS durable_terminal,
                       state->>'body' AS body, state->'topics' AS topics, state->'sources' AS sources,
                       state->>'phase' AS phase
                FROM curiosity_compaction c LEFT JOIN durable_admission_runs d ON d.run_id=c.run_id
                ORDER BY window_start DESC, cadence LIMIT :limit
            """), {"limit": limit}).mappings()]

    def acquire(self, cadence):
        conn = self.engine.connect()
        key = 873201 if cadence == "weekly" else 873202
        try:
            locked = conn.execute(text("SELECT pg_try_advisory_lock(:key)"), {"key": key}).scalar()
            conn.commit()
            if not locked:
                conn.close()
                return None
            return Session(conn, cadence, key)
        except BaseException:
            conn.invalidate()
            conn.close()
            raise


class Session:
    def __init__(self, conn, cadence, key):
        self.conn, self.cadence, self.key = conn, cadence, key

    def read(self, sql, params=None):
        try:
            rows = [dict(r) for r in self.conn.execute(text(sql), params or {}).mappings()]
            self.conn.commit()
            return rows
        except BaseException:
            self.conn.rollback()
            raise

    def prepare(self, now, cycle_id):
        previous = self.read("SELECT * FROM curiosity_compaction WHERE cadence=:cadence "
                             "ORDER BY window_start DESC LIMIT 1", {"cadence": self.cadence})
        latest = previous[0] if previous else None
        window = due_window(now, self.cadence, latest)
        if window is None:
            return None
        if latest and latest["status"] == "pending":
            terminal = self.read("SELECT terminal FROM durable_admission_runs WHERE run_id=:id",
                                 {"id": latest["run_id"]})
            outcome = terminal[0]["terminal"] if terminal else None
            if outcome in {"cancelled", "abandoned"}:
                latest["status"] = "cancelled"
                self.save(latest)
                return None
            if outcome == "failed":
                # Retry only an authoritative terminal failure; keep all saved topic work.
                generation = latest["generation"] + 1
                run_id = f"{latest['compaction_id']}:r{generation}"
                self.conn.execute(text("UPDATE curiosity_compaction SET run_id=:new, generation=:generation "
                                       "WHERE compaction_id=:id AND run_id=:old"),
                                  {"new": run_id, "generation": generation, "id": latest["compaction_id"],
                                   "old": latest["run_id"]})
                self.conn.commit()
                latest.update(run_id=run_id, generation=generation)
            return latest
        start, end = window
        params = {"start": start, "end": end}
        journals = self.read("SELECT entry_id, source_ref, body, created_at FROM journal_entries "
                             "WHERE source_ref LIKE 'curiosity:%' AND created_at >= :start "
                             "AND created_at < :end ORDER BY created_at, entry_id", params)
        weeks = self.read("SELECT * FROM curiosity_compaction WHERE cadence='weekly' "
                          "AND status='completed' AND window_start >= :start AND window_end <= :end "
                          "ORDER BY window_start", params) if self.cadence == "monthly" else []
        material, sources = source_material(journals, weeks)
        row = dict(compaction_id=window_id(self.cadence, start), cadence=self.cadence,
                   window_start=start, window_end=end, cycle_id=cycle_id, status="pending",
                   run_id=window_id(self.cadence, start) + ":r0", generation=0,
                   state={"phase": "topics", "sources": sources, "batches": batches(material), "partials": []})
        self.save(row)
        return row

    def save(self, row):
        row["updated_at"] = datetime.now(timezone.utc)
        try:
            result = self.conn.execute(text("""
                INSERT INTO curiosity_compaction
                  (compaction_id, cadence, window_start, window_end, cycle_id, status, state, updated_at, run_id, generation)
                VALUES (:compaction_id, :cadence, :window_start, :window_end, :cycle_id, :status,
                        CAST(:state AS jsonb), :updated_at, :run_id, :generation)
                ON CONFLICT (compaction_id) DO UPDATE SET
                  status=EXCLUDED.status, state=EXCLUDED.state, updated_at=EXCLUDED.updated_at
                WHERE curiosity_compaction.run_id=EXCLUDED.run_id
            """), {**row, "state": json.dumps(row["state"])})
            if result.rowcount != 1:
                raise RuntimeError("stale compaction run fenced")
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def unjournaled(self):
        return self.read("SELECT c.* FROM curiosity_compaction c WHERE c.cadence=:cadence "
                         "AND c.status='completed' AND NOT EXISTS "
                         "(SELECT 1 FROM journal_entries j WHERE j.source_ref=c.compaction_id) "
                         "ORDER BY c.window_start LIMIT 2", {"cadence": self.cadence})

    def close(self):
        try:
            self.conn.rollback()
            self.conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": self.key})
            self.conn.commit()
        except BaseException:
            self.conn.invalidate()  # never return a locked session to the pool
            raise
        finally:
            self.conn.close()
