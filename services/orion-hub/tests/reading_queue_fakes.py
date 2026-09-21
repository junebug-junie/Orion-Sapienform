"""Add general ingress SQL to the legacy worker fakes; real SQL is tested separately."""
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from orion.world_pulse_read.retry import is_transient_failure


class ReadingQueueFakeMixin:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def is_in_transaction(self):
        return False

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def execute(self, sql, *args):
        result = await super().execute(sql, *args)
        if "INSERT INTO world_pulse_read_seed" in sql and result == "INSERT 0 1":
            row = self.rows[args[0]]
            row.update(request_id=args[8], request_json=json.loads(args[9]),
                       root_request_id=args[10], duplicate_of=args[11],
                       stage2_result_json=None, landing_at=None,
                       attempts=0, stage2_attempts=0)
            if args[11]:
                row.update(status="skipped", stage2_status="skipped")
        if "stage2_result_json = COALESCE" in sql and args[0] in self.rows:
            self.rows[args[0]]["stage2_result_json"] = json.loads(args[2]) if args[2] else None
        return result

    def _returning(self, row):
        return {**super()._returning(row), "request_json": row.get("request_json")}

    # Retry ordering (a fresh seed claimed before one that already burned an
    # attempt, at the same priority) lives in the real CLAIM_SQL / CLAIM_STAGE2_SQL
    # (queue.py) and is exercised by the real-Postgres suite
    # (test_reading_postgres.py). Not duplicated here: each caller's own
    # `_claim_pending`/`_claim_stage2` already tracks caller-specific state
    # (e.g. `claimed_ids`) that an override here would shadow via MRO.

    def _mark_failed(self, row, *, stage2, error, trace_id, max_attempts):
        """Interpret MARK_FAILED_SQL / MARK_STAGE2_FAILED_SQL exactly as Postgres would."""
        a_key, s_key, e_key = ("stage2_attempts", "stage2_status", "stage2_error") if stage2 else ("attempts", "status", "last_error")
        c_key, d_key = ("stage2_claimed_at", "stage2_completed_at") if stage2 else ("claimed_at", "completed_at")
        attempts = int(row.get(a_key, 0)) + 1
        row[a_key] = attempts
        row[e_key] = error
        if stage2 and trace_id:
            row["stage2_trace_id"] = trace_id
        if is_transient_failure(error) and attempts < int(max_attempts):
            row[s_key] = "pending"
            row[c_key] = None
            row[d_key] = None
        else:
            row[s_key] = "failed"
            row[d_key] = datetime.now(timezone.utc)
        return {s_key: row[s_key], a_key: attempts}

    async def fetchrow(self, sql, *args):
        if "AS position," in sql and "AS depth" in sql:
            # Interprets STAGE1_QUEUE_POSITION_SQL (queue.py): same ordering
            # as _claim_pending -- (priority, attempts, created_at, seed_id).
            self.executed.append((sql, args))
            priority, attempts, created_at, seed_id = args
            key = (priority, attempts, created_at, seed_id)
            pending = [r for r in self.rows.values() if r["status"] == "pending"]
            position = sum(
                1 for r in pending
                if (r["priority"], r.get("attempts", 0), r.get("created_at", 0), r["seed_id"]) < key
            ) + 1
            return {"position": position, "depth": len(pending)}
        if "stage2_attempts = stage2_attempts + 1" in sql:
            self.executed.append((sql, args))
            row = self.rows.get(args[0])
            if row is None:
                return None
            return self._mark_failed(row, stage2=True, error=args[1], trace_id=args[2], max_attempts=args[4])
        if "attempts = attempts + 1" in sql:
            self.executed.append((sql, args))
            row = self.rows.get(args[0])
            if row is None:
                return None
            return self._mark_failed(row, stage2=False, error=args[1], trace_id=None, max_attempts=args[3])
        if "AS stage1_pending_retry" in sql:
            mx = int(args[0])
            rows = list(self.rows.values())
            return {
                "stage1_pending_retry": sum(1 for r in rows if r["status"] == "pending" and r.get("attempts", 0) > 0),
                "stage2_pending_retry": sum(1 for r in rows if r["status"] == "done" and r.get("stage2_status") == "pending" and r.get("stage2_attempts", 0) > 0),
                "stage1_exhausted": sum(1 for r in rows if r["status"] == "failed" and r.get("attempts", 0) >= mx),
                "stage2_exhausted": sum(1 for r in rows if r.get("stage2_status") == "failed" and r.get("stage2_attempts", 0) >= mx),
            }
        if "WHERE request_id = $1" in sql:
            return next((r for r in self.rows.values() if r.get("request_id") == args[0]), None)
        if sql.startswith("SELECT") and "WHERE seed_id = $1" in sql:
            return self.rows.get(args[0])
        if "WHERE url = $1 AND duplicate_of IS NULL" in sql:
            return next((r for r in self.rows.values() if r["url"] == args[0]
                         and not r.get("duplicate_of") and (r["status"] in ("pending", "claimed")
                         or (r["status"] == "done" and r.get("stage2_status", "pending") in ("pending", "claimed")))), None)
        return await super().fetchrow(sql, *args)

    async def fetchval(self, sql, *args):
        return sum(1 for r in self.rows.values() if r.get("root_request_id") == args[0]
                   and r.get("request_id") != args[0] and not r.get("duplicate_of"))

    async def fetch(self, sql, *args):
        if "SET landing_at = now()" in sql:
            return []
        return await super().fetch(sql, *args)
