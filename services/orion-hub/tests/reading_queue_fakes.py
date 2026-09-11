"""Add general ingress SQL to the legacy worker fakes; real SQL is tested separately."""
import json
from contextlib import asynccontextmanager


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
                       stage2_result_json=None, landing_at=None)
            if args[11]:
                row.update(status="skipped", stage2_status="skipped")
        if "stage2_result_json = COALESCE" in sql and args[0] in self.rows:
            self.rows[args[0]]["stage2_result_json"] = json.loads(args[2]) if args[2] else None
        return result

    def _returning(self, row):
        return {**super()._returning(row), "request_json": row.get("request_json")}

    async def fetchrow(self, sql, *args):
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
