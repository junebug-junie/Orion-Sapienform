"""Periodic queue fairness/load evaluation against a disposable Postgres DSN.

Produces inspectable JSON for twenty mixed native/widening requests. No model
inference or production bus is called. Eval-only lanes are marked fixtures.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from orion.durable_admission.broker import ResourceBroker
from orion.durable_admission.store import PostgresAdmissionStore
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.resource_admission import ResourceRequirementV1


async def main():
    dsn = os.environ["ORION_ADMISSION_TEST_DSN"]
    schema = "admission_eval_"+uuid4().hex
    async with await AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
    now = [datetime(2026, 9, 12, tzinfo=timezone.utc)]
    async with AsyncConnectionPool(dsn, min_size=1, max_size=4, open=False,
        kwargs={"autocommit": True, "row_factory": dict_row, "options": f"-c search_path={schema},public"}) as pool:
        store = PostgresAdmissionStore(pool, clock=lambda: now[0])
        await store.setup()
        lanes = {lane: {"backend_key": "http://fixture-"+lane, "healthy": True, "configured": True,
                        "compatible_with": ["agent"], "capabilities": {}}
                 for lane in ("agent", "metacog")}
        broker = ResourceBroker(store, lanes, widening_enabled=True, lease_seconds=90)
        for index in range(20):
            preferred = "metacog" if index % 4 == 0 else "agent"
            requirement = ResourceRequirementV1(preferred_lane=preferred, resource="llm.route."+preferred,
                                                alternatives=["metacog"] if preferred == "agent" else [])
            request = DurableRunRequestV1(run_id=f"eval-{index:03}", workflow="curiosity.investigate",
                correlation_id=str(uuid4()), admission=requirement,
                brief={"prompt": "Fixture admission-only study", "session_id": "eval", "timeout_sec": 1800})
            await store.submit(request.model_dump(mode="json"))
            await store.register_demand(request.run_id, requirement.model_dump(mode="json"))
            now[0] += timedelta(seconds=1)
        now[0] += timedelta(seconds=1200)
        served = []
        max_parallel = 0
        for _ in range(20):
            grants = await broker.tick()
            max_parallel = max(max_parallel, len(grants))
            if len({g["backend_key"] for g in grants}) != len(grants):
                raise AssertionError("physical double grant")
            for lease in grants:
                served.append({"run_id": lease["run_id"], "lane": lease["lane"]})
                await store.release(lease, "eval_completion")
                await store.mark_terminal(lease["run_id"], "completed")
            now[0] += timedelta(seconds=1800)
        snapshot = await store.queue_snapshot()
        if len(served) != 20 or len({r["run_id"] for r in served}) != 20 or snapshot["queued"] or snapshot["active"]:
            raise AssertionError("starved or duplicate run")
        print(json.dumps({"verdict": "PASS", "submitted": 20, "served": len(served),
                          "max_parallel": max_parallel, "final_queue": snapshot, "order": served}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
