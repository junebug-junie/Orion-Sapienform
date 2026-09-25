"""Isolated Postgres contention eval for the permits durable-runs still serves (``/capacity``).

Stage 4.5: durable runs hold GPU pool holds, not durable leases, so the broker half of this eval
went with the broker. What remains until stage 5 is the world-model / visual-chain permit store:
two gateways contending for one backend must never both win, a permit renews and expires, and the
FROZEN legacy tables must not starve it -- a frozen pending demand no longer reserves the backend
(no broker exists to grant it), while an active legacy lease still fences it (why the cutover
runbook waits for zero active leases first). No production bus or model invocation.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from orion.durable_admission.capacity import PostgresCapacityStore  # noqa: E402
from orion.durable_admission.store import PostgresAdmissionStore  # noqa: E402
from orion.schemas.durable_run import DurableRunRequestV1  # noqa: E402
from orion.schemas.resource_admission import CapacityAcquireV1, CapacityTokenV1  # noqa: E402

BACKEND = "http://fixture-world"


def call(**changes):
    return CapacityAcquireV1(request_id=uuid4().hex, correlation_id=str(uuid4()), lane="world",
                             backend_key=BACKEND, max_inflight=1, budget_sec=120, **changes)


def token(result):
    return CapacityTokenV1(**{key: result["permit"][key] for key in ("request_id", "permit_id")})


async def main() -> int:
    dsn = os.environ["ORION_ADMISSION_TEST_DSN"]
    schema = "capacity_eval_" + uuid4().hex
    async with await AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
    async with AsyncConnectionPool(dsn, min_size=1, max_size=8, open=False,
        kwargs={"autocommit": True, "row_factory": dict_row, "options": f"-c search_path={schema},public"}) as pool:
        store = PostgresAdmissionStore(pool)
        await store.setup()
        # The 4.5 wiring (app/main.py): no drain reservation for frozen demands.
        gateways = [PostgresCapacityStore(store, reserve_waiting=False) for _ in range(2)]
        # A frozen pre-cutover pending demand whose last broker decision reserved this backend.
        run = DurableRunRequestV1(run_id="capacity-eval-frozen", correlation_id=str(uuid4()),
            workflow="curiosity.investigate", admission={},
            brief={"prompt": "Admission-only fixture; no cognition invoked", "session_id": "capacity-eval", "timeout_sec": 120})
        await store.submit(run.model_dump(mode="json"))
        async with pool.connection() as conn:
            await conn.execute("INSERT INTO durable_resource_demands(demand_id,run_id,requirement,status,decision) "
                               "VALUES ('frozen:turn',%s,%s,'pending',%s)",
                               (run.run_id, Jsonb(run.admission.model_dump(mode="json")),
                                Jsonb({"eligible_backend_keys": [BACKEND]})))
        # Under the pre-4.5 wiring that frozen demand would starve every permit on the backend.
        old_wiring = (await PostgresCapacityStore(store).acquire(call()))["reason"]
        winners = {"first": 0, "second": 0}
        for index in range(40):
            results = await asyncio.gather(gateways[0].acquire(call()), gateways[1].acquire(call()))
            won = [i for i, r in enumerate(results) if r["acquired"]]
            assert len(won) == 1, f"round {index}: {results}"
            winners["first" if won[0] == 0 else "second"] += 1
            permit = results[won[0]]
            assert (await gateways[won[0]].renew(token(permit)))["valid"]
            assert (await gateways[1 - won[0]].acquire(call()))["reason"] == "capacity_full"
            assert (await gateways[won[0]].release(token(permit)))["released"]
        # An active legacy lease on the backend still fences it until it ends.
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO durable_resource_leases(lease_id,demand_id,run_id,resource_key,lane,backend_key,"
                "granted_at,expires_at,heartbeat_at,status) VALUES ('legacy','frozen:turn',%s,'llm.route.agent',"
                "'agent',%s,now(),now()+interval '1 hour',now(),'active')", (run.run_id, BACKEND))
        fenced = (await gateways[0].acquire(call()))["reason"]
        async with pool.connection() as conn:
            await conn.execute("UPDATE durable_resource_leases SET status='released'")
        after = await gateways[0].acquire(call())
        await gateways[0].release(token(after))
        snapshot = await gateways[0].snapshot()
        checks = {"one_winner_every_round": sum(winners.values()) == 40,
                  "frozen_demand_would_have_reserved_pre_4_5": old_wiring == "durable_waiting",
                  "legacy_lease_fences_backend": fenced == "durable_lease_active",
                  "backend_free_after_legacy_lease_ends": bool(after["acquired"]),
                  "no_permit_left": not snapshot["active_permits"]}
        failed = [k for k, ok in checks.items() if not ok]
        print(json.dumps({"verdict": "FAIL" if failed else "PASS", "contended_rounds": 40,
                          "competing_gateways": 2, "winners": winners, "checks": checks,
                          "final_capacity": snapshot}, indent=2))
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
