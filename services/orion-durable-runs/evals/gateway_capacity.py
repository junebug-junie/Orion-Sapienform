"""Isolated Postgres contention eval; no production bus or model invocation."""
from __future__ import annotations

import asyncio
import json
import os
from uuid import uuid4

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from orion.durable_admission.broker import ResourceBroker
from orion.durable_admission.capacity import PostgresCapacityStore
from orion.durable_admission.store import PostgresAdmissionStore
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.resource_admission import CapacityAcquireV1, CapacityTokenV1


async def main():
    dsn = os.environ["ORION_ADMISSION_TEST_DSN"]
    schema = "capacity_eval_" + uuid4().hex
    async with await AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
    async with AsyncConnectionPool(dsn, min_size=1, max_size=8, open=False,
        kwargs={"autocommit": True, "row_factory": dict_row, "options": f"-c search_path={schema},public"}) as pool:
        store = PostgresAdmissionStore(pool)
        await store.setup()
        gateways = [PostgresCapacityStore(store), PostgresCapacityStore(store)]
        broker = ResourceBroker(store, {"agent": {"backend_key": "http://fixture-agent", "configured": True,
            "healthy": True, "capabilities": {}}}, capacity=gateways[0])
        winners = {"ordinary": 0, "durable": 0}
        for index in range(40):
            run = DurableRunRequestV1(run_id=f"capacity-eval-{index}", correlation_id=str(uuid4()),
                workflow="curiosity.investigate", admission={},
                brief={"prompt": "Admission-only fixture; no cognition invoked", "session_id": "capacity-eval", "timeout_sec": 120})
            await store.submit(run.model_dump(mode="json"))
            await store.register_demand(run.run_id, run.admission.model_dump(mode="json"))
            requests = [CapacityAcquireV1(request_id=uuid4().hex, correlation_id=str(uuid4()), lane="agent",
                backend_key="http://fixture-agent" + ("/" if gateway else ""), max_inflight=1, budget_sec=120)
                for gateway in range(2)]
            operations = [gateways[0].acquire(requests[0]), gateways[1].acquire(requests[1]), broker.tick()]
            if index % 2:
                lease_grants, second, first = await asyncio.gather(*reversed(operations))
            else:
                first, second, lease_grants = await asyncio.gather(*operations)
            ordinary = [result for result in (first, second) if result["acquired"]]
            assert len(ordinary) + len(lease_grants) == 1, "ordinary request overlapped durable ownership"
            winners["ordinary" if ordinary else "durable"] += 1
            if ordinary:
                token = CapacityTokenV1(**{key: ordinary[0]["permit"][key] for key in ("request_id", "permit_id")})
                await gateways[0].release(token)
                lease_grants = await broker.tick()
            assert len(lease_grants) == 1
            # Re-validate the typed body instead of fabricating a token shape.
            owner_request = CapacityAcquireV1(**{**requests[0].model_dump(mode="json"),
                "request_id": uuid4().hex, "lease": lease_grants[0]})
            owner = await gateways[0].acquire(owner_request)
            assert owner["acquired"]
            assert not (await gateways[1].acquire(requests[1].model_copy(update={"request_id": uuid4().hex})))["acquired"]
            token = CapacityTokenV1(**{key: owner["permit"][key] for key in ("request_id", "permit_id")})
            await store.finish_projection(run.run_id, "completed", {})
            assert (await gateways[0].renew(token))["valid"], "revocation prematurely forgot running request"
            assert not (await gateways[1].acquire(requests[1].model_copy(update={"request_id": uuid4().hex})))["acquired"]
            await gateways[0].release(token)
        snapshot = await gateways[0].snapshot()
        queue = await store.queue_snapshot()
        assert not snapshot["active_permits"] and not queue["queued"] and not queue["active"]
        print(json.dumps({"verdict": "PASS", "contended_rounds": 40, "competing_gateways": 2,
            "winners": winners, "completed_durable_runs": 40, "final_capacity": snapshot,
            "final_queued": queue["queued"], "final_leases": queue["active"]}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
