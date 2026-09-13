"""Kill a real runner process while its separately owned backend stays busy.

The checkpoint, broker, ownership rows and process loss are real. A loopback
fixture replaces the Hub/model transport; the full service contract has its own
acceptance test. Only ORION_ADMISSION_TEST_DSN is used, in a fresh test schema.
"""
from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[2])]

from test_admission_runtime_postgres import DSN, Runner, request, with_database
from app.admission_runtime import AdmissionRuntime
from app.settings import Settings
from orion.durable_admission.broker import ResourceBroker
from orion.durable_admission.capacity import PostgresCapacityStore
from orion.durable_admission.store import PostgresAdmissionStore
from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1
from orion.schemas.resource_admission import CapacityAcquireV1, CapacityTokenV1

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")
BACKEND = "http://test-backend"


def permit_token(permit):
    return CapacityTokenV1(request_id=permit["request_id"], permit_id=permit["permit_id"])


class TransportRunner(Runner):
    def __init__(self, saver, endpoint):
        super().__init__(saver)
        self.endpoint = endpoint
        self.journals = []

    def _deps(self):
        deps = super()._deps()

        async def turn(req):
            self.calls.append(req)
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                response = await client.post(self.endpoint, json=req.model_dump(mode="json"))
                response.raise_for_status()
                return CuriosityTurnResultV1.model_validate(response.json())

        async def journal(entry):
            self.journals.append(entry.entry_id)
            return entry.entry_id

        deps.run_turn, deps.publish_journal = turn, journal
        return deps


def make_runtime(pool, saver, store, endpoint):
    settings = Settings(
        _env_file=None, DURABLE_RUNS_GRAPH_HOST="", POSTGRES_URI=DSN, ORION_BUS_ENABLED=False,
        DURABLE_RUNS_ADMISSION_ENABLED=True, DURABLE_RUNS_ADMISSION_SHADOW=False,
        DURABLE_RUNS_CAPACITY_ENABLED=True, DURABLE_RUNS_LEASE_HEARTBEAT_SEC=0.03,
        DURABLE_RUNS_LEASE_SECONDS=90,
    )
    capacity = PostgresCapacityStore(store)
    broker = ResourceBroker(store, lanes={"agent": {
        "backend_key": BACKEND, "configured": True, "healthy": True,
        "external_busy": False, "capabilities": {},
    }}, capacity=capacity)
    return AdmissionRuntime(settings, TransportRunner(saver, endpoint), pool, store=store, broker=broker)


async def drive_in_process(dsn, schema, endpoint, run_id):
    async with AsyncConnectionPool(dsn, min_size=1, max_size=10, open=False, kwargs={
        "autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
        "options": f"-c search_path={schema},public",
    }) as pool:
        saver = AsyncPostgresSaver(pool)
        store = PostgresAdmissionStore(pool)
        runtime = make_runtime(pool, saver, store, endpoint)
        await runtime._drive(await store.get_run(run_id))
        await runtime.close()


def test_killed_running_worker_replays_only_after_orphan_backend_drains(tmp_path):
    async def scenario(pool, saver, store):
        capacity = PostgresCapacityStore(store)
        entered, orphan_finished = asyncio.Event(), asyncio.Event()
        finish_orphan = asyncio.Event()
        backend_tasks = set()
        calls, failures = [], []
        occupancy = {"active": 0, "maximum": 0}

        async def respond(reader, writer):
            task = asyncio.current_task()
            backend_tasks.add(task)
            acquired = None
            index = None
            try:
                head = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
                length = next(int(line.split(":", 1)[1]) for line in head.split("\r\n")
                              if line.lower().startswith("content-length:"))
                req = CuriosityTurnRequestV1.model_validate_json(await reader.readexactly(length))
                acquired = await capacity.acquire(CapacityAcquireV1(
                    request_id=uuid4().hex, correlation_id=req.correlation_id,
                    lane=req.lease.lane, backend_key=req.lease.backend_key,
                    max_inflight=1, budget_sec=req.timeout_sec, lease=req.lease,
                ))
                assert acquired["acquired"], acquired
                index = len(calls)
                calls.append({"request": req, "permit": acquired["permit"]})
                occupancy["active"] += 1
                occupancy["maximum"] = max(occupancy["maximum"], occupancy["active"])
                if index == 0:
                    entered.set()
                    # Like an already-dispatched blocking Gateway thread, this
                    # work survives loss of the original runner connection.
                    await finish_orphan.wait()
                result = CuriosityTurnResultV1(
                    run_id=req.run_id, correlation_id=req.correlation_id,
                    text="orphan result must be discarded" if index == 0 else "Recovered study fixture evidence",
                )
                body = result.model_dump_json().encode()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                             + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
                try:
                    await writer.drain()
                except (ConnectionError, BrokenPipeError):
                    pass  # The old runner was killed while this request ran.
            except Exception as exc:
                failures.append(exc)
                entered.set()
            finally:
                if acquired and acquired["acquired"]:
                    await capacity.release(permit_token(acquired["permit"]))
                    occupancy["active"] -= 1
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, BrokenPipeError):
                    pass
                backend_tasks.discard(task)
                if index == 0:
                    orphan_finished.set()

        server = await asyncio.start_server(respond, "127.0.0.1", 0)
        endpoint = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/turn"
        runtime = make_runtime(pool, saver, store, endpoint)
        req = request("worker-kill-recovery")
        await runtime.submit(req)
        old_lease = (await runtime.broker.tick())[0]
        async with pool.connection() as conn:
            schema = (await (await conn.execute("SELECT current_schema() AS schema")).fetchone())["schema"]
        log_path = tmp_path / "killed-runner.log"
        process = None
        try:
            with log_path.open("wb") as log:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), DSN, schema, endpoint, req.run_id],
                    stdout=log, stderr=subprocess.STDOUT,
                )
                try:
                    await asyncio.wait_for(entered.wait(), 15)
                except TimeoutError:
                    pytest.fail(f"runner did not reach inference: {log_path.read_text()}")
                assert not failures, failures
                assert calls[0]["request"].lease.generation == old_lease["generation"]
                checkpoint = await runtime.graph.aget_state(runtime.config(req.run_id))
                assert checkpoint.next == ("harness_turn",)
                assert checkpoint.values["status"] == "running"
                assert "text" not in checkpoint.values
                process.kill()  # SIGKILL: no runner finally/close callback runs.
                assert await asyncio.to_thread(process.wait, 5) == -signal.SIGKILL

            # Recreate the driver/checkpointer after real process death. The
            # broker may not steal the still-live backend request permit.
            recovered = make_runtime(pool, AsyncPostgresSaver(pool), store, endpoint)
            await recovered._drive(await store.get_run(req.run_id))
            assert not await store.validate(old_lease)
            assert not recovered.runner.calls
            checkpoint = await recovered.graph.aget_state(recovered.config(req.run_id))
            assert checkpoint.next == ("resource_wait",)
            assert checkpoint.values["status"] == "waiting_resource"
            assert not recovered.active
            old_permit = permit_token(calls[0]["permit"])
            assert (await capacity.renew(old_permit))["valid"]
            stale = CapacityAcquireV1(
                request_id=uuid4().hex, correlation_id=req.correlation_id,
                lane=old_lease["lane"], backend_key=BACKEND, max_inflight=1,
                budget_sec=30, lease=old_lease,
            )
            assert (await capacity.acquire(stale))["reason"] == "resource_lease_stale"
            for _ in range(3):
                assert await recovered.broker.tick() == []
                await recovered._drive(await store.get_run(req.run_id))
            assert len(calls) == 1 and occupancy["active"] == 1
            assert len((await capacity.snapshot())["active_permits"]) == 1

            finish_orphan.set()
            await asyncio.wait_for(orphan_finished.wait(), 3)
            assert not failures, failures
            new_lease = (await recovered.broker.tick())[0]
            assert new_lease["generation"] > old_lease["generation"]
            assert new_lease["lane"] == old_lease["lane"]
            assert not await store.validate(old_lease)
            row = await store.get_run(req.run_id)
            # Duplicate wakeups/replica drivers must not dispatch a second
            # replay or write a second terminal result.
            await asyncio.gather(recovered._drive(row), recovered._drive(row))
            await recovered._drive(await store.get_run(req.run_id))
            status = await recovered.status(req.run_id)
            assert status["status"] == "completed"
            checkpoint = await recovered.graph.aget_state(recovered.config(req.run_id))
            assert checkpoint.values["text"] == "Recovered study fixture evidence"
            assert len(calls) == 2
            assert calls[0]["request"].correlation_id != calls[1]["request"].correlation_id
            assert calls[1]["request"].lease.generation == new_lease["generation"]
            assert len(recovered.runner.journals) == 1
            assert occupancy == {"active": 0, "maximum": 1}
            assert (await capacity.snapshot())["active_permits"] == []
            assert await store.get_lease(req.run_id) is None
            history = await store.history(req.run_id)
            assert sum(item["event"] == "run.completed" for item in history) == 1
            assert any(item["event"] == "resource.lease_released"
                       and item["detail"].get("reason") == "worker_recovery" for item in history)
            await recovered.close()
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                await asyncio.to_thread(process.wait, 5)
            finish_orphan.set()
            await asyncio.gather(*backend_tasks, return_exceptions=True)
            server.close()
            await server.wait_closed()
            await runtime.close()

    asyncio.run(with_database(scenario))


if __name__ == "__main__":
    asyncio.run(drive_in_process(*sys.argv[1:]))
