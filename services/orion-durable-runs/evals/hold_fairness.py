"""Periodic eval: durable runs on GPU pool holds (stage 4.5), against a disposable Postgres DSN.

Replaces admission_fairness.py and elastic_fairness.py, which exercised the durable broker and the
gpu2 elastic decider deleted in 4.5. The same questions, now answered by the pool through the real
durable-runs runtime, the real in-process pool runtime (fake clock, fixture llama.cpp servers and
a fixture gpu2 actuator) and real Postgres checkpoints:

A. home card, 20 runs: grants are FIFO by arrival, at most one hold on the agent card at a time,
   each run holds exactly one hold, and during every run a system-priority agent call still gets
   the card between the run's own calls (shared gaps, Juniper 2026-09-25) without waiting.
B. gpu2: with the agent card held for a long run, waiting runs past the seat's 1200 s trigger make
   the pool load the second 27B; the oldest waiting run is granted on agent-gpu2 first; when gpu2
   idles past swap_idle_unload_sec the pool unloads it and diffusion is restored.

Prints inspectable JSON; exits non-zero on any failed check. No model inference or production bus.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE.parents[1]), str(SERVICE), str(SERVICE / "tests")]

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: E402
from psycopg import AsyncConnection  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402
from psycopg_pool import AsyncConnectionPool  # noqa: E402

from app.admission_runtime import AdmissionRuntime  # noqa: E402
from app.graph import Deps  # noqa: E402
from app.pool_hold import PoolHolds  # noqa: E402
from app.settings import Settings  # noqa: E402
from orion.durable_admission.store import PostgresAdmissionStore  # noqa: E402
from orion.schemas.durable_run import CuriosityTurnResultV1, DurableRunRequestV1  # noqa: E402
from orion.schemas.gpu_pool import GpuActuateResultV1, GpuActuateV1, GpuLeaseRequestV1  # noqa: E402
from pool_fixture import CFG, LIVE, InProcessPool, PoolBus  # noqa: E402

SEAT = "agent-gpu2"


class EvalRunner:
    """Fixture cognition: each turn makes one interleaved system agent call on the pool (what
    cortex-exec does between a run's calls) and returns a fixed finding."""

    def __init__(self, saver, gpu):
        self._checkpointer, self._bus, self.gpu = saver, None, gpu
        self.turns: list[tuple[str, str]] = []            # (run_id, role the hold landed on)
        self.interleaved: list[str] = []                  # status of each system call made mid-run

    def _curiosity_deps(self):
        async def turn(req):
            self.turns.append((req.run_id, req.gpu_lease.role))
            probe = await self.gpu.dispatch(GpuLeaseRequestV1(verb="acquire", request_id=uuid4().hex,
                holder="cortex-exec", work_class="agent", priority="system"))
            self.interleaved.append(f"{probe.status}:{probe.grant.role if probe.grant else ''}")
            if probe.lease_id:
                await self.gpu.dispatch(GpuLeaseRequestV1(verb="release", lease_id=probe.lease_id, outcome="ok"))
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id,
                                         text="Eval fixture finding")

        async def read(run_id):
            return {"graph_readable": True}

        async def ok(_):
            return True

        async def journal(entry):
            return entry.entry_id

        return Deps(turn, read, ok, journal)

    def _self_sense_deps(self):
        from app.self_sense_graph import Deps as SelfSenseDeps

        async def turn(req):
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text="x")

        async def rows(rows):
            return len(rows), 0

        return SelfSenseDeps(run_turn=turn, publish_rows=rows)

    def _reflect_deps(self):
        from app.reflect_graph import Deps as ReflectDeps

        async def call(reflect_input, llm_route, gpu_lease=None):
            return []

        return ReflectDeps(call_reflect_llm=call)

    async def _publish(self, channel, kind, model, corr):
        return True

    def _corr_for_admission(self, value):
        return value


def granted_holds(gpu) -> list[tuple[str, str]]:
    return [(e["holder"].split(":", 1)[1], e.get("role")) for e in gpu.events("granted")
            if str(e.get("holder") or "").startswith("durable-runs:eval-")]


async def drive_all(rt, store, run_ids, *, rounds=60):
    for _ in range(rounds):
        pending = [r for r in run_ids if not (await store.get_run(r))["terminal"]]
        if not pending:
            return
        for run_id in pending:
            await rt._drive(await store.get_run(run_id))
    raise AssertionError("runs did not finish")


async def scenario_home(pool, saver, store) -> dict:
    gpu = await InProcessPool().boot()
    runner = EvalRunner(saver, gpu)
    settings = Settings(_env_file=None, POSTGRES_URI="postgresql://eval", ORION_BUS_ENABLED=False,
                        DURABLE_RUNS_ADMISSION_ENABLED=True, DURABLE_RUNS_HOLD_STATUS_POLL_SEC=0.0001,
                        DURABLE_RUNS_LEASE_HEARTBEAT_SEC=0.05)
    rt = AdmissionRuntime(settings, runner, pool, store=store, holds=PoolHolds(PoolBus(gpu), source="eval", cfg=CFG))
    run_ids = []
    for index in range(20):
        req = DurableRunRequestV1(run_id=f"eval-home-{index:02d}", workflow="curiosity.investigate",
            correlation_id=str(uuid4()), admission={}, brief={"prompt": "Eval-only fixture; no cognition invoked",
            "session_id": "hold-eval", "timeout_sec": 120})
        await rt.submit(req)
        # Every run asks the pool (and joins its queue) in arrival order, a second apart.
        gpu.clock.advance(1)
        await rt._drive(await store.get_run(req.run_id))
        run_ids.append(req.run_id)
    await drive_all(rt, store, run_ids)
    grants = granted_holds(gpu)
    order = [run for run, _ in grants]
    holds_per_run = {r: len(gpu.leases(holder=f"durable-runs:{r}", kind="hold")) for r in run_ids}
    concurrent = max(sum(1 for r in gpu.leases(kind="hold") if r["status"] in ("granted", "recalling")), 0)
    await rt.close()
    checks = {
        "fifo_grant_order": order == run_ids,
        "one_hold_per_run": set(holds_per_run.values()) == {1},
        "all_completed": [(await store.get_run(r))["terminal"] for r in run_ids] == ["completed"] * len(run_ids),
        "no_hold_left_granted": concurrent == 0,
        "interleaved_system_calls_granted_on_agent": runner.interleaved == ["granted:agent"] * len(run_ids),
        "all_turns_on_agent": {role for _, role in runner.turns} == {"agent"},
    }
    return {"runs": len(run_ids), "grant_order_head": order[:5], "interleaved": sorted(set(runner.interleaved)),
            "checks": checks}


async def scenario_gpu2(pool, saver, store) -> dict:
    gpu = await InProcessPool(actuate=(SEAT,)).boot()
    actions: list[tuple[str, str]] = []

    async def actuator(channel, env):
        if channel != "orion:gpu_pool:actuate:request":
            return
        msg = GpuActuateV1.model_validate(env.payload)
        actions.append((msg.role, msg.action))

        async def answer(status, **kw):
            await gpu.rt.on_actuate_result(GpuActuateResultV1(action_id=msg.action_id, generation=msg.generation,
                role=msg.role, action=msg.action, status=status, **kw))

        await answer("accepted")
        if msg.action == "load":
            gpu.live[SEAT] = LIVE["agent"]
            gpu.down.add("diffusion")
            await answer("succeeded", observed={SEAT: "running", "diffusion": "exited"})
        else:
            gpu.live.pop(SEAT, None)
            gpu.down.discard("diffusion")
            await answer("succeeded", observed={SEAT: "exited", "diffusion": "running"})

    tasks: set = set()

    class ActuatorBus:
        """Delivers like the real bus: later, never inside the pool's own lock."""
        async def publish(self, channel, env):
            task = asyncio.ensure_future(actuator(channel, env))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

    gpu.bus = ActuatorBus()
    runner = EvalRunner(saver, gpu)
    settings = Settings(_env_file=None, POSTGRES_URI="postgresql://eval", ORION_BUS_ENABLED=False,
                        DURABLE_RUNS_ADMISSION_ENABLED=True, DURABLE_RUNS_HOLD_STATUS_POLL_SEC=0.0001,
                        DURABLE_RUNS_LEASE_HEARTBEAT_SEC=0.05)
    rt = AdmissionRuntime(settings, runner, pool, store=store, holds=PoolHolds(PoolBus(gpu), source="eval", cfg=CFG))
    home = await gpu.dispatch(GpuLeaseRequestV1(verb="acquire", request_id="eval-long-run:1", kind="hold",
        holder="durable-runs:eval-long-run", work_class="agent", priority="background", retryable=True))
    run_ids = []
    for index in range(3):
        req = DurableRunRequestV1(run_id=f"eval-gpu2-{index}", workflow="curiosity.investigate",
            correlation_id=str(uuid4()), admission={}, brief={"prompt": "Eval-only fixture; no cognition invoked",
            "session_id": "hold-eval", "timeout_sec": 120})
        await rt.submit(req)
        gpu.clock.advance(1)
        await rt._drive(await store.get_run(req.run_id))
        run_ids.append(req.run_id)

    async def settle(sec, beat=()):
        """Advance in 5 s steps and let the actuator answer between them (its ack budget is 10 s)."""
        for _ in range(int(sec // 5)):
            await gpu.later(5, beat=beat)
            await asyncio.gather(*tasks)

    before_trigger = list(actions)
    await gpu.later(CFG.swap_after_wait_sec(SEAT) - 30, beat=[home.lease_id])
    assert actions == [], "the pool loaded gpu2 before the 1200 s trigger"
    await settle(90, beat=[home.lease_id])                                     # load, then discovery confirms
    await drive_all(rt, store, run_ids)
    await gpu.dispatch(GpuLeaseRequestV1(verb="release", lease_id=home.lease_id, outcome="ok"))
    await settle(CFG.defaults.swap_idle_unload_sec + 60)                       # gpu2 idles: unload, restore
    await rt.close()
    grants = granted_holds(gpu)
    checks = {
        "no_load_before_trigger": before_trigger == [],
        "pool_loaded_then_unloaded_gpu2": actions == [(SEAT, "load"), (SEAT, "unload")],
        "no_failed_actuation": not gpu.events("swap_failed"),
        "fifo_on_gpu2": [run for run, role in grants if role == SEAT] == run_ids,
        "all_completed": [(await store.get_run(r))["terminal"] for r in run_ids] == ["completed"] * len(run_ids),
        "diffusion_restored": "diffusion" not in gpu.down,
        "turns_ran_on_gpu2": {role for _, role in runner.turns} == {SEAT},
    }
    return {"actions": actions, "grants": grants, "checks": checks}


async def with_schema(dsn, name, scenario):
    schema = f"hold_eval_{name}_" + uuid4().hex
    async with await AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
    async with AsyncConnectionPool(dsn, min_size=1, max_size=8, open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                    "options": f"-c search_path={schema},public"}) as pool:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        store = PostgresAdmissionStore(pool)
        await store.setup()
        return await scenario(pool, saver, store)


async def main() -> int:
    dsn = os.environ["ORION_ADMISSION_TEST_DSN"]
    report = {"home_card": await asyncio.wait_for(with_schema(dsn, "home", scenario_home), 300),
              "gpu2": await asyncio.wait_for(with_schema(dsn, "gpu2", scenario_gpu2), 300)}
    failed = [f"{name}.{check}" for name, part in report.items() for check, ok in part["checks"].items() if not ok]
    report["verdict"] = "FAIL" if failed else "PASS"
    report["failed_checks"] = failed
    print(json.dumps(report, indent=2, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
