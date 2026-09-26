"""A pool that cannot take a hold request right now must not fail the run (incident 2026-09-26).

durable-runs 4.5 came up ~20 s before the 4.5 GPU pool. The old pool's request model did not know
``hold_lease_id``/``hold_generation`` and is extra="forbid", so it answered every acquire
``unavailable reason=invalid:2 validation errors for GpuLeaseRequestV1 ... extra_forbidden`` --
and durable-runs failed all 11 resumed runs terminally. Here the SAME refusal is reproduced by
validating the real wire payload against the pre-4.x request model (the new fields removed), and
the run must wait, back off visibly, and complete once the pool answers normally.

Real Postgres, real in-process pool, real client + codec (tests/pool_fixture.py).
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ConfigDict, create_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pool_hold import is_pool_trouble, refusal_is_terminal  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope  # noqa: E402
from orion.schemas.gpu_pool import GPU_LEASE_REPLY_KIND, GpuLeaseReplyV1, GpuLeaseRequestV1  # noqa: E402
from pool_fixture import CFG, SOURCE, InProcessPool, PoolBus  # noqa: E402
from test_admission_runtime_postgres import (  # noqa: E402
    DSN, occupy_agent, pool_event, request, runtime, until, with_database,
)

pg = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")

# The pool's request model before stage 4.1 added attach: same fields minus the hold ref, and
# extra="forbid" -- exactly what answered durable-runs on 2026-09-26.
_OLD_FIELDS = {name: (field.annotation, field) for name, field in GpuLeaseRequestV1.model_fields.items()
               if name not in ("hold_lease_id", "hold_generation")}
OldPoolLeaseRequest = create_model("GpuLeaseRequestV1", __config__=ConfigDict(extra="forbid"), **_OLD_FIELDS)


class ScriptedPoolBus(PoolBus):
    """PoolBus with two knobs: ``old_pool`` (every RPC answered the way the pre-4.x pool's
    services/orion-gpu-pool/app/main.py ``_on_lease`` answers a payload its model rejects) and
    ``script`` (verb -> reply override, for pool states the fake clock cannot reach cheaply)."""

    def __init__(self, pool):
        super().__init__(pool)
        self.old_pool = False
        self.old_verbs: set[str] | None = None   # None = every verb
        self.script: dict[str, object] = {}
        self.acquires = 0

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec, health_label=None):
        wire = self.codec.decode(self.codec.encode(envelope)).envelope
        verb = (wire.payload or {}).get("verb")
        if verb == "acquire":
            self.acquires += 1
        reply = None
        if self.old_pool and (self.old_verbs is None or verb in self.old_verbs):
            try:
                OldPoolLeaseRequest.model_validate(wire.payload)
            except Exception as exc:  # noqa: BLE001 -- mirrors the pool's own handler
                reply = GpuLeaseReplyV1(status="unavailable", reason=f"invalid:{exc}"[:300])
        if reply is None and verb in self.script:
            made = self.script[verb]
            reply = made(GpuLeaseRequestV1.model_validate(wire.payload)) if callable(made) else made
            if asyncio.iscoroutine(reply):
                reply = await reply
        if reply is None:
            return await super().rpc_request(channel, envelope, reply_channel=reply_channel,
                                             timeout_sec=timeout_sec, health_label=health_label)
        env = BaseEnvelope(kind=GPU_LEASE_REPLY_KIND, source=SOURCE, correlation_id=envelope.correlation_id,
                           payload=reply.model_dump(mode="json"))
        return {"type": "message", "channel": reply_channel, "data": self.codec.encode(env)}


class FakeNow:
    """rt.now under test control: backoff checks never race a slow Postgres round trip."""

    def __init__(self):
        self.t = datetime.now(timezone.utc)

    def __call__(self):
        return self.t

    def advance(self, sec):
        self.t += timedelta(seconds=sec)


def fake_now(rt) -> FakeNow:
    clock = FakeNow()
    rt.now = clock
    return clock


def scripted(rt) -> ScriptedPoolBus:
    bus = ScriptedPoolBus(rt.gpu)
    rt.holds.bus = bus
    return bus


async def waiting_events(store, run_id):
    return [e["detail"] for e in await store.history(run_id)
            if e["event"] == "run.waiting_resource" and e["detail"].get("transient")]


# --- classification (no database) ------------------------------------------------------------------
@pytest.mark.parametrize("reason,terminal", [
    ("invalid:2 validation errors for GpuLeaseRequestV1 hold_lease_id Extra inputs are not permitted", False),
    ("unknown_verb:attach", False),
    ("unknown_class:agent", False),          # we know the class: the pool runs another gpu_pool.yaml
    ("unknown_class:no-such-class", True),   # neither side knows it
    ("operator_only_class", False),          # our config has no operator-only role in agent
    ("deadline", True),
    ("min_ctx_exceeds_class:262144", True),
    ("backlog_max_age", True),
    ("replay_payload_too_large", True),
    ("unknown_lease", False),
    ("", False),
    ("some_reason_nobody_classified", False),
])
def test_only_refusals_about_the_run_itself_are_terminal(reason, terminal):
    assert refusal_is_terminal(CFG, reason, "agent") is terminal


def test_a_skew_refusal_to_a_lease_verb_says_nothing_about_the_lease():
    assert is_pool_trouble(GpuLeaseReplyV1(status="unavailable", reason="invalid:bad"))
    # the real pool echoes the asked-about lease_id on an unknown verb (gpu-pool main.py dispatch_lease)
    assert is_pool_trouble(GpuLeaseReplyV1(status="unavailable", lease_id="L1", reason="unknown_verb:status"))
    assert not is_pool_trouble(GpuLeaseReplyV1(status="unavailable", lease_id="L1", reason="deadline"))
    assert not is_pool_trouble(GpuLeaseReplyV1(status="unknown_lease", lease_id="L1"))


# --- the incident, end to end ---------------------------------------------------------------------
@pg
def test_version_skewed_pool_refusal_waits_backs_off_and_completes_when_the_pool_is_upgraded():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, DURABLE_RUNS_POOL_RETRY_BASE_SEC=10,
                     DURABLE_RUNS_POOL_RETRY_MAX_SEC=30)
        clock = fake_now(rt)
        bus = scripted(rt)
        bus.old_pool = True
        req = request("skew-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))

        # Not failed: waiting, with the pool's own reason on a visible event.
        assert (await store.get_run(req.run_id))["terminal"] is None
        snap = await rt.graph.aget_state(rt.config(req.run_id))
        assert snap.next == ("resource_wait",)
        assert snap.values["hold"]["request_id"] == "skew-001:1" and snap.values["hold"]["refusals"] == 1
        [first] = await waiting_events(store, req.run_id)
        assert first["pool_status"] == "unavailable"
        assert first["reason"].startswith("invalid:") and "extra_forbidden" in first["reason"]
        assert first["work_class"] == "agent" and first["refusals"] == 1 and first["retry_at"]
        assert not gpu.leases(), "the old pool created nothing"

        # Inside the backoff nothing is asked again -- no retry storm against a skewed pool.
        asked = bus.acquires
        await rt._drive(await store.get_run(req.run_id))
        assert bus.acquires == asked

        # Backoff over, pool still old: more refusals, still waiting; delays 10 s, 20 s, 30 s (40 capped).
        for wait in (10, 20):
            clock.advance(wait - 0.5)
            await rt._drive(await store.get_run(req.run_id))
            assert bus.acquires == asked, "asked again before retry_at"
            clock.advance(0.5)
            await rt._drive(await store.get_run(req.run_id))
            assert bus.acquires > asked
            asked = bus.acquires
        assert (await store.get_run(req.run_id))["terminal"] is None
        events = await waiting_events(store, req.run_id)
        assert [e["refusals"] for e in events] == [1, 2, 3]
        gaps = [(datetime.fromisoformat(e["retry_at"]) - datetime.fromisoformat(p["retry_at"])).total_seconds()
                for p, e in zip(events, events[1:])]
        assert gaps == [20.0, 30.0]   # each ask happens at retry_at(n-1), so the gap is delay(n)

        # The upgraded pool comes up: the same request id is granted and the run completes.
        bus.old_pool = False
        clock.advance(30)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert len(rt.runner.calls) == 1
        assert [r["request_id"] for r in gpu.leases(holder="durable-runs:skew-001")] == ["skew-001:1"]
        assert not any(e["event"] == "run.failed" for e in await store.history(req.run_id))
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_rpc_timeout_backs_off_visibly_and_the_run_completes():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, DURABLE_RUNS_POOL_RETRY_BASE_SEC=10,
                     DURABLE_RUNS_POOL_RETRY_MAX_SEC=300)
        clock = fake_now(rt)
        bus = scripted(rt)
        bus.fail_next = 2
        req = request("timeout-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        clock.advance(10)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] is None
        events = await waiting_events(store, req.run_id)
        assert [e["pool_status"] for e in events] == ["unreachable", "unreachable"]
        assert all("TimeoutError" in e["reason"] for e in events)
        snap = await rt.graph.aget_state(rt.config(req.run_id))
        assert snap.values["hold"]["refusals"] == 2
        clock.advance(19)                                     # 2nd backoff is 20 s: not over yet
        await rt._drive(await store.get_run(req.run_id))
        assert bus.fail_next == 0 and (await store.get_run(req.run_id))["terminal"] is None
        clock.advance(1)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_a_timed_out_acquire_that_landed_is_woken_by_its_grant_event_mid_backoff():
    """The acquire RPC timed out but the pool took it and granted it: its granted event wakes the
    run at once instead of leaving the seat idle until retry_at."""
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, DURABLE_RUNS_POOL_RETRY_BASE_SEC=300)
        fake_now(rt)
        bus = scripted(rt)

        async def landed_but_lost(req):
            await gpu.dispatch(req)                    # the pool grants it...
            raise TimeoutError("fixture: reply lost")  # ...but the reply never arrives
        bus.script["acquire"] = landed_but_lost
        req = request("landed-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] is None
        del bus.script["acquire"]
        [hold] = gpu.leases(holder="durable-runs:landed-001")
        await rt.on_pool_event(pool_event(gpu, hold["lease_id"], "granted"))
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert [r["request_id"] for r in gpu.leases(holder="durable-runs:landed-001")] == ["landed-001:1"]
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_a_status_skew_while_waiting_keeps_the_hold_and_it_is_granted_later():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        blocker = await occupy_agent(gpu)
        rt = runtime(pool, saver, store, gpu=gpu, DURABLE_RUNS_POOL_RETRY_BASE_SEC=10)
        clock = fake_now(rt)
        bus = scripted(rt)
        req = request("skew-status-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        [hold] = gpu.leases(holder="durable-runs:skew-status-001")
        bus.old_pool = True
        await rt.on_pool_event({"holder": "durable-runs:skew-status-001", "event": "retried"})
        await rt._drive(await store.get_run(req.run_id))
        assert (await gpu.lease(hold["lease_id"]))["status"] == "queued", "never released on a skew reply"
        assert not any(e["event"] == "resource.lease_released" for e in await store.history(req.run_id))
        snap = await rt.graph.aget_state(rt.config(req.run_id))
        assert snap.values["hold"]["lease_id"] == hold["lease_id"]
        bus.old_pool = False
        await gpu.dispatch(GpuLeaseRequestV1(verb="release", lease_id=blocker, outcome="ok"))
        clock.advance(10)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert len(gpu.leases(holder="durable-runs:skew-status-001")) == 1
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_a_skewed_release_is_retried_until_the_pool_can_read_it():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        bus = scripted(rt)
        req = request("skew-release-001")
        await rt.submit(req)
        bus.old_pool, bus.old_verbs = True, {"release"}
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        [hold] = gpu.leases(holder="durable-runs:skew-release-001")
        assert hold["status"] == "granted" and hold["lease_id"] in rt._pending_release
        assert not any(e["event"] == "resource.lease_released" for e in await store.history(req.run_id))
        bus.old_pool = False
        await rt.reconcile()
        assert (await gpu.lease(hold["lease_id"]))["status"] == "released" and not rt._pending_release
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_a_skewed_boundary_heartbeat_keeps_the_lease():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        bus = scripted(rt)
        req = request("skew-guard-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        [hold] = gpu.leases(holder="durable-runs:skew-guard-001")
        lease = {"lease_id": hold["lease_id"], "generation": hold["generation"], "role": hold["role"],
                 "holder": "durable-runs:skew-guard-001"}
        # a completed run's row has no control; guard reads it, then heartbeats the hold
        await gpu.dispatch(GpuLeaseRequestV1(verb="acquire", request_id="skew-guard-001:9", kind="hold",
                           holder="durable-runs:x", work_class="agent", priority="background", retryable=True))
        bus.old_pool, bus.old_verbs = True, {"heartbeat"}
        state = {"run_id": req.run_id, "admission": {}, "lease": lease}
        assert await rt.guard(state) == lease
        assert not any(e["event"] == "resource.lease_expired" for e in await store.history(req.run_id))
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_a_pool_config_roll_unknown_class_waits_but_a_class_nobody_knows_fails():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, DURABLE_RUNS_POOL_RETRY_BASE_SEC=0.05)
        bus = scripted(rt)
        bus.script["acquire"] = GpuLeaseReplyV1(status="unavailable", reason="unknown_class:agent")
        req = request("roll-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] is None
        del bus.script["acquire"]
        await asyncio.sleep(0.06)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"

        bus.script["acquire"] = GpuLeaseReplyV1(status="unavailable", reason="operator_only_class")
        ok = request("roll-002")
        await rt.submit(ok)
        await rt._drive(await store.get_run(ok.run_id))
        assert (await store.get_run(ok.run_id))["terminal"] is None   # our config: not operator-only

        bus.script["acquire"] = GpuLeaseReplyV1(status="unavailable", reason="unknown_class:retired-class")
        gone = request("roll-003")
        await rt.submit(gone)
        await rt._drive(await store.get_run(gone.run_id))
        failed = next(e for e in await store.history(gone.run_id) if e["event"] == "run.failed")
        assert failed["detail"]["error"] == "gpu_pool_unavailable:unknown_class:retired-class"
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_a_skewed_heartbeat_mid_turn_does_not_kill_the_turn():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        block = asyncio.Event()
        rt = runtime(pool, saver, store, block, gpu=gpu)
        bus = scripted(rt)
        req = request("skew-beat-001")
        await rt.submit(req)
        driver = asyncio.create_task(rt._drive(await store.get_run(req.run_id)))
        await until(lambda: rt.runner.calls)
        bus.old_pool, bus.old_verbs = True, {"heartbeat"}
        await asyncio.sleep(0.15)                     # several 0.03 s beats, all refused as invalid
        bus.old_pool = False
        block.set()
        await asyncio.wait_for(driver, 5)
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert len(rt.runner.calls) == 1, "the turn was not stopped and replayed"
        [hold] = gpu.leases(holder="durable-runs:skew-beat-001")
        assert hold["status"] == "released"
        await rt.close()
    asyncio.run(with_database(scenario))


# --- spec Decision 1 rule 6: backlogged waits, unavailable:deadline fails --------------------------
@pg
def test_a_backlogged_hold_keeps_waiting_and_is_granted_when_a_role_comes_up():
    """Spec Decision 1 rule 6: ``backlogged`` -> interrupt again, never fail. (The in-process pool
    backlogs only when no role of the class is usable OR loadable -- agent-gpu2 is always loadable
    in config/gpu_pool.yaml -- so the pool's answer is scripted; everything else is real.)"""
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, DURABLE_RUNS_HOLD_STATUS_POLL_SEC=0.01)
        bus = scripted(rt)

        async def backlogged(req):
            real = await gpu.dispatch(req)
            return GpuLeaseReplyV1(status="backlogged", lease_id=real.lease_id, reason="no_serviceable_role")
        bus.script["acquire"] = backlogged
        bus.script["status"] = backlogged
        req = request("backlog-001")
        await rt.submit(req)
        for _ in range(3):
            await rt._drive(await store.get_run(req.run_id))
            await asyncio.sleep(0.02)
        assert (await store.get_run(req.run_id))["terminal"] is None
        assert not rt.runner.calls
        waiting = [e["detail"] for e in await store.history(req.run_id)
                   if e["event"] == "run.waiting_resource" and "pool_status" in e["detail"]]
        assert [w["pool_status"] for w in waiting] == ["backlogged"]
        bus.script.clear()                                  # a role came up: the pool grants it
        await asyncio.sleep(0.02)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_pool_deadline_on_a_waiting_hold_fails_the_run_as_workflow_deadline():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        await occupy_agent(gpu)
        rt = runtime(pool, saver, store, gpu=gpu)
        bus = scripted(rt)
        req = request("deadline-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        [hold] = gpu.leases(holder="durable-runs:deadline-001")
        # The hold's deadline_at is the run's: the pool gave up on it.
        bus.script["status"] = GpuLeaseReplyV1(status="unavailable", lease_id=hold["lease_id"], reason="deadline")
        await rt.on_pool_event({"holder": "durable-runs:deadline-001", "event": "unavailable"})
        await rt._drive(await store.get_run(req.run_id))
        failed = next(e for e in await store.history(req.run_id) if e["event"] == "run.failed")
        assert failed["detail"]["error"] == "workflow_deadline"
        await rt.close()
    asyncio.run(with_database(scenario))
