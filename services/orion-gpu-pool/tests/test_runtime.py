"""End-to-end runtime behaviour: real scheduler, real lease graph (MemorySaver), in-memory store,
fake bus. Every test drives the runtime through its public verbs and tick, like production."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from langgraph.checkpoint.memory import MemorySaver

from app.runtime import PoolRuntime
from app.store import MemoryStore
from orion.gpu_pool.config import load_pool_config
from orion.gpu_pool.discovery import Probe, load_profiles
from orion.gpu_pool.lease_graph import build_lease_graph
from orion.schemas.gpu_pool import GpuLeaseRequestV1, GpuPoolControlV1, LlmWorkerAnnounceV1

CFG = load_pool_config()
PROFILES = load_profiles()
TOKEN = "t0ken"
LIVE = {  # what circe's llama.cpp servers report today (see spec, "The machine")
    "chat": ("qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition", "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf", 1, 65536),
    "agent": ("qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex", "Qwen3.8-27B-UD-Q4_K_XL.gguf", 1, 131072),
    "metacog": ("qwen3-8b-q5km-v100-16gb-atlas-metacog-16k", "Qwen_Qwen3-8B-Q5_K_M.gguf", 4, 4096),
    "fast": ("qwen3-8b-q4km-v100-16gb-balanced", "Qwen_Qwen3-8B-Q4_K_M.gguf", 4, 4096),
}


class Clock:
    def __init__(self):
        self.t = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def advance(self, sec):
        self.t += timedelta(seconds=sec)


class FakeBus:
    def __init__(self):
        self.published, self.hops = [], []

    async def publish(self, channel, env):
        self.published.append((channel, env))

    def record_hop_success(self, hop, ms):
        self.hops.append(("ok", hop, ms))

    def record_hop_timeout(self, hop, ms=None):
        self.hops.append(("timeout", hop, ms))

    def events(self, name=None):
        out = [e.payload for c, e in self.published if c == "orion:gpu_pool:event"]
        return [e for e in out if name is None or e["event"] == name]

    def grammar(self):
        return [e.payload for c, e in self.published if c == "orion:grammar:event"]


def make(down=(), store=None, saver=None, clock=None, bus=None):
    clock = clock or Clock()

    async def prober(role, url, kind, health):
        if role in down:
            return Probe(False, error="refused", checked_at=clock())
        if kind == "service":
            return Probe(True, checked_at=clock())
        if role not in LIVE:
            return Probe(False, error="refused", checked_at=clock())
        _, file, slots, ctx = LIVE[role]
        return Probe(True, {"model_path": f"/models/gguf/{file}", "total_slots": slots,
                            "default_generation_settings": {"n_ctx": ctx}, "modalities": {"vision": False}},
                     checked_at=clock())

    rt = PoolRuntime(cfg=CFG, profiles=PROFILES, store=store or MemoryStore(),
                     graph=build_lease_graph(lambda: CFG, saver or MemorySaver()), bus=bus or FakeBus(),
                     prober=prober, now=clock, operator_token=TOKEN, probe_interval_sec=0)
    return rt, clock


async def announce(rt):
    for role, (profile, _, _, _) in LIVE.items():
        await rt.on_announce(LlmWorkerAnnounceV1(host="circe", role=role, profile_name=profile,
                                                 port=CFG.roles[role].port, announced_at=rt.now()))


async def boot(rt):
    await rt.start()
    await announce(rt)
    await rt.tick()
    return rt


async def later(rt, clock, sec):
    """Advance time the way production does: workers keep re-announcing every 30s."""
    clock.advance(sec)
    await announce(rt)
    await rt.tick()


def acq(cls, rid=None, **kw):
    return GpuLeaseRequestV1(verb="acquire", work_class=cls, holder="test", request_id=rid, **kw)


def acq_r(cls, rid=None, **kw):
    """A caller that will use a re-grant (durable run / gateway replay): retries and backlog apply."""
    return acq(cls, rid, retryable=True, **kw)


def run(coro):
    return asyncio.run(coro)


def test_discovery_confirms_live_roles_and_grants_with_the_discovered_model():
    async def go():
        rt, _ = make()
        await boot(rt)
        status = {d.role: d.status for d in rt.discovered}
        assert status["metacog"] == "confirmed" and status["agent-gpu2"] == "unloaded"
        r = await rt.acquire(acq("metacog"))
        assert r.status == "granted" and r.grant.role == "metacog"
        assert r.grant.model_file == "Qwen_Qwen3-8B-Q5_K_M.gguf" and r.grant.ctx_per_slot == 4096
        assert r.grant.url == "http://100.112.254.99:8012"
    run(go())


def test_queue_then_release_hands_the_slot_on_and_records_wait_outside_transport():
    async def go():
        rt, clock = make()
        await boot(rt)
        first = await rt.acquire(acq("chat", priority="interactive"))
        second = await rt.acquire(acq("chat", priority="interactive"))
        assert first.status == "granted" and second.status == "queued" and second.position == 1
        clock.advance(4)
        await rt.release(first.lease_id, "ok")
        row = await rt.store.lease(second.lease_id)
        assert row["status"] == "granted"
        [g] = [e for e in rt.bus.events("granted") if e["lease_id"] == second.lease_id]
        assert g["waited_ms"] == 4000 and g["detail"]["grant"]["role"] == "chat"
        assert ("ok", "gpu_pool:chat#gpu_pool_wait", 4000) in rt.bus.hops
        assert all(g["atom"]["layer"] == "capacity" for g in rt.bus.grammar())
    run(go())


def test_acquire_is_idempotent_on_request_id():
    async def go():
        rt, _ = make()
        await boot(rt)
        a = await rt.acquire(acq("fast", rid="same"))
        b = await rt.acquire(acq("fast", rid="same"))
        assert a.lease_id == b.lease_id and len(rt.store.leases) == 1
    run(go())


def test_failures_retry_then_dead_letter_then_operator_replay():
    async def go():
        rt, clock = make()
        await boot(rt)
        r = await rt.acquire(acq_r("metacog"))
        for attempt in range(CFG.defaults.retry.max_attempts):
            row = await rt.store.lease(r.lease_id)
            assert row["status"] == "granted", (attempt, row["status"])
            await rt.release(r.lease_id, "upstream_error", "HTTP 500")
            await later(rt, clock, CFG.defaults.retry.max_sec + 1)
        row = await rt.store.lease(r.lease_id)
        assert row["status"] == "dead_letter" and rt.bus.events("dead_lettered")
        bad = await rt.control(GpuPoolControlV1(verb="replay", operator_token="nope", lease_id=r.lease_id))
        assert not bad.ok and bad.reason == "operator_token_rejected"
        ok = await rt.control(GpuPoolControlV1(verb="replay", operator_token=TOKEN, lease_id=r.lease_id))
        assert ok.ok and (await rt.store.lease(r.lease_id))["status"] == "granted"
        path = [h["event"] for h in await rt.history(r.lease_id)]
        assert path[:3] == ["admit", "grant", "release_failed"] and "replay" in path
    run(go())


def test_lost_heartbeat_expires_and_retries():
    async def go():
        rt, clock = make()
        await boot(rt)
        r = await rt.acquire(acq_r("agent"))
        clock.advance(CFG.defaults.request_lease_ttl_sec - 1)
        assert (await rt.heartbeat(r.lease_id)).status == "granted"
        clock.advance(CFG.defaults.request_lease_ttl_sec + 1)
        await rt.tick()
        assert (await rt.store.lease(r.lease_id))["status"] == "retry_wait"
        assert rt.bus.events("expired")
    run(go())


def test_gpu0_lend_lets_agent_borrow_and_chat_recalls_it():
    async def go():
        rt, clock = make()
        await boot(rt)
        await rt.acquire(acq("agent"))                     # takes gpu1's only slot
        waiting = await rt.acquire(acq("agent"))
        assert waiting.status == "queued"
        await rt.control(GpuPoolControlV1(verb="lend", operator_token=TOKEN, card="gpu0"))
        borrowed = await rt.store.lease(waiting.lease_id)
        assert borrowed["status"] == "granted" and borrowed["role"] == "chat"
        chat = await rt.acquire(acq("chat", priority="interactive"))
        assert chat.status == "queued"
        assert (await rt.store.lease(waiting.lease_id))["status"] == "recalling"
        clock.advance(CFG.defaults.clawback_grace_sec)
        await rt.tick()
        assert (await rt.store.lease(chat.lease_id))["status"] == "granted"
        assert rt.bus.events("aborted")
    run(go())


def test_backlog_replays_when_the_role_returns():
    async def go():
        store, saver, clock = MemoryStore(), MemorySaver(), Clock()
        rt, _ = make(down=("world",), store=store, saver=saver, clock=clock)
        await boot(rt)
        r = await rt.acquire(acq_r("world"))
        assert r.status == "backlogged"
        rt2, _ = make(store=store, saver=saver, clock=clock)   # restart, world is back
        await boot(rt2)
        assert (await store.lease(r.lease_id))["status"] == "granted"
    run(go())


def test_restart_keeps_granted_leases_and_heartbeats_work():
    async def go():
        store, saver, clock = MemoryStore(), MemorySaver(), Clock()
        rt, _ = make(store=store, saver=saver, clock=clock)
        await boot(rt)
        r = await rt.acquire(acq("metacog"))
        rt2, _ = make(store=store, saver=saver, clock=clock)
        await boot(rt2)
        assert (await rt2.heartbeat(r.lease_id)).status == "granted"
        assert (await rt2.release(r.lease_id, "ok")).status == "ok"
    run(go())


def test_operator_only_class_refused_over_lease_rpc():
    async def go():
        rt, _ = make()
        await boot(rt)
        r = await rt.acquire(acq("experiment"))
        assert r.status == "unavailable" and r.reason == "operator_only_class"
    run(go())


def test_mismatched_worker_gets_no_grants_and_is_reported():
    async def go():
        rt, _ = make()
        await boot(rt)
        await rt.on_announce(LlmWorkerAnnounceV1(host="circe", role="metacog", profile_name=LIVE["fast"][0],
                                                 port=8012, announced_at=rt.now()))
        await rt.tick()
        assert {d.role: d.status for d in rt.discovered}["metacog"] == "mismatch"
        assert (await rt.acquire(acq("metacog"))).grant.role == "fast"
        assert rt.bus.events("discovery_mismatch")
    run(go())


def test_observe_mode_publishes_swap_requests_without_touching_cards():
    async def go():
        rt, clock = make()
        await boot(rt)
        await rt.acquire(acq("agent", kind="hold"))   # 90s heartbeat window: stays held
        await rt.acquire(acq("agent"))
        await later(rt, clock, CFG.defaults.swap_after_wait_sec + 1)
        await rt.tick()
        [s] = rt.bus.events("swap_requested")
        assert s["role"] == "agent-gpu2" and s["detail"] == {"action": "load", "actuated": False, "mode": "observe"}
        assert not rt.cards["gpu2"].swapped_in
    run(go())


def test_backfill_preview_then_linked_children():
    async def go():
        rt, _ = make()
        await boot(rt)
        for _ in range(3):
            r = await rt.acquire(acq("fast"))
            await rt.release(r.lease_id, "ok")
        spec = {"work_class": "fast", "status": "released"}
        prev = await rt.control(GpuPoolControlV1(verb="backfill", operator_token=TOKEN, backfill=spec))
        assert prev.detail == {"would_replay": 3}
        done = await rt.control(GpuPoolControlV1(verb="backfill", operator_token=TOKEN,
                                                 backfill={**spec, "preview": False}))
        children = done.detail["children"]
        assert len(children) == 3
        assert all([(await rt.store.lease(c))["parent_lease_id"] for c in children])
    run(go())


def test_state_snapshot_shows_cards_roles_and_queue():
    async def go():
        rt, _ = make()
        await boot(rt)
        await rt.acquire(acq("chat", priority="interactive"))
        await rt.acquire(acq("chat", priority="interactive"))
        state = await rt.snapshot()
        assert {c.card for c in state.cards} == set(CFG.cards)
        assert state.queue_depth == {"chat": 1} and state.mode == "observe"
        assert state.config_digest == CFG.digest
    run(go())


def test_non_retryable_failure_ends_and_is_never_regranted():
    async def go():
        rt, clock = make()
        await boot(rt)
        r = await rt.acquire(acq("metacog"))
        await rt.release(r.lease_id, "upstream_error", "HTTP 500")
        row = await rt.store.lease(r.lease_id)
        assert row["status"] == "released" and "HTTP 500" in row["reason"]
        await later(rt, clock, 400)
        assert (await rt.store.lease(r.lease_id))["status"] == "released"
    run(go())


def test_release_ok_after_abort_ends_the_lease():
    async def go():
        rt, clock = make()
        await boot(rt)
        await rt.acquire(acq("agent"))
        waiting = await rt.acquire(acq_r("agent"))
        await rt.control(GpuPoolControlV1(verb="lend", operator_token=TOKEN, card="gpu0"))
        await rt.acquire(acq("chat", priority="interactive"))
        await later(rt, clock, CFG.defaults.clawback_grace_sec)
        assert (await rt.store.lease(waiting.lease_id))["status"] == "retry_wait"
        assert (await rt.release(waiting.lease_id, "ok")).status == "ok"
        await later(rt, clock, 400)
        assert (await rt.store.lease(waiting.lease_id))["status"] == "released"
    run(go())


def test_expiry_to_dead_letter_reports_both_facts():
    async def go():
        rt, clock = make()
        await boot(rt)
        r = await rt.acquire(acq_r("metacog"))
        for _ in range(CFG.defaults.retry.max_attempts):
            await later(rt, clock, CFG.defaults.request_lease_ttl_sec + 1)   # heartbeat lost
            await later(rt, clock, CFG.defaults.retry.max_sec + 1)           # retry due, regranted
        assert (await rt.store.lease(r.lease_id))["status"] == "dead_letter"
        assert rt.bus.events("expired") and rt.bus.events("dead_lettered")
    run(go())


def test_swap_request_is_reported_again_when_it_recurs():
    async def go():
        rt, clock = make()
        await boot(rt)
        held = await rt.acquire(acq("agent", kind="hold"))
        await rt.acquire(acq("agent", kind="hold", deadline_at=rt.now() + timedelta(seconds=40)))
        await later(rt, clock, CFG.defaults.swap_after_wait_sec + 1)
        assert len(rt.bus.events("swap_requested")) == 1
        await later(rt, clock, 20)             # the waiter hits its deadline: demand gone
        await rt.release(held.lease_id, "ok")
        await rt.acquire(acq("agent", kind="hold"))
        await rt.acquire(acq("agent", kind="hold"))
        await later(rt, clock, CFG.defaults.swap_after_wait_sec + 1)
        assert len(rt.bus.events("swap_requested")) == 2
    run(go())


def test_removed_class_in_live_rows_does_not_kill_the_tick():
    async def go():
        rt, _ = make()
        await boot(rt)
        r = await rt.acquire(acq("fast"))
        rt.store.leases[r.lease_id]["work_class"] = "retired_class"
        await rt.tick()
        assert (await rt.store.lease(r.lease_id))["status"] == "released"
        assert (await rt.acquire(acq("fast"))).status == "granted"
    run(go())


def test_projection_heals_from_checkpoint():
    async def go():
        rt, _ = make()
        await boot(rt)
        r = await rt.acquire(acq("fast"))
        rt.store.leases[r.lease_id]["status"] = "queued"      # a crash before the row upsert
        rt.store.leases[r.lease_id]["role"] = None
        await rt.tick()                                       # Grant rejected by the graph -> heal
        row = await rt.store.lease(r.lease_id)
        assert row["status"] == "granted" and row["role"] == "fast"
    run(go())


def test_operator_hold_via_control_and_release():
    async def go():
        rt, _ = make()
        await boot(rt)
        bad = await rt.control(GpuPoolControlV1(verb="hold", operator_token="x", work_class="experiment"))
        assert not bad.ok
        held = await rt.control(GpuPoolControlV1(verb="hold", operator_token=TOKEN, work_class="experiment"))
        assert held.ok and held.detail["status"] == "queued"      # it drains every card first
        rel = await rt.control(GpuPoolControlV1(verb="release", operator_token=TOKEN,
                                                lease_id=held.detail["lease_id"]))
        assert rel.ok
    run(go())
