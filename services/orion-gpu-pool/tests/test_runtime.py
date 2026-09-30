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


def make(down=(), store=None, saver=None, clock=None, bus=None, mode="enforce"):
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
                     prober=prober, now=clock, probe_interval_sec=0, mode=mode)
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


CLEAR_GUARDS = {"thermal": None}
SEAT_WAIT = CFG.swap_after_wait_sec("agent-gpu2")


async def beating(rt, clock, sec, lease_ids, step=60):
    """Advance ``sec`` like production: workers re-announce, holders heartbeat inside their TTL."""
    left = sec
    while left > 0:
        dt = min(step, left)
        clock.advance(dt)
        left -= dt
        for lid in lease_ids:
            await rt.heartbeat(lid)
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
        roles = {g["atom"]["semantic_role"] for g in rt.bus.grammar()}
        assert "gpu_lease_granted" not in roles        # routine grants stay out of grammar
    run(go())


TURN = "0f1e2d3c-4b5a-4968-8776-655443322110"


def test_pool_events_travel_on_their_own_correlation_id_not_the_turns():
    """Spec "Transport-metric and reader impacts" item 1: bus-mirror chains every shared envelope
    correlation_id into CAUSALLY_FOLLOWED_BY edges, so a pool event on the turn's id put the pool
    inside the turn's causal chain. Envelope id = the event's own id; the turn rides in the payload."""
    import uuid

    async def go():
        rt, clock = make()
        await boot(rt)
        first = await rt.acquire(acq("chat", priority="interactive", turn_correlation_id=TURN))
        second = await rt.acquire(acq("chat", priority="interactive", turn_correlation_id=TURN,
                                      deadline_at=rt.now() + timedelta(seconds=2)))
        assert first.status == "granted" and second.status == "queued"
        await later(rt, clock, 5)                       # second's deadline passes: an "unavailable" exception
        pool = [e for c, e in rt.bus.published if c == "orion:gpu_pool:event"
                and e.payload.get("turn_correlation_id") == TURN]
        assert {e.payload["event"] for e in pool} >= {"admitted", "granted", "unavailable"}
        for env in pool:
            assert str(env.correlation_id) != TURN
            assert env.correlation_id == uuid.UUID(hex=env.payload["event_id"])
        assert len({e.correlation_id for e in pool}) == len(pool)   # fresh per event, no shared chain
        grammar = [e for c, e in rt.bus.published if c == "orion:grammar:event"
                   and e.payload.get("correlation_id") == TURN]
        assert grammar, "the unavailable exception must still reach grammar with the turn in its payload"
        for env in grammar:
            assert str(env.correlation_id) != TURN
            assert env.correlation_id == uuid.UUID(hex=env.payload["provenance"]["source_event_id"])
    run(go())


def test_event_envelope_correlation_falls_back_to_fresh_uuid_for_a_non_hex_event_id():
    from app.runtime import event_envelope_correlation
    from orion.schemas.gpu_pool import GpuPoolEventV1

    ev = GpuPoolEventV1(event="granted", event_id="not-a-uuid", turn_correlation_id=TURN)
    a, b = event_envelope_correlation(ev), event_envelope_correlation(ev)
    assert str(a) != TURN and a != b


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
        ok = await rt.control(GpuPoolControlV1(verb="replay", lease_id=r.lease_id))
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
        await rt.control(GpuPoolControlV1(verb="lend", card="gpu0"))
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
        rt, _ = make(down=("diffusion",), store=store, saver=saver, clock=clock)   # a backlog class
        await boot(rt)
        r = await rt.acquire(acq_r("diffusion"))
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


def test_paused_pool_publishes_swap_requests_without_touching_cards():
    async def go():
        rt, clock = make()
        await boot(rt)
        rt.guard_states = dict(CLEAR_GUARDS)
        await rt.control(GpuPoolControlV1(verb="pause_actuation"))
        h = await rt.acquire(acq("agent", kind="hold"))
        await rt.acquire(acq("agent"))
        await beating(rt, clock, SEAT_WAIT + 1, [h.lease_id])
        [s] = rt.bus.events("swap_requested")
        assert s["role"] == "agent-gpu2" and s["reason"] == "actuation_paused"
        assert s["detail"] == {"action": "load", "actuated": False, "mode": "enforce", "paused": True, "wanted": "demand"}
        assert not rt.cards["gpu2"].swapped_in and rt.cards["gpu2"].swap_state == "idle"
    run(go())


def test_backfill_preview_then_linked_children():
    async def go():
        rt, _ = make()
        await boot(rt)
        for _ in range(3):
            r = await rt.acquire(acq("fast"))
            await rt.release(r.lease_id, "ok")
        spec = {"work_class": "fast", "status": "released"}
        prev = await rt.control(GpuPoolControlV1(verb="backfill", backfill=spec))
        assert prev.detail == {"would_replay": 3}
        done = await rt.control(GpuPoolControlV1(verb="backfill",
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
        assert state.queue_depth == {"chat": 1} and state.mode == "enforce" and state.actuation_paused is None
        assert state.config_digest == CFG.digest
        # Stage 5.5: Hub's biometrics labels join nvidia-smi on the card index, for the pool's host.
        assert state.host == CFG.host.name
        assert {c.card: c.index for c in state.cards} == {c: spec.index for c, spec in CFG.cards.items()}
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
        await rt.control(GpuPoolControlV1(verb="lend", card="gpu0"))
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
        rt.guard_states = dict(CLEAR_GUARDS)
        await rt.control(GpuPoolControlV1(verb="pause_actuation"))   # reported, never sent
        held = await rt.acquire(acq("agent", kind="hold"))
        await rt.acquire(acq("agent", deadline_at=rt.now() + timedelta(seconds=SEAT_WAIT + 40)))
        await beating(rt, clock, SEAT_WAIT + 1, [held.lease_id])
        assert len(rt.bus.events("swap_requested")) == 1
        await beating(rt, clock, 60, [held.lease_id])   # the waiter hits its deadline: demand gone
        await rt.release(held.lease_id, "ok")
        h2 = await rt.acquire(acq("agent", kind="hold"))
        await rt.acquire(acq("agent"))
        await beating(rt, clock, SEAT_WAIT + 1, [h2.lease_id])
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


def test_operator_hold_on_a_seat_nothing_can_load_is_refused_in_every_mode():
    """Stage 5.7: experiment has no launch block. Its hold would drain every card and load nothing."""
    async def go():
        for mode in ("enforce", "observe"):
            rt, _ = make(mode=mode)
            await boot(rt)
            refused = await rt.control(GpuPoolControlV1(verb="hold", work_class="experiment"))
            assert not refused.ok and refused.reason == "not_actuatable:experiment"
            assert not rt.store.leases                             # nothing admitted, nothing drains
            direct = await rt.acquire(acq("experiment", kind="hold"), operator=True)
            assert direct.status == "unavailable" and direct.reason == "not_actuatable:experiment"
    run(go())


def test_state_request_can_carry_config_and_a_lease_history():
    async def go():
        rt, _ = make()
        await boot(rt)
        r = await rt.acquire(acq("fast"))
        await rt.release(r.lease_id, "ok")
        plain = await rt.snapshot()
        assert plain.config is None and plain.history is None      # broadcasts stay small
        full = await rt.snapshot(include_config=True, history_for=r.lease_id)
        assert set(full.config["roles"]) == set(CFG.roles) and "cards:" in full.config_yaml
        assert full.config["routes"]["metacog"]["class"] == "metacog"
        assert [h["event"] for h in full.history] == ["admit", "grant", "release_ok"]
    run(go())


def test_grant_served_by_keeps_the_node_worker_shape():
    async def go():
        rt, _ = make()
        await boot(rt)
        r = await rt.acquire(acq("metacog"))
        assert r.grant.served_by == "circe-worker-metacog"
        assert r.grant.served_by.split("-worker")[0] == "circe"      # cortex-exec node attribution
    run(go())


def test_observe_mode_counts_a_swap_seat_loaded_when_its_worker_is_really_up():
    """observe (the stage 5.7 rollback) keeps the liveness shortcut; enforce asks the actuator
    instead (tests/test_stage5_7_enforce.py)."""
    async def go():
        rt, clock = make(mode="observe")
        LIVE["agent-gpu2"] = ("qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex", "Qwen3.8-27B-UD-Q4_K_XL.gguf", 1, 131072)
        try:
            await boot(rt)
            assert {d.role: d.status for d in rt.discovered}["agent-gpu2"] == "confirmed"
            assert "agent-gpu2" in rt.cards["gpu2"].swapped_in
            assert {d.role: d.status for d in rt.discovered}["diffusion"] == "evicted"
        finally:
            del LIVE["agent-gpu2"]
        await later(rt, clock, 1)                    # worker gone -> seat unloaded, diffusion back
        assert "agent-gpu2" not in rt.cards["gpu2"].swapped_in
    run(go())


def test_a_briefly_down_role_keeps_its_context_so_big_prompts_wait_for_it():
    """agent (131072/slot) restarts while gpu0 is lent: the class's only other role is chat
    (65536/slot). A 100k prompt must wait for agent, not be refused as bigger than the class."""
    async def go():
        down: set[str] = set()
        rt, clock = make(down=down)
        await boot(rt)
        await rt.control(GpuPoolControlV1(verb="lend", card="gpu0"))
        down.add("agent")
        await later(rt, clock, 1)
        assert not rt.roles["agent"].healthy and rt._ctx_seen["agent"] == 131072
        r = await rt.acquire(acq_r("agent", min_ctx_tokens=100_000))
        assert r.status == "backlogged", r.reason
    run(go())


def test_urgent_preempted_hold_requeues_in_place_and_says_why():
    """U2: a background durable-run hold paused for urgent work goes back in line in its original
    place (same created_at), without spending an attempt, and its heartbeat says why it is queued."""
    async def go():
        rt, clock = make()
        await boot(rt)
        bg = await rt.acquire(acq_r("agent", kind="hold", priority="background"))
        assert bg.status == "granted" and bg.grant.role == "agent"
        before = await rt.store.lease(bg.lease_id)
        u = await rt.acquire(acq_r("agent", kind="hold", priority="urgent"))
        assert u.status == "queued"
        row = await rt.store.lease(bg.lease_id)
        assert row["status"] == "recalling" and row["reason"] == "urgent_preempt"
        assert rt._view(row).reason == "urgent_preempt"            # the scheduler's dedupe sees it
        for reply in (await rt.heartbeat(bg.lease_id), await rt.status(bg.lease_id)):
            # The holder can tell an urgent pause (wait for the in-place re-queue) from other recalls.
            assert reply.status == "recall" and reply.reason == "urgent_preempt" and reply.recall_by
        clock.advance(CFG.defaults.urgent_preempt_grace_sec)
        await rt.tick()
        row = await rt.store.lease(bg.lease_id)
        assert row["status"] == "queued" and row["reason"] == "urgent_preempt"
        assert row["attempt"] == before["attempt"] and row["created_at"] == before["created_at"]
        assert row["role"] is None and row["not_before"] is None
        [ab] = [e for e in rt.bus.events("aborted") if e["lease_id"] == bg.lease_id]
        assert ab["reason"] == "urgent_preempt" and ab["attempt"] == before["attempt"]
        assert not [e for e in rt.bus.events("retried") if e["lease_id"] == bg.lease_id]
        await later(rt, clock, 1)                                   # the next pass seats the urgent hold
        assert (await rt.store.lease(u.lease_id))["status"] == "granted"
        assert (await rt.store.lease(bg.lease_id))["status"] == "queued"
        hb = await rt.heartbeat(bg.lease_id)
        assert hb.status == "queued" and hb.reason == "urgent_preempt" and hb.lease_id == bg.lease_id
        st = await rt.status(bg.lease_id)
        assert st.status == "queued" and st.reason == "urgent_preempt"
    run(go())


def test_a_slow_lock_holder_is_named_and_counted(caplog):
    """Every verb and the tick share one lock; a slow holder must show up as evidence (the log
    line names the op and its phases, /v1/lock-stats counts it) instead of as unexplained RPC lag."""
    import logging

    from app import runtime as runtime_mod

    async def go():
        rt, _ = make()
        await boot(rt)
        rt.lock_stats.drain()
        real = rt.store.live_leases

        async def slow_live_leases():
            await asyncio.sleep(0.3)
            return await real()

        rt.store.live_leases = slow_live_leases
        with caplog.at_level(logging.WARNING, logger=runtime_mod.logger.name):
            waiting = asyncio.create_task(rt.acquire(acq("fast")))
            await asyncio.sleep(0.01)
            await rt.tick()                                  # queued behind the slow acquire
            await waiting
        stats = rt.lock_stats.drain()
        assert stats["acquire"]["max_hold_ms"] >= 250 and stats["acquire"]["slow"] == 1
        assert stats["tick"]["max_wait_ms"] >= 200
        slow = [r.getMessage() for r in caplog.records if "gpu_pool_slow_lock" in r.getMessage()]
        assert any("op=acquire" in m and "live_leases" in m for m in slow), slow
        assert rt.lock_stats.drain() == {}                   # drained
    run(go())


def test_serialize_with_reports_queued_reason_once_then_grants_after_release():
    """Stage 5 Z1: world waits while diffusion computes on gpu2 and the pool says why, once."""
    async def go():
        rt, clock = make()
        await boot(rt)
        d = await rt.acquire(acq("diffusion"))
        assert d.status == "granted"
        w = await rt.acquire(acq("world"))
        assert w.status == "queued"
        await later(rt, clock, 1)
        await later(rt, clock, 1)
        why = [e for e in rt.bus.events("queued") if e["reason"] == "serialized:diffusion"]
        assert len(why) == 1                                   # edge-triggered, not every tick
        assert why[0]["lease_id"] == w.lease_id and why[0]["role"] == "world"
        assert why[0]["cards"] == ["gpu2"] and why[0]["detail"] == {"serialized": True}
        assert (await store_status(rt, w.lease_id)) == "queued"   # a report, not a transition
        await rt.release(d.lease_id, "ok")
        await later(rt, clock, 1)
        assert (await store_status(rt, w.lease_id)) == "granted"
    run(go())


async def store_status(rt, lease_id):
    return (await rt.store.lease(lease_id))["status"]
