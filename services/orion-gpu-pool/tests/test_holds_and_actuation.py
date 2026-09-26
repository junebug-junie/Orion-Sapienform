"""Stage 4.3 end to end through the runtime: holds, attach, shared gaps, hold recall/expiry/resume,
and the swap actuation engine driven by a fake actuator on the bus. Real scheduler, real lease
graph (MemorySaver), in-memory store. Spec:
docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.runtime import STATUS_POLL_SEC, STATUS_REPLY_SEC, PoolRuntime, validate_actuate_roles
from app.store import MemoryStore
from orion.gpu_pool.config import launch_digest
from orion.gpu_pool.discovery import Probe
from orion.gpu_pool.lease_graph import build_lease_graph
from orion.schemas.gpu_pool import GpuActuateResultV1, GpuActuateV1, GpuLeaseRequestV1, GpuPoolControlV1, \
    LlmWorkerAnnounceV1
from tests.test_runtime import CFG, CLEAR_GUARDS, LIVE, PROFILES, SEAT_WAIT, Clock, FakeBus, acq, run

ACTUATE = "orion:gpu_pool:actuate:request"
SEAT = "agent-gpu2"
LIVE2 = {**LIVE, SEAT: ("qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex", "Qwen3.8-27B-UD-Q4_K_XL.gguf", 1, 131072)}


class World:
    """Which workers are really up (what the probe and announcements see)."""

    def __init__(self):
        self.up = set(LIVE) | {"world", "diffusion"}


def make(actuate=(SEAT,), store=None, saver=None, clock=None, world=None, bus=None, guards=CLEAR_GUARDS):
    clock = clock or Clock()
    world = world or World()

    async def prober(role, url, kind, health):
        if role not in world.up:
            return Probe(False, error="refused", checked_at=clock())
        if kind == "service":
            return Probe(True, checked_at=clock())
        _, file, slots, ctx = LIVE2[role]
        return Probe(True, {"model_path": f"/models/gguf/{file}", "total_slots": slots,
                            "default_generation_settings": {"n_ctx": ctx}, "modalities": {"vision": False}},
                     checked_at=clock())

    rt = PoolRuntime(cfg=CFG, profiles=PROFILES, store=store or MemoryStore(),
                     graph=build_lease_graph(lambda: CFG, saver or MemorySaver()), bus=bus or FakeBus(),
                     prober=prober, now=clock, probe_interval_sec=0, actuate_roles=actuate)
    rt.guard_states = dict(guards)
    rt._world = world
    return rt, clock


async def announce(rt):
    for role, (profile, _, _, _) in LIVE2.items():
        if role in rt._world.up:
            await rt.on_announce(LlmWorkerAnnounceV1(host="circe", role=role, profile_name=profile,
                                                     port=CFG.roles[role].port, announced_at=rt.now()))


async def boot(rt):
    await rt.start()
    await announce(rt)
    await rt.tick()
    return rt


async def step(rt, clock, sec, beat=(), every=30):
    left = sec
    while left > 0:
        dt = min(every, left)
        clock.advance(dt)
        left -= dt
        for lid in beat:
            await rt.heartbeat(lid)
        await announce(rt)
        await rt.tick()


def hold(rid=None, **kw):
    return GpuLeaseRequestV1(verb="acquire", work_class="agent", holder=f"durable-runs:{rid or 'r'}",
                             request_id=rid, kind="hold", priority="background", retryable=True, **kw)


def attach(h, rid, generation=None, **kw):
    return GpuLeaseRequestV1(verb="attach", work_class="agent", holder="orion-llm-gateway", request_id=rid,
                             hold_lease_id=h.lease_id, hold_generation=generation or h.grant.generation, **kw)


def actuations(rt):
    return [GpuActuateV1.model_validate(e.payload) for c, e in rt.bus.published if c == ACTUATE]


async def result(rt, msg, status, **kw):
    await rt.on_actuate_result(GpuActuateResultV1(action_id=msg.action_id, generation=msg.generation,
                                                  role=msg.role, action=msg.action, status=status, **kw))


# --- holds ------------------------------------------------------------------------------------
def test_hold_is_granted_with_the_hold_ttl_and_status_reads_it():
    async def go():
        rt, clock = make(actuate=())
        await boot(rt)
        h = await rt.acquire(hold("run1:1"))
        assert h.status == "granted" and h.grant.role == "agent"
        row = await rt.store.lease(h.lease_id)
        assert row["kind"] == "hold" and (row["expires_at"] - clock()).total_seconds() == CFG.defaults.hold_lease_ttl_sec
        s = await rt.status(h.lease_id)
        assert s.status == "granted" and s.grant.generation == h.grant.generation
        assert (await rt.status("nope")).status == "unknown_lease"
        assert (await rt.store.lease(h.lease_id))["updated_at"] == row["updated_at"]   # status wrote nothing
    run(go())


def test_attach_runs_in_the_holds_slot_and_never_takes_a_second():
    async def go():
        rt, _ = make(actuate=())
        await boot(rt)
        h = await rt.acquire(hold("run1:1"))
        c = await rt.attach(attach(h, "call-1", turn_correlation_id="t1"))
        assert c.status == "granted" and c.grant.role == "agent" and c.lease_id != h.lease_id
        row = await rt.store.lease(c.lease_id)
        assert row["hold_lease_id"] == h.lease_id and row["parent_lease_id"] is None and row["kind"] == "request"
        assert (await rt.attach(attach(h, "call-1"))).lease_id == c.lease_id    # idempotent on request_id
        # the one agent slot is taken by the pair: an interactive call waits one inference
        other = await rt.acquire(acq("agent", priority="interactive"))
        assert other.status == "queued"
        await rt.release(c.lease_id, "ok")
        assert (await rt.store.lease(other.lease_id))["status"] == "granted"   # the gap is shared
        snap = await rt.snapshot()
        assert {r.lease_id: r.hold_lease_id for r in snap.leases if r.lease_id == c.lease_id} in ({}, {c.lease_id: h.lease_id})
    run(go())


def test_attach_refuses_a_stale_or_missing_hold():
    async def go():
        rt, _ = make(actuate=())
        await boot(rt)
        h = await rt.acquire(hold("run1:1"))
        stale = await rt.attach(attach(h, "c-stale", generation=h.grant.generation + 1))
        assert stale.status == "unavailable" and stale.reason.startswith("stale_hold_generation")
        req = await rt.acquire(acq("fast"))
        not_hold = await rt.attach(GpuLeaseRequestV1(verb="attach", work_class="agent", request_id="c-x",
                                                     hold_lease_id=req.lease_id, hold_generation=1))
        assert not_hold.reason == "not_a_hold"
        await rt.release(h.lease_id, "ok")
        gone = await rt.attach(attach(h, "c-gone"))
        assert gone.status == "unavailable" and gone.reason == "hold_not_granted:released"
        unknown = await rt.attach(GpuLeaseRequestV1(verb="attach", work_class="agent", request_id="c-u",
                                                    hold_lease_id="nope", hold_generation=1))
        assert unknown.status == "unknown_lease"
    run(go())


def test_gap_sharing_is_by_priority():
    async def go():
        rt, _ = make(actuate=())
        await boot(rt)
        h = await rt.acquire(hold("run1:1"))            # background hold, no call in flight
        bg = await rt.acquire(acq("agent", priority="background"))
        assert bg.status == "queued"                      # equal priority never interleaves
        sys_ = await rt.acquire(acq("agent", priority="system"))
        assert sys_.status == "granted" and sys_.grant.role == "agent"   # strictly higher uses the gap
        # the run's next call now waits exactly for that one inference, then goes before bg
        c = await rt.attach(attach(h, "call-2"))
        assert c.status == "queued"
        await rt.release(sys_.lease_id, "ok")
        assert (await rt.store.lease(c.lease_id))["status"] == "granted"
        assert (await rt.store.lease(bg.lease_id))["status"] == "queued"
    run(go())


def test_recall_gives_the_hold_grace_then_aborts_and_requeues_the_same_lease():
    async def go():
        rt, clock = make(actuate=())
        await boot(rt)
        await rt.control(GpuPoolControlV1(verb="lend", card="gpu0"))
        home = await rt.acquire(hold("run-home:1"))
        assert home.grant.role == "agent"
        borrowed = await rt.acquire(hold("run-b:1"))
        assert borrowed.grant.role == "chat"              # agent class may borrow lent gpu0
        owner = await rt.acquire(acq("chat", priority="interactive", deadline_at=clock() + timedelta(hours=1)))
        assert owner.status == "granted"                  # uses the borrowed hold's gap...
        beat = await rt.heartbeat(borrowed.lease_id)      # ...and the hold is asked to give it back
        assert beat.status == "recall"
        assert (beat.recall_by - clock()).total_seconds() == CFG.defaults.hold_clawback_grace_sec
        await step(rt, clock, CFG.defaults.hold_clawback_grace_sec - 30, beat=[home.lease_id, borrowed.lease_id])
        assert (await rt.store.lease(borrowed.lease_id))["status"] == "recalling"   # still inside the grace
        await step(rt, clock, 60, beat=[home.lease_id, borrowed.lease_id])
        names = [e["event"] for e in rt.bus.events() if e.get("lease_id") == borrowed.lease_id]
        assert names.index("aborted") < len(names) - 1 and "queued" in names[names.index("aborted"):]
        st = await rt.status(borrowed.lease_id)
        assert st.status in ("queued", "granted") and st.lease_id == borrowed.lease_id   # same lease id
    run(go())


def test_hold_expires_after_missed_heartbeats_then_is_regranted_with_a_new_generation():
    async def go():
        rt, clock = make(actuate=())
        await boot(rt)
        h = await rt.acquire(hold("run1:1"))
        await step(rt, clock, CFG.defaults.hold_lease_ttl_sec + 1)      # durable-runs is dead
        assert "expired" in [e["event"] for e in rt.bus.events() if e.get("lease_id") == h.lease_id]
        await step(rt, clock, 30)                                        # retry delay passes
        row = await rt.store.lease(h.lease_id)
        assert row["status"] == "granted" and row["generation"] == h.grant.generation + 1
        # a caller still holding the old generation may not attach
        stale = await rt.attach(attach(h, "c-old"))
        assert stale.reason.startswith("stale_hold_generation")
    run(go())


def test_restart_resumes_a_hold_by_lease_id():
    async def go():
        store, saver, clock = MemoryStore(), MemorySaver(), Clock()
        rt1, _ = make(actuate=(), store=store, saver=saver, clock=clock)
        await boot(rt1)
        h = await rt1.acquire(hold("run1:1"))
        rt2, _ = make(actuate=(), store=store, saver=saver, clock=clock)
        await boot(rt2)
        st = await rt2.status(h.lease_id)
        assert st.status == "granted" and st.grant.generation == h.grant.generation
        assert (await rt2.heartbeat(h.lease_id)).status == "granted"
        assert (await rt2.attach(attach(h, "after-restart"))).status == "granted"
    run(go())


# --- actuation engine -------------------------------------------------------------------------
async def demand_gpu2(rt, clock):
    """agent busy with one run's hold, a second run's hold waiting past the seat's after_wait."""
    home = await rt.acquire(hold("home:1"))
    waiting = await rt.acquire(hold("wait:1"))
    assert waiting.status == "queued"
    await step(rt, clock, SEAT_WAIT + 1, beat=[home.lease_id])
    return home, waiting


def test_actuation_is_off_unless_the_role_is_listed():
    async def go():
        rt, clock = make(actuate=())
        await boot(rt)
        await demand_gpu2(rt, clock)
        assert actuations(rt) == []
        [s] = rt.bus.events("swap_requested")
        assert s["detail"]["actuated"] is False and rt.cards["gpu2"].swap_state == "idle"
    run(go())


def test_actuate_roles_are_validated_at_boot():
    assert validate_actuate_roles(CFG, ["agent-gpu2", " "]) == frozenset({"agent-gpu2"})
    for bad in (["agent"], ["experiment"], ["nope"]):
        with pytest.raises(ValueError):
            validate_actuate_roles(CFG, bad)


def test_load_success_path_then_grant_on_the_seat():
    async def go():
        rt, clock = make()
        await boot(rt)
        home, waiting = await demand_gpu2(rt, clock)
        [msg] = actuations(rt)
        assert (msg.role, msg.action, msg.actuator, msg.cards, msg.generation) == (SEAT, "load", "circe", ["gpu2"], 1)
        assert msg.launch_digest == launch_digest(CFG, SEAT) and msg.profile is None
        assert rt.cards["gpu2"].swap_state == "loading" and rt.bus.events("swap_started")
        stored = (await rt.store.cards())
        assert {c["card"]: c["swap_state"] for c in stored}["gpu2"] == "loading"   # persisted before sending
        await result(rt, msg, "accepted")
        await result(rt, msg, "progress", phase="draining")
        assert rt.cards["gpu2"].swap_action["phase"] == "draining"
        # diffusion gets no grants while its card is mid-load
        d = await rt.acquire(acq("diffusion"))
        assert d.status != "granted"
        await rt.cancel(d.lease_id)
        await step(rt, clock, 30, beat=[home.lease_id])
        assert len(actuations(rt)) == 1                  # no second transition while one is running
        rt._world.up.discard("diffusion")
        rt._world.up.add(SEAT)
        await result(rt, msg, "succeeded", elapsed_ms=120000, observed={SEAT: "running", "diffusion": "exited"})
        assert SEAT in rt.cards["gpu2"].swapped_in and rt.cards["gpu2"].loaded_at == clock()
        [sw] = rt.bus.events("swapped")
        assert sw["detail"]["action_id"] == msg.action_id
        await step(rt, clock, 30, beat=[home.lease_id, waiting.lease_id])   # discovery confirms the 27B
        assert (await rt.store.lease(waiting.lease_id))["role"] == SEAT
    run(go())


def test_no_ack_is_actuator_unreachable_and_cools_down():
    async def go():
        rt, clock = make()
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        [msg] = actuations(rt)
        await step(rt, clock, CFG.defaults.actuate_ack_sec, beat=[home.lease_id], every=5)
        [f] = rt.bus.events("swap_failed")
        assert f["reason"] == "actuator_unreachable" and rt.cards["gpu2"].swap_state == "idle"
        assert rt.cards["gpu2"].cooldown_until > clock()
        await step(rt, clock, 60, beat=[home.lease_id])
        assert len(actuations(rt)) == 1                  # cooling down: no retry storm
        assert any(e["reason"] == "cooldown" for e in rt.bus.events("swap_requested"))
    run(go())


def test_failed_load_restored_cools_down_and_unrestored_faults_the_card():
    async def go():
        rt, clock = make()
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        [msg] = actuations(rt)
        await result(rt, msg, "accepted")
        await result(rt, msg, "failed", phase="ready_wait", restored=True, reason="ready_timeout",
                     observed={SEAT: "exited", "diffusion": "running"})
        [f] = rt.bus.events("swap_failed")
        assert f["detail"]["restored"] is True and rt.cards["gpu2"].swap_state == "idle"
        assert rt.cards["gpu2"].cooldown_until > clock() and SEAT not in rt.cards["gpu2"].swapped_in

        rt2, clock2 = make()
        await boot(rt2)
        home2, _ = await demand_gpu2(rt2, clock2)
        [m2] = actuations(rt2)
        await result(rt2, m2, "accepted")
        rt2._world.up.discard("diffusion")                # the rollback really did not bring it back
        await result(rt2, m2, "failed", restored=False, reason="rollback_failed")
        assert rt2.cards["gpu2"].swap_state == "fault" and rt2.cards["gpu2"].swap_role == SEAT
        w = await rt2.acquire(acq("world"))
        assert w.status != "granted"                      # no grants on any role of a faulted card
        snap = await rt2.snapshot()
        gpu2 = next(c for c in snap.cards if c.card == "gpu2")
        assert gpu2.swap_state == "fault" and gpu2.actuation["outcome"] == "failed"
        await step(rt2, clock2, 60, beat=[home2.lease_id])
        assert len(actuations(rt2)) == 1                  # no auto-retry from fault
    run(go())


def test_fault_clears_when_discovery_sees_the_residents_healthy():
    async def go():
        rt, clock = make()
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        [msg] = actuations(rt)
        await result(rt, msg, "accepted")
        rt._world.up.discard("diffusion")
        await result(rt, msg, "failed", restored=False, reason="rollback_failed")
        await step(rt, clock, 30, beat=[home.lease_id])
        assert rt.cards["gpu2"].swap_state == "fault"      # residents still down
        rt._world.up.add("diffusion")                      # an operator restarted diffusion
        await step(rt, clock, 30, beat=[home.lease_id])
        assert rt.cards["gpu2"].swap_state == "idle" and rt.cards["gpu2"].cooldown_until > clock()
        assert any(e["reason"] == "fault_cleared:discovery" for e in rt.bus.events("swapped"))
        assert len(actuations(rt)) == 1                   # backs off before trying again
    run(go())


def test_refused_action_is_reported_and_backs_off():
    async def go():
        rt, clock = make()
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        [msg] = actuations(rt)
        await result(rt, msg, "refused", reason="launch_digest_mismatch")
        [r] = rt.bus.events("actuate_refused")
        assert r["reason"] == "launch_digest_mismatch" and rt.cards["gpu2"].swap_state == "idle"
        assert rt.cards["gpu2"].cooldown_until > clock()
    run(go())


def test_overdue_result_asks_status_and_adopts_what_it_reports():
    async def go():
        rt, clock = make()
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        [msg] = actuations(rt)
        await result(rt, msg, "accepted")
        sent = rt.cards["gpu2"].swap_action["sent_at"]
        assert (msg.deadline_at - datetime.fromisoformat(sent)).total_seconds() == \
            CFG.roles[SEAT].launch.timeout_sec + CFG.roles["diffusion"].launch.timeout_sec
        timeout = (msg.deadline_at - clock()).total_seconds()
        await step(rt, clock, timeout + 1, beat=[home.lease_id], every=60)
        [_, status] = actuations(rt)
        assert status.action == "status" and status.role == SEAT
        rt._world.up.add(SEAT)
        rt._world.up.discard("diffusion")
        await result(rt, status, "succeeded", in_flight=False, observed={SEAT: "running", "diffusion": "exited"})
        assert rt.cards["gpu2"].swap_state == "idle" and SEAT in rt.cards["gpu2"].swapped_in
    run(go())


def test_unanswered_status_faults_the_card():
    async def go():
        rt, clock = make()
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        [msg] = actuations(rt)
        await result(rt, msg, "accepted")
        rt._world.up.discard("diffusion")                 # mid-transition: nothing healthy to clear to
        await step(rt, clock, (msg.deadline_at - clock()).total_seconds() + 1, beat=[home.lease_id], every=60)
        assert actuations(rt)[-1].action == "status"
        await step(rt, clock, CFG.defaults.actuate_ack_sec + 5, beat=[home.lease_id], every=5)
        assert rt.cards["gpu2"].swap_state == "loading"   # docker may take ~60s to answer status
        await step(rt, clock, STATUS_REPLY_SEC, beat=[home.lease_id], every=5)
        assert rt.cards["gpu2"].swap_state == "fault"
        assert rt.bus.events("swap_failed")[-1]["reason"] == "actuator_unreachable"
    run(go())


def test_pool_restart_mid_load_sends_status_never_a_second_transition():
    async def go():
        store, saver, clock = MemoryStore(), MemorySaver(), Clock()
        rt1, _ = make(store=store, saver=saver, clock=clock)
        await boot(rt1)
        await demand_gpu2(rt1, clock)
        [msg] = actuations(rt1)
        await result(rt1, msg, "accepted")
        bus2 = FakeBus()
        rt2, _ = make(store=store, saver=saver, clock=clock, bus=bus2)
        await rt2.start()
        sent = actuations(rt2)
        assert [(m.action, m.role) for m in sent] == [("status", SEAT)]
        assert rt2.cards["gpu2"].swap_state == "loading"
        # the real 4.2 reply while its transition runs: succeeded + phase, no in_flight field
        await result(rt2, sent[0], "succeeded", phase="starting", observed={SEAT: "running", "diffusion": "exited"})
        assert rt2.cards["gpu2"].swap_state == "loading"
        await result(rt2, msg, "succeeded", observed={SEAT: "running", "diffusion": "exited"})  # original finishes
        assert rt2.cards["gpu2"].swap_state == "idle" and SEAT in rt2.cards["gpu2"].swapped_in
        assert len(actuations(rt2)) == 1
    run(go())


def test_idle_unload_then_min_residency_blocks_reload():
    async def go():
        rt, clock = make()
        await boot(rt)
        home, waiting = await demand_gpu2(rt, clock)
        [load] = actuations(rt)
        await result(rt, load, "accepted")
        rt._world.up.add(SEAT)
        rt._world.up.discard("diffusion")
        await result(rt, load, "succeeded", observed={SEAT: "running", "diffusion": "exited"})
        await step(rt, clock, 30, beat=[home.lease_id, waiting.lease_id])
        await rt.release(waiting.lease_id, "ok")          # the run finishes: the seat goes idle
        await step(rt, clock, CFG.defaults.swap_idle_unload_sec - 5, beat=[home.lease_id])
        assert len(actuations(rt)) == 1
        await step(rt, clock, 10, beat=[home.lease_id], every=10)
        unload = actuations(rt)[-1]
        assert (unload.action, unload.reason, unload.generation) == ("unload", "idle", 2)
        assert rt.cards["gpu2"].swap_state == "unloading"
        await result(rt, unload, "accepted")
        rt._world.up.discard(SEAT)
        rt._world.up.add("diffusion")
        await result(rt, unload, "succeeded", observed={SEAT: "exited", "diffusion": "running"})
        assert SEAT not in rt.cards["gpu2"].swapped_in
        assert rt.cards["gpu2"].residency_until == clock() + timedelta(seconds=CFG.defaults.swap_min_residency_sec)
        again = await rt.acquire(hold("again:1"))
        assert again.status == "queued"
        await step(rt, clock, CFG.defaults.swap_min_residency_sec - 60, beat=[home.lease_id])
        # demand needs after_wait (1200s) > residency (600s): residency is over first
        assert len(actuations(rt)) == 2
    run(go())


def test_min_residency_is_reported_when_demand_arrives_inside_it():
    async def go():
        rt, clock = make()
        await boot(rt)
        rt.cards["gpu2"].residency_until = clock() + timedelta(seconds=SEAT_WAIT + 600)
        await demand_gpu2(rt, clock)
        assert actuations(rt) == []
        assert [e["reason"] for e in rt.bus.events("swap_requested")] == ["min_residency"]
    run(go())


def test_guards_block_an_armed_load_and_say_which():
    async def go():
        rt, clock = make(guards={"thermal": "hot:temp_over_hot", "visual_baseline": None})
        await boot(rt)
        await demand_gpu2(rt, clock)
        assert actuations(rt) == []
        [s] = rt.bus.events("swap_requested")
        assert s["reason"] == "guard:thermal" and s["detail"]["guard_state"] == "hot:temp_over_hot"
        rt.guard_states = {"thermal": None, "visual_baseline": "visual_baseline_urgent"}
        await step(rt, clock, 1)
        assert rt.bus.events("swap_requested")[-1]["reason"] == "guard:visual_baseline"
        rt.guard_states = dict(CLEAR_GUARDS)
        await step(rt, clock, 1)
        assert [m.action for m in actuations(rt)] == ["load"]
    run(go())


def test_an_unread_guard_blocks_loading():
    async def go():
        rt, clock = make(guards={"thermal": "unread", "visual_baseline": "unread"})
        await boot(rt)
        await demand_gpu2(rt, clock)
        assert actuations(rt) == [] and rt.bus.events("swap_requested")[0]["reason"] == "guard:thermal"
    run(go())


def test_armed_pool_adopts_a_seat_the_old_path_already_loaded():
    async def go():
        rt, clock = make()
        rt._world.up.add(SEAT)
        rt._world.up.discard("diffusion")
        await boot(rt)
        assert SEAT in rt.cards["gpu2"].swapped_in and rt.cards["gpu2"].loaded_at == clock()
        assert actuations(rt) == []                       # not idle-unloaded on the first tick either
        h = await rt.acquire(hold("home:1"))
        h2 = await rt.acquire(hold("second:1"))
        assert h2.grant.role == SEAT and actuations(rt) == []   # used as-is, not reloaded
    run(go())


# --- the 4.2 actuator's status semantics (PR #2350) -------------------------------------------
def test_status_reply_saying_in_flight_keeps_polling_past_the_deadline_without_fault():
    async def go():
        rt, clock = make()
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        [msg] = actuations(rt)
        await result(rt, msg, "accepted")
        rt._world.up.discard("diffusion")                 # mid-load: nothing consistent on the card
        await step(rt, clock, (msg.deadline_at - clock()).total_seconds() + 1, beat=[home.lease_id], every=60)
        for n in range(6):                                # the actuator's worst case outlasts the deadline
            status = actuations(rt)[-1]
            assert status.action == "status"
            await result(rt, status, "succeeded", in_flight=True, observed={SEAT: "running", "diffusion": "exited"})
            assert rt.cards["gpu2"].swap_state == "loading"
            await step(rt, clock, STATUS_POLL_SEC + 1, beat=[home.lease_id], every=STATUS_POLL_SEC + 1)
        assert rt.cards["gpu2"].swap_state == "loading" and not rt.bus.events("swap_failed")
        await result(rt, msg, "succeeded", observed={SEAT: "running", "diffusion": "exited"})   # finally
        assert rt.cards["gpu2"].swap_state == "idle" and SEAT in rt.cards["gpu2"].swapped_in
    run(go())


def test_status_replays_the_recorded_result_under_its_own_action_id_first():
    """4.2: `status` re-publishes the last recorded GpuActuateResultV1 (original action_id), then
    answers the status request. The replay settles the card; the status answer is then stale."""
    async def go():
        store, saver, clock = MemoryStore(), MemorySaver(), Clock()
        rt1, _ = make(store=store, saver=saver, clock=clock)
        await boot(rt1)
        await demand_gpu2(rt1, clock)
        [load] = actuations(rt1)
        rt2, _ = make(store=store, saver=saver, clock=clock, bus=FakeBus())
        rt2._world.up.add(SEAT)
        rt2._world.up.discard("diffusion")
        await rt2.start()
        [status] = actuations(rt2)
        await result(rt2, load, "succeeded", observed={SEAT: "running", "diffusion": "exited"})   # the replay
        assert rt2.cards["gpu2"].swap_state == "idle" and SEAT in rt2.cards["gpu2"].swapped_in
        await result(rt2, status, "succeeded", in_flight=False, last_action_id=load.action_id,
                     observed={SEAT: "running", "diffusion": "exited"})
        assert rt2.cards["gpu2"].swap_state == "idle" and len(rt2.bus.events("swapped")) == 1
    run(go())


def test_status_with_nothing_in_flight_adopts_the_containers_or_faults_a_half_done_card():
    async def go():
        for observed, state, loaded in (({SEAT: "exited", "diffusion": "running"}, "idle", False),
                                        ({SEAT: "running", "diffusion": "exited"}, "idle", True),
                                        ({SEAT: "exited", "diffusion": "exited"}, "fault", None)):
            rt, clock = make()
            await boot(rt)
            home, _ = await demand_gpu2(rt, clock)
            [msg] = actuations(rt)
            await result(rt, msg, "accepted")
            rt._world.up.discard("diffusion")
            await step(rt, clock, (msg.deadline_at - clock()).total_seconds() + 1, beat=[home.lease_id], every=60)
            status = actuations(rt)[-1]
            await result(rt, status, "succeeded", in_flight=False, last_action_id="older", observed=observed)
            assert rt.cards["gpu2"].swap_state == state
            if loaded is not None:
                assert (SEAT in rt.cards["gpu2"].swapped_in) is loaded
    run(go())


def test_failed_load_with_restored_none_reads_the_containers():
    async def go():
        for observed, state in (({SEAT: "absent", "diffusion": "running"}, "idle"),
                                ({SEAT: "exited", "diffusion": "exited"}, "fault")):
            rt, clock = make()
            await boot(rt)
            await demand_gpu2(rt, clock)
            [msg] = actuations(rt)
            await result(rt, msg, "accepted")
            rt._world.up.discard("diffusion")
            await result(rt, msg, "failed", restored=None, reason="drain_timeout", observed=observed)
            assert rt.cards["gpu2"].swap_state == state
            if state == "idle":
                assert rt.cards["gpu2"].cooldown_until > clock() and rt.bus.events("swap_failed")[0]["detail"]["restored"] is None
    run(go())


def test_in_flight_fields_are_status_only():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GpuActuateResultV1(action_id="a", generation=1, role=SEAT, action="load", status="succeeded", in_flight=True)
    GpuActuateResultV1(action_id="a", generation=1, role=SEAT, action="status", status="succeeded", in_flight=False,
                       last_action_id="x")


# --- review findings: the 4.2 actuator's REAL status reply (no in_flight, phase may be None) ------
async def overdue(rt, clock, home):
    [msg] = actuations(rt)
    return msg


def test_restart_before_the_ack_with_a_running_actuator_is_not_unreachable():
    async def go():
        store, saver, clock = MemoryStore(), MemorySaver(), Clock()
        rt1, _ = make(store=store, saver=saver, clock=clock)
        await boot(rt1)
        home, _ = await demand_gpu2(rt1, clock)
        [load] = actuations(rt1)                                  # no ack before the pool dies
        rt2, _ = make(store=store, saver=saver, clock=clock, bus=FakeBus())
        rt2._world.up.discard("diffusion")
        await rt2.start()
        [status] = actuations(rt2)
        await result(rt2, status, "succeeded", phase="draining", observed={SEAT: "absent", "diffusion": "running"})
        await step(rt2, clock, CFG.defaults.actuate_ack_sec + 5, beat=[home.lease_id], every=5)
        assert rt2.cards["gpu2"].swap_state == "loading" and not rt2.bus.events("swap_failed")
        await result(rt2, load, "succeeded", observed={SEAT: "running", "diffusion": "exited"})
        assert rt2.cards["gpu2"].swap_state == "idle" and SEAT in rt2.cards["gpu2"].swapped_in
    run(go())


def test_status_before_the_first_progress_is_not_taken_as_finished():
    """Accepted but no phase yet: 4.2 reports phase=None and the containers still look unloaded.
    One such answer must not settle the card; the next poll shows the phase."""
    async def go():
        rt, clock = make()
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        [msg] = actuations(rt)
        await result(rt, msg, "accepted")
        await step(rt, clock, (msg.deadline_at - clock()).total_seconds() + 1, beat=[home.lease_id], every=60)
        st = actuations(rt)[-1]
        await result(rt, st, "succeeded", phase=None, observed={SEAT: "absent", "diffusion": "running"})
        assert rt.cards["gpu2"].swap_state == "loading"
        await step(rt, clock, STATUS_POLL_SEC + 1, beat=[home.lease_id], every=STATUS_POLL_SEC + 1)
        st2 = actuations(rt)[-1]
        assert st2.action == "status" and st2.action_id != st.action_id
        await result(rt, st2, "succeeded", phase="draining", observed={SEAT: "absent", "diffusion": "exited"})
        assert rt.cards["gpu2"].swap_state == "loading"
        await result(rt, msg, "succeeded", observed={SEAT: "running", "diffusion": "exited"})
        assert rt.cards["gpu2"].swap_state == "idle" and SEAT in rt.cards["gpu2"].swapped_in
    run(go())


def test_status_without_in_flight_adopts_only_after_repeated_answers():
    async def go():
        from app.runtime import MAX_STATUS_EXTENSIONS
        rt, clock = make()
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        [msg] = actuations(rt)
        await result(rt, msg, "accepted")
        await step(rt, clock, (msg.deadline_at - clock()).total_seconds() + 1, beat=[home.lease_id], every=60)
        for _ in range(MAX_STATUS_EXTENSIONS):
            await result(rt, actuations(rt)[-1], "succeeded", observed={SEAT: "absent", "diffusion": "running"})
            assert rt.cards["gpu2"].swap_state == "loading"
            await step(rt, clock, STATUS_POLL_SEC + 1, beat=[home.lease_id], every=STATUS_POLL_SEC + 1)
        await result(rt, actuations(rt)[-1], "succeeded", observed={SEAT: "absent", "diffusion": "running"})
        assert rt.cards["gpu2"].swap_state == "idle" and SEAT not in rt.cards["gpu2"].swapped_in
    run(go())


def test_an_action_still_running_after_two_timeouts_faults_the_card():
    async def go():
        from app.runtime import MAX_ACTION_TIMEOUTS
        rt, clock = make()
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        [msg] = actuations(rt)
        await result(rt, msg, "accepted")
        rt._world.up.discard("diffusion")
        limit = MAX_ACTION_TIMEOUTS * (msg.deadline_at - datetime.fromisoformat(rt.cards["gpu2"].swap_action["sent_at"])).total_seconds()
        t = 0.0
        while rt.cards["gpu2"].swap_state == "loading" and t < limit + 120:
            last = actuations(rt)[-1]
            if last.action == "status":
                await result(rt, last, "succeeded", phase="ready_wait", observed={SEAT: "running", "diffusion": "exited"})
            await step(rt, clock, 30, beat=[home.lease_id], every=30)
            t += 30
        assert rt.cards["gpu2"].swap_state == "fault"
        assert rt.bus.events("swap_failed")[-1]["reason"] == "actuator_stuck"
        assert limit - 30 <= t <= limit + 60
    run(go())


def test_operator_clears_a_fault_discovery_cannot():
    async def go():
        rt, clock = make()
        await boot(rt)
        home, _ = await demand_gpu2(rt, clock)
        [msg] = actuations(rt)
        await result(rt, msg, "accepted")
        rt._world.up.discard("diffusion")
        await result(rt, msg, "failed", restored=False, reason="rollback_failed",
                     observed={SEAT: "exited", "diffusion": "exited"})
        await step(rt, clock, 30, beat=[home.lease_id])
        assert rt.cards["gpu2"].swap_state == "fault"          # both down: discovery cannot clear it
        assert (await rt.acquire(acq("world"))).status != "granted"
        out = await rt.control(GpuPoolControlV1(verb="clear_fault", card="gpu2", actor="juniper"))
        assert out.ok and out.reason == "reconciling"
        status = actuations(rt)[-1]
        assert status.action == "status"
        rt._world.up.add("diffusion")                          # the actuator's rollback retry worked
        await result(rt, status, "succeeded", observed={SEAT: "absent", "diffusion": "running"})
        card = rt.cards["gpu2"]
        assert card.swap_state == "idle" and SEAT not in card.swapped_in and card.cooldown_until > clock()
        await step(rt, clock, 30, beat=[home.lease_id])
        assert len([m for m in actuations(rt) if m.action == "load"]) == 1   # no reload inside the cooldown
        bad = await rt.control(GpuPoolControlV1(verb="clear_fault", card="gpu2"))
        assert not bad.ok and bad.reason == "not_faulted:idle"
    run(go())


def test_operator_clear_of_a_fault_whose_seat_left_the_yaml_settles_from_discovery():
    async def go():
        rt, clock = make()
        await boot(rt)
        rt.cards["gpu2"].swap_state, rt.cards["gpu2"].swap_role = "fault", "renamed-seat"
        await step(rt, clock, 30)
        assert rt.cards["gpu2"].swap_state == "fault"
        out = await rt.control(GpuPoolControlV1(verb="clear_fault", card="gpu2"))
        assert out.ok and out.reason == "cleared" and rt.cards["gpu2"].swap_state == "idle"
    run(go())


def test_attach_never_hands_back_another_lease_and_is_never_retryable():
    async def go():
        rt, _ = make(actuate=())
        await boot(rt)
        h = await rt.acquire(hold("run1:1"))
        clash = await rt.attach(attach(h, "run1:1"))           # the hold's own request_id
        assert clash.status == "unavailable" and clash.reason == "request_id_conflict" and clash.lease_id is None
        c = await rt.attach(attach(h, "c1", retryable=True))
        assert (await rt.store.lease(c.lease_id))["retryable"] is False
        assert (await rt.store.lease(h.lease_id))["status"] == "granted"
    run(go())


def test_min_ctx_hold_loads_gpu2_once_seen_even_after_a_pool_restart():
    """4.5 finding: a hold with min_ctx_tokens never triggered the gpu2 load (an unloaded seat has
    no live ctx) and nothing said why. Now: unknown -> swap_requested ctx_unknown; once the seat
    has been seen its context is persisted and a restarted pool loads it for such a hold."""
    async def go():
        store, saver, clock = MemoryStore(), MemorySaver(), Clock()
        rt, _ = make(store=store, saver=saver, clock=clock)
        await boot(rt)
        home = await rt.acquire(hold("home:1"))
        big = await rt.acquire(hold("big:1", min_ctx_tokens=32768))
        assert big.status == "queued"
        await step(rt, clock, SEAT_WAIT + 1, beat=[home.lease_id])
        assert actuations(rt) == []
        assert [e["reason"] for e in rt.bus.events("swap_requested")] == ["ctx_unknown"]
        # the seat is seen once (loaded by the old path), then goes away again
        rt._world.up.add(SEAT)
        rt._world.up.discard("diffusion")
        await step(rt, clock, 30, beat=[home.lease_id])
        assert rt._ctx_seen[SEAT] == 131072
        assert {c["card"]: c.get("seen_ctx") for c in await store.cards()}["gpu2"][SEAT] == 131072
        assert (await rt.store.lease(big.lease_id))["role"] == SEAT   # it ran there meanwhile
        await rt.release(big.lease_id, "ok")
        rt2, _ = make(store=store, saver=saver, clock=clock, bus=FakeBus())
        rt2._world.up.discard(SEAT)
        rt2._world.up.add("diffusion")
        await boot(rt2)
        assert rt2._ctx_seen[SEAT] == 131072 and SEAT not in rt2.cards["gpu2"].swapped_in
        big2 = await rt2.acquire(hold("big:2", min_ctx_tokens=32768))
        assert big2.status == "queued"
        await step(rt2, clock, SEAT_WAIT + 1, beat=[home.lease_id])
        [load] = actuations(rt2)
        assert (load.role, load.action) == (SEAT, "load")
    run(go())
