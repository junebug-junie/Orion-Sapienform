"""Stage 5.7: enforce is the end state. Real runtime, real scheduler, real lease graph, in-memory store,
a hand-driven actuator on a fake bus.

Pinned here:
- a seat is actuated iff it has a launch block (no GPU_POOL_ACTUATE_ROLES);
- enforce adopts a seat loaded or unloaded outside the pool by asking the actuator (one read-only
  ``status`` per seat at boot and on resume), never from liveness and never by reloading it;
- operator holds work for a seat that can actuate, and are refused -- with a named reason, and without
  draining anything -- for one that cannot (experiment);
- the one emergency stop, ``pause_actuation``: persisted, survives a restart, sends nothing, drains
  nothing, and still follows an action already in flight.
Spec: docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md (Decision 6,
5.7, "Corrections from building 5.1" item 5)."""
from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

from app.runtime import STATUS_REPLY_SEC
from app.store import MemoryStore
from orion.gpu_pool.lease_graph import build_lease_graph
from orion.gpu_pool.scheduler import SwapLoad
from orion.schemas.gpu_pool import GpuActuateResultV1, GpuPoolControlV1
from tests.test_holds_and_actuation import SEAT, actuations, boot, demand_gpu2, hold, make, result, step
from tests.test_runtime import CFG, Clock, FakeBus, acq, run


def enforce(**kw):
    return make(mode="enforce", **kw)


def statuses(rt):
    return [m for m in actuations(rt) if m.action == "status"]


async def answer_status(rt, msg, observed, **kw):
    await result(rt, msg, "succeeded", observed=observed, in_flight=kw.pop("in_flight", False), **kw)


LOADED = {SEAT: "running", "diffusion": "exited"}
UNLOADED = {SEAT: "exited", "diffusion": "running"}


# --- adoption ----------------------------------------------------------------------------------
def test_boot_adopts_a_seat_loaded_outside_the_pool_with_one_status_and_no_reload():
    """Acceptance check 9: a pool restart with the 27B loaded adopts it via `status`."""
    async def go():
        rt, clock = enforce()
        rt._world.up.add(SEAT)
        rt._world.up.discard("diffusion")
        await boot(rt)
        [st] = statuses(rt)
        assert st.role == SEAT and st.reason == "reconcile:boot" and st.cards == ["gpu2"]
        assert SEAT not in rt.cards["gpu2"].swapped_in       # enforce never adopts from liveness
        await answer_status(rt, st, LOADED)
        card = rt.cards["gpu2"]
        assert SEAT in card.swapped_in and card.swap_state == "idle"
        assert card.loaded_at == clock() and card.last_active_at == clock()   # max-hold/idle clocks start now
        [ev] = rt.bus.events("swapped")
        assert ev["reason"] == "adopted:boot" and ev["detail"]["loaded"] is True
        assert card.swap_action["action"] == "status" and card.swap_action["outcome"] == "adopted"
        await step(rt, clock, 30)
        assert [m.action for m in actuations(rt)] == ["status"]          # no load, no unload
        assert {d.role: d.status for d in rt.discovered}[SEAT] == "confirmed"
    run(go())


def test_boot_adopts_a_seat_unloaded_by_hand():
    async def go():
        store = MemoryStore()
        await store.upsert_card({"card": "gpu2", "lent": False, "swapped_in": [SEAT], "swap_state": "idle",
                                 "swap_generation": 5})
        rt, clock = enforce(store=store)
        await boot(rt)
        [st] = statuses(rt)
        assert st.generation == 5
        await answer_status(rt, st, UNLOADED)
        card = rt.cards["gpu2"]
        assert SEAT not in card.swapped_in and card.loaded_at is None
        assert card.residency_until is not None               # residents get their min residency
        assert rt.bus.events("swapped")[0]["reason"] == "adopted:boot"
    run(go())


def test_boot_reconcile_that_agrees_changes_nothing():
    """Every ordinary restart lands here: the card keeps its last load/unload record (in memory and in
    the store), so the Hub never shows the reconcile as an action "in flight"."""
    async def go():
        store = MemoryStore()
        last = {"action_id": "agent-gpu2:unload:g79:abc", "role": SEAT, "action": "unload", "generation": 79,
                "outcome": "succeeded", "reason": "idle"}
        await store.upsert_card({"card": "gpu2", "lent": False, "swapped_in": [], "swap_state": "idle",
                                 "swap_generation": 79, "swap_action": last})
        rt, clock = enforce(store=store)
        await boot(rt)
        [st] = statuses(rt)
        await answer_status(rt, st, UNLOADED)
        assert not rt.bus.events("swapped") and not rt.bus.events("swap_failed")
        assert rt.cards["gpu2"].swap_state == "idle" and not rt.cards["gpu2"].swapped_in
        assert not rt._reconciling
        assert rt.cards["gpu2"].swap_action == last
        assert {c["card"]: c for c in await store.cards()}["gpu2"]["swap_action"] == last
        gpu2 = next(c for c in (await rt.snapshot()).cards if c.card == "gpu2")
        assert gpu2.actuation == last
    run(go())


def test_a_reconcile_that_cannot_say_whether_something_runs_keeps_the_card():
    async def go():
        rt, _ = enforce()
        await boot(rt)
        [st] = statuses(rt)
        await result(rt, st, "succeeded", observed=LOADED)          # in_flight=None: an older actuator
        assert SEAT not in rt.cards["gpu2"].swapped_in and rt.cards["gpu2"].swap_state == "idle"
        assert not rt.bus.events("swapped") and not rt.bus.events("swap_failed")
    run(go())


def test_nothing_is_drained_on_a_seat_while_its_reconcile_is_open():
    """The stored belief is what the reconcile is checking; it must not drive recalls until answered."""
    async def go():
        store = MemoryStore()
        await store.upsert_card({"card": "gpu2", "lent": False, "swapped_in": [SEAT], "swap_state": "idle",
                                 "swap_generation": 5, "loaded_at": Clock()(), "last_active_at": Clock()()})
        rt, clock = enforce(store=store)
        rt._world.up.add(SEAT)
        rt._world.up.discard("diffusion")
        await boot(rt)
        [st] = statuses(rt)
        on_agent = await rt.acquire(hold("run:1"))
        on_seat = await rt.acquire(hold("run:2"))
        assert on_agent.grant.role == "agent" and on_seat.grant.role == SEAT
        await rt.acquire(acq("diffusion", kind="hold"))              # gpu2's owner wants it back
        await step(rt, clock, 30, beat=[on_agent.lease_id, on_seat.lease_id])
        assert (await rt.store.lease(on_seat.lease_id))["status"] == "granted"   # not drained yet
        await answer_status(rt, st, LOADED)                           # stored belief confirmed
        await step(rt, clock, 1, beat=[on_agent.lease_id, on_seat.lease_id])
        assert (await rt.store.lease(on_seat.lease_id))["status"] == "recalling"  # now the reclaim runs
    run(go())


def test_boot_reconcile_faults_a_half_done_card_and_an_unknown_action():
    async def go():
        rt, _ = enforce()
        await boot(rt)
        [st] = statuses(rt)
        await answer_status(rt, st, {SEAT: "exited", "diffusion": "exited"})
        assert rt.cards["gpu2"].swap_state == "fault"
        assert rt.bus.events("swap_failed")[0]["reason"] == "reconcile_ambiguous:boot"

        rt2, _ = enforce()
        await boot(rt2)
        [st2] = statuses(rt2)
        await answer_status(rt2, st2, UNLOADED, in_flight=True, phase="starting")
        assert rt2.cards["gpu2"].swap_state == "fault"
        assert rt2.bus.events("swap_failed")[0]["reason"] == "reconcile_foreign_action:boot"
    run(go())


def test_an_unanswered_reconcile_keeps_the_card_and_then_actuation_proceeds():
    async def go():
        rt, clock = enforce()
        await boot(rt)
        # a load decided while the reconcile is open is deferred, silently (decided again next tick)
        await rt._swap(SwapLoad(SEAT, "demand"))
        assert [m.action for m in actuations(rt)] == ["status"] and not rt.bus.events("swap_requested")
        await step(rt, clock, STATUS_REPLY_SEC)
        # nothing to fault (nothing was in flight): the card keeps its persisted state
        assert not rt._reconciling and rt.cards["gpu2"].swap_state == "idle"
        await demand_gpu2(rt, clock)
        assert rt.cards["gpu2"].swap_state == "loading"
        assert [m.action for m in actuations(rt)] == ["status", "load"]
    run(go())


def test_a_stale_replay_during_the_reconcile_is_ignored():
    """The actuator re-publishes its last recorded result before answering a status. That row's
    `observed` is as old as its action; only the fresh answer decides."""
    async def go():
        store = MemoryStore()
        last = {"action_id": "agent-gpu2:unload:g4:abc", "role": SEAT, "action": "unload", "generation": 4,
                "outcome": "succeeded"}
        await store.upsert_card({"card": "gpu2", "lent": False, "swapped_in": [], "swap_state": "idle",
                                 "swap_generation": 4, "swap_action": last})
        rt, _ = enforce(store=store)
        await boot(rt)
        [st] = statuses(rt)
        await rt.on_actuate_result(GpuActuateResultV1(action_id=last["action_id"], generation=4, role=SEAT,
                                                      action="load", status="succeeded", observed=LOADED))
        assert SEAT not in rt.cards["gpu2"].swapped_in and not rt.bus.events("swapped")
        await answer_status(rt, st, UNLOADED)
        assert SEAT not in rt.cards["gpu2"].swapped_in and not rt.bus.events("swapped")
    run(go())


def test_observe_mode_sends_no_reconcile():
    async def go():
        rt, _ = make(mode="observe")
        await boot(rt)
        assert actuations(rt) == []
    run(go())


# --- operator holds -----------------------------------------------------------------------------
def _launchable_experiment_cfg():
    launch = CFG.roles[SEAT].launch
    roles = {**CFG.roles, "experiment": CFG.roles["experiment"].model_copy(update={"launch": launch})}
    return CFG.model_copy(update={"roles": roles})


def test_operator_hold_is_allowed_for_a_seat_that_can_actuate_and_drains_its_cards():
    async def go():
        cfg = _launchable_experiment_cfg()
        rt, _ = enforce()
        rt.cfg, rt.graph, rt.actuated = cfg, build_lease_graph(lambda: cfg, MemorySaver()), cfg.actuated_seats()
        await boot(rt)
        assert {m.role for m in statuses(rt)} == {"agent-gpu2", "experiment"}
        rt._reconciling.clear()        # the boot reconcile is not this test's subject (it freezes the seats)
        busy = await rt.acquire(acq("chat", priority="interactive"))
        held = await rt.control(GpuPoolControlV1(verb="hold", work_class="experiment", actor="juniper"))
        assert held.ok and held.detail["status"] == "queued"
        row = await rt.store.lease(busy.lease_id)
        assert row["status"] == "recalling"                  # its cards drain for the operator seat
        paused = await rt.control(GpuPoolControlV1(verb="pause_actuation"))
        assert paused.ok
        again = await rt.control(GpuPoolControlV1(verb="hold", work_class="experiment"))
        assert not again.ok and again.reason == "actuation_paused"
        rel = await rt.control(GpuPoolControlV1(verb="release", lease_id=held.detail["lease_id"]))
        assert rel.ok
        rt.mode = "observe"
        assert (await rt.control(GpuPoolControlV1(verb="hold", work_class="experiment"))).reason \
            == "hold_refused_observe_mode"
    run(go())


def test_experiment_hold_is_refused_named_and_nothing_drains():
    async def go():
        rt, _ = enforce()
        await boot(rt)
        busy = await rt.acquire(acq("chat", priority="interactive"))
        refused = await rt.control(GpuPoolControlV1(verb="hold", work_class="experiment", actor="juniper"))
        assert not refused.ok and refused.reason == "not_actuatable:experiment"
        await rt.tick()
        assert (await rt.store.lease(busy.lease_id))["status"] == "granted"
        assert not rt.bus.events("recalled")
    run(go())


# --- the emergency stop ------------------------------------------------------------------------
def test_a_pause_row_for_a_card_no_longer_configured_does_not_pause_and_resume_clears_every_row():
    async def go():
        store = MemoryStore()
        await store.upsert_card({"card": "gpu9", "swapped_in": [], "swap_state": "idle",
                                 "actuation_paused_at": Clock()(), "actuation_paused_by": "old"})
        rt, _ = enforce(store=store)
        await boot(rt)
        assert rt.paused is None                                     # gpu9 is not in the YAML
        await rt.control(GpuPoolControlV1(verb="pause_actuation", actor="j"))
        await rt.control(GpuPoolControlV1(verb="resume_actuation", actor="j"))
        assert all(r["actuation_paused_at"] is None for r in await store.cards())   # gpu9 too
    run(go())


def test_a_pause_that_cannot_be_persisted_changes_nothing():
    async def go():
        rt, _ = enforce()
        await boot(rt)

        async def broken(*a, **kw):
            raise OSError("db down")
        rt.store.set_actuation_paused = broken
        out = await rt.control(GpuPoolControlV1(verb="pause_actuation", actor="j"))
        assert not out.ok and out.reason == "not_persisted:OSError" and rt.paused is None
        assert not rt.bus.events("actuation_paused")
    run(go())


def test_pause_is_persisted_published_and_survives_a_restart():
    async def go():
        store, saver, clock = MemoryStore(), MemorySaver(), Clock()
        rt, _ = enforce(store=store, saver=saver, clock=clock)
        await boot(rt)
        out = await rt.control(GpuPoolControlV1(verb="pause_actuation", actor="juniper"))
        assert out.ok and out.reason == "paused" and out.detail["paused"] is True
        assert (await rt.control(GpuPoolControlV1(verb="pause_actuation"))).reason == "already_paused"
        assert all(r["actuation_paused_at"] == clock() and r["actuation_paused_by"] == "juniper"
                   for r in await store.cards())
        [ev] = rt.bus.events("actuation_paused")
        assert ev["holder"] == "juniper" and ev["detail"]["actuated"] == [SEAT]
        state = (await rt.snapshot()).actuation_paused
        assert state == {"paused": True, "since": clock().isoformat(), "by": "juniper"}

        rt2, _ = enforce(store=store, saver=saver, clock=clock, bus=FakeBus())
        await boot(rt2)
        assert rt2.paused == {"since": clock(), "by": "juniper"}
        home, waiting = await demand_gpu2(rt2, clock)
        assert [m.action for m in actuations(rt2)] == ["status"]       # the boot reconcile, no load
        assert rt2.bus.events("swap_requested")[0]["reason"] == "actuation_paused"
        res = await rt2.control(GpuPoolControlV1(verb="resume_actuation", actor="juniper"))
        assert res.ok and res.reason == "resumed" and rt2.paused is None
        assert all(r["actuation_paused_at"] is None for r in await store.cards())
        assert rt2.bus.events("actuation_resumed")
        # resume asks the actuator again before anything moves
        assert [m.action for m in actuations(rt2)] == ["status", "status"]
        await answer_status(rt2, statuses(rt2)[-1], UNLOADED)
        await step(rt2, clock, 1, beat=[home.lease_id, waiting.lease_id])
        assert [m.action for m in actuations(rt2)] == ["status", "status", "load"]
    run(go())


def test_pause_never_drains_the_loaded_seat_and_still_follows_an_action_in_flight():
    async def go():
        rt, clock = enforce()
        rt._world.up.add(SEAT)
        rt._world.up.discard("diffusion")
        await boot(rt)
        await answer_status(rt, statuses(rt)[0], LOADED)
        await step(rt, clock, 1)                              # discovery confirms the adopted seat
        on_seat = await rt.acquire(hold("run:1"))
        assert on_seat.grant.role == "agent"
        second = await rt.acquire(hold("run:2"))
        assert second.grant.role == SEAT
        await rt.control(GpuPoolControlV1(verb="pause_actuation"))
        # the owner wants gpu2 back: while paused the 27B keeps its hold, nothing drains or unloads
        img = await rt.acquire(acq("diffusion", kind="hold"))
        await step(rt, clock, 60, beat=[on_seat.lease_id, second.lease_id, img.lease_id])
        assert (await rt.store.lease(second.lease_id))["status"] == "granted"
        assert [m.action for m in actuations(rt)] == ["status"]
        assert [e["reason"] for e in rt.bus.events("recalled") if e.get("lease_id") == second.lease_id] == []
        # an action already in flight when the pause lands is followed to its end
        await rt.control(GpuPoolControlV1(verb="resume_actuation"))
        await answer_status(rt, statuses(rt)[-1], LOADED)
        await step(rt, clock, 1, beat=[on_seat.lease_id, second.lease_id, img.lease_id])
        assert any(e["lease_id"] == second.lease_id for e in rt.bus.events("recalled"))   # drains again
        await rt.release(second.lease_id, "ok")
        await step(rt, clock, 1, beat=[on_seat.lease_id, img.lease_id])
        [unload] = [m for m in actuations(rt) if m.action == "unload"]
        await rt.control(GpuPoolControlV1(verb="pause_actuation"))
        await result(rt, unload, "accepted")
        await result(rt, unload, "succeeded", observed=UNLOADED)
        assert SEAT not in rt.cards["gpu2"].swapped_in and rt.cards["gpu2"].swap_state == "idle"
    run(go())


def test_the_deleted_key_stays_deleted_and_enforce_is_the_default():
    """Kill means kill: no surface reads GPU_POOL_ACTUATE_ROLES, and every default says enforce."""
    from app.settings import Settings

    svc = Path(__file__).resolve().parents[1]
    for name in (".env_example", "docker-compose.yml", "app/settings.py", "app/main.py", "app/runtime.py"):
        text = (svc / name).read_text()
        assert "ACTUATE_ROLES=" not in text and "actuate_roles" not in text, name
    assert "GPU_POOL_MODE=enforce" in (svc / ".env_example").read_text()
    assert "GPU_POOL_MODE=${GPU_POOL_MODE:-enforce}" in (svc / "docker-compose.yml").read_text()
    assert Settings(_env_file=None, ORION_BUS_URL="redis://x", POSTGRES_URI="postgresql://x").mode == "enforce"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ORION_BUS_URL="redis://x", POSTGRES_URI="postgresql://x", GPU_POOL_MODE="enforced")


def test_the_shell_emergency_stop_script_round_trips_through_the_real_control_path():
    """scripts/gpu_pool_pause.py: the envelope it sends is what app.main._on_control validates, and it
    reads the pool's real reply (runbook "Emergency stop", for when the Hub is down)."""
    import importlib.util
    import json

    from orion.schemas.gpu_pool import GPU_POOL_CONTROL_REPLY_PREFIX, GPU_POOL_CONTROL_REQUEST_CHANNEL

    path = Path(__file__).resolve().parents[3] / "scripts" / "gpu_pool_pause.py"
    spec = importlib.util.spec_from_file_location("gpu_pool_pause", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.GPU_POOL_CONTROL_REQUEST_CHANNEL == GPU_POOL_CONTROL_REQUEST_CHANNEL

    async def go():
        rt, _ = enforce()
        await boot(rt)
        await rt.control(GpuPoolControlV1(verb="lend", card="gpu0"))   # a busy-ish pool, not a fresh one
        for action, want in (("pause", "PAUSED since"), ("pause", "PAUSED since"), ("resume", "RUNNING")):
            reply_channel, env = mod.build(action, "juniper-shell")
            assert reply_channel.startswith(GPU_POOL_CONTROL_REPLY_PREFIX) and env.reply_to == reply_channel
            out = await rt.control(GpuPoolControlV1.model_validate(env.payload))   # as _on_control does
            raw = {"type": "message", "data": json.dumps({"payload": out.model_dump(mode="json")})}
            line = mod.describe(action, mod.parse(raw))
            assert line.startswith(want), line
        assert rt.paused is None
        assert [e["holder"] for e in rt.bus.events("actuation_paused")] == ["juniper-shell"]
        refused = mod.parse({"payload": {"ok": False, "reason": "invalid:x", "detail": {}}})
        assert mod.describe("pause", refused) == "REFUSED: invalid:x"
    run(go())
