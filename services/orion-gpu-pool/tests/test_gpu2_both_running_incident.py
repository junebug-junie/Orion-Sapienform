"""Incident 2026-10-16 03:02 -> 2026-10-20 05:08: no images for 4 days.

A post-pull rebuild restarted diffusion-host on gpu2 while the agent-gpu2 seat (llama-server,
23828 MiB) held the card. diffusion's FLUX pipeline "loaded" into CPU RAM (model CPU offload) and
held almost no GPU memory. The pool's boot reconcile saw both containers running and faulted gpu2
(``reconcile_ambiguous:boot``). Discovery probed diffusion on ``/health``, which answered 200 the
whole time, so ``_clear_faults`` saw the seat alive AND the resident "up" and never cleared it;
a human's ``docker stop`` of diffusion-host cleared it 45 s later.

These tests drive the real runtime with the real prober (app.main.make_prober) over an
httpx.MockTransport that answers like the two live servers did, so the probe PATH is under test,
not a fake that ignores it.
"""
from __future__ import annotations

import httpx
import pytest

from app.main import _probe, make_prober
from tests.test_holds_and_actuation import LIVE2, SEAT, actuations, announce, boot, make, result, step
from tests.test_runtime import CFG, acq_r, run

BOTH = {SEAT: "running", "diffusion": "running"}


class Circe:
    """What each role's port answers. diffusion: /health always 200 (as live); /ready per state."""

    def __init__(self, up: set[str]):
        self.up = up
        self.diffusion_can_generate = False   # loaded into CPU RAM, but no room on the card
        self.paths: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        port, path = request.url.port, request.url.path
        role = next(r for r, s in CFG.roles.items() if s.port == port)
        self.paths.append(f"{role}{path}")
        if role not in self.up:
            raise httpx.ConnectError("refused")
        if role == "diffusion":
            if path == "/health":
                return httpx.Response(200, json={"ok": True, "model_loaded": True})
            if path == "/ready":
                ready = self.diffusion_can_generate
                return httpx.Response(200 if ready else 503, json={
                    "ready": ready, "draining": False, "model_loaded": True,
                    "gpu_available_gb": 30.5 if ready else 7.6, "gpu_min_gb": 20.0})
        if CFG.roles[role].kind == "service":
            return httpx.Response(200, json={"ok": True})
        _, file, slots, ctx = LIVE2[role]
        return httpx.Response(200, json={"model_path": f"/models/gguf/{file}", "total_slots": slots,
                                         "default_generation_settings": {"n_ctx": ctx},
                                         "modalities": {"vision": False}})


def wire(rt, circe: Circe):
    client = httpx.AsyncClient(transport=httpx.MockTransport(circe.handle))
    rt.prober = make_prober(client, CFG)


def test_discovery_probes_diffusion_on_its_launch_ready_path_and_every_other_service_on_health():
    assert CFG.roles["diffusion"].discovery_path == "/ready" and CFG.roles["diffusion"].discovery_strict
    for role, spec in CFG.roles.items():
        if role != "diffusion" and spec.kind == "service":
            assert spec.discovery_path == spec.health and not spec.discovery_strict, role


@pytest.mark.asyncio
@pytest.mark.parametrize("status, body, ok", [
    (200, {"ready": True}, True),
    (200, {"status": "ok"}, True),
    (503, {"ready": False, "model_loaded": True}, False),   # the incident: loaded, no room
    (200, {"ready": False}, False),                          # 2xx is not enough
    (200, {"ok": True, "model_loaded": True}, False),        # a /health-shaped body is not ready
])
async def test_strict_probe_needs_a_2xx_and_the_ready_body(status, body, ok):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(status, json=body))) as c:
        probe = await _probe(c, "diffusion", "http://circe:8014", "service", "/ready", strict=True)
    assert probe.ok is ok


@pytest.mark.asyncio
async def test_non_strict_service_probe_is_unchanged():
    # kev's /docs is HTML and world's /health has no "ready": both stay "answers below 500".
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>"))) as c:
        assert (await _probe(c, "kev", "http://circe:8009", "service", "/docs")).ok


def test_incident_gpu2_returns_to_idle_with_no_human_action():
    async def go():
        rt, clock = make(mode="enforce")
        rt._world.up |= {SEAT, "diffusion"}
        circe = Circe(rt._world.up)
        wire(rt, circe)
        await boot(rt)
        [st] = [m for m in actuations(rt) if m.action == "status"]
        await result(rt, st, "succeeded", observed=BOTH, in_flight=False)
        card = rt.cards["gpu2"]
        assert card.swap_state == "fault" and card.swap_role == SEAT
        assert rt.bus.events("swap_failed")[0]["reason"] == "reconcile_ambiguous:boot"

        await step(rt, clock, 30)
        assert card.swap_state == "idle", "gpu2 still faulted: discovery believed the empty diffusion"
        # Discovery asked diffusion's /ready, never its /health.
        assert "diffusion/ready" in circe.paths and "diffusion/health" not in circe.paths
        assert SEAT in card.swapped_in                     # the side really holding the card
        [cleared] = [e for e in rt.bus.events("swapped") if e["reason"] == "fault_cleared:discovery"]
        assert cleared["detail"]["loaded"] is True
        await step(rt, clock, 15)                          # discovery runs before fault clearing
        assert {d.role: d.status for d in rt.discovered}[SEAT] == "confirmed"

        # Diffusion work arrives. The pool unloads the seat by itself (the controller now resolves
        # the both-running state -- services/orion-gpu-lane-controller/tests/test_launch_exec.py).
        run_hold = await rt.acquire(acq_r("diffusion", "paint:1", kind="hold", priority="background"))
        assert run_hold.status != "granted"
        await step(rt, clock, 1800, beat=[], every=30)
        unload = [m for m in actuations(rt) if m.action == "unload"]
        assert unload and unload[0].role == SEAT
        rt._world.up.discard(SEAT)
        circe.diffusion_can_generate = True
        await result(rt, unload[0], "accepted")
        await result(rt, unload[0], "succeeded", observed={SEAT: "exited", "diffusion": "running"})
        await step(rt, clock, 30)
        assert card.swap_state == "idle" and SEAT not in card.swapped_in
        assert {d.role: d.status for d in rt.discovered}["diffusion"] == "static"
        assert not any(e["detail"].get("actor") for e in rt.bus.events("swapped"))   # no operator verb
    run(go())


def test_boot_before_the_seat_announces_does_not_clear_the_fault_the_wrong_way():
    """04:01:20 on 10-16: one second after the fault, discovery cleared it as "diffusion owns gpu2"
    (seat not yet announced after the pool restart, diffusion /health 200). The next load then hit
    both-running and faulted again. With the honest probe neither side looks up, so it waits."""
    async def go():
        rt, clock = make(mode="enforce")
        rt._world.up |= {SEAT, "diffusion"}
        circe = Circe(rt._world.up)
        wire(rt, circe)
        await rt.start()
        [st] = [m for m in actuations(rt) if m.action == "status"]
        await result(rt, st, "succeeded", observed=BOTH, in_flight=False)
        clock.advance(1)
        await rt.tick()                                   # no announcement yet
        assert rt.cards["gpu2"].swap_state == "fault"
        await announce(rt)
        clock.advance(15)
        await rt.tick()
        assert rt.cards["gpu2"].swap_state == "idle" and SEAT in rt.cards["gpu2"].swapped_in
    run(go())


# --- nobody was told: a card stuck in fault must reach Juniper ----------------------------------
# 10-16 -> 10-20 the only alert was the painting-gap card (error severity, once, 12 h in). The
# guardian watches controller refusals; this was a card fault with a healthy controller.

def test_a_card_stuck_in_fault_raises_a_critical_alert_repeats_and_recovers():
    from app.controller_alert import build_fault_request

    async def go():
        rt, clock = make(mode="enforce")
        rt._world.up |= {SEAT, "diffusion"}            # both look up: discovery cannot clear it
        sent: list[tuple[str, str, dict]] = []

        async def alert(card, state, view):
            sent.append((card, state, view))
            return True
        rt.fault_alert = alert
        await boot(rt)
        [st] = [m for m in actuations(rt) if m.action == "status"]
        await result(rt, st, "succeeded", observed=BOTH, in_flight=False)
        assert rt.cards["gpu2"].swap_state == "fault"

        await step(rt, clock, 29 * 60, every=60)
        await rt.drain_alerts()
        assert sent == []                              # self-repair gets its chance first
        await step(rt, clock, 2 * 60, every=60)
        await rt.drain_alerts()
        [(card, state, view)] = sent
        assert (card, state) == ("gpu2", "faulted") and view["seat"] == SEAT
        assert view["reason"] == "reconcile_ambiguous:boot" and view["minutes"] >= 30
        req = build_fault_request(card, state, view, source="orion-gpu-pool")
        assert req.severity == "critical" and req.require_ack   # emailed at once by orion-notify

        await step(rt, clock, 5 * 3600, every=600)
        await rt.drain_alerts()
        assert len(sent) == 1                          # no spam
        await step(rt, clock, 3600 + 60, every=600)
        await rt.drain_alerts()
        assert [s for _, s, _ in sent] == ["faulted", "faulted"]   # still stuck: told again at 6 h

        rt._world.up.discard("diffusion")              # someone stops the empty diffusion
        await step(rt, clock, 60)
        await rt.drain_alerts()
        assert rt.cards["gpu2"].swap_state == "idle"
        assert sent[-1][1] == "recovered"
        assert build_fault_request("gpu2", "recovered", sent[-1][2], source="x").severity == "info"
    run(go())


def test_an_undelivered_fault_alert_is_retried_in_minutes_not_hours():
    async def go():
        rt, clock = make(mode="enforce")
        rt._world.up |= {SEAT, "diffusion"}
        calls = []

        async def alert(card, state, view):
            calls.append(state)
            return len(calls) > 1                      # notify down the first time
        rt.fault_alert = alert
        await boot(rt)
        [st] = [m for m in actuations(rt) if m.action == "status"]
        await result(rt, st, "succeeded", observed=BOTH, in_flight=False)
        await step(rt, clock, 31 * 60, every=60)
        await rt.drain_alerts()
        assert calls == ["faulted"]
        await step(rt, clock, 11 * 60, every=60)
        await rt.drain_alerts()
        assert calls == ["faulted", "faulted"]
    run(go())


def test_an_operator_clear_is_not_recovery_until_the_card_settles():
    from orion.schemas.gpu_pool import GpuPoolControlV1

    async def go():
        rt, clock = make(mode="enforce")
        rt._world.up |= {SEAT, "diffusion"}
        sent = []

        async def alert(card, state, view):
            sent.append((state, view))
            return True
        rt.fault_alert = alert
        await boot(rt)
        [st] = [m for m in actuations(rt) if m.action == "status"]
        await result(rt, st, "succeeded", observed=BOTH, in_flight=False)
        await step(rt, clock, 31 * 60, every=60)
        await rt.drain_alerts()
        assert [s for s, _ in sent] == ["faulted"]
        since = rt._fault_since["gpu2"]
        reply = await rt.control(GpuPoolControlV1(verb="clear_fault", card="gpu2", actor="juniper"))
        assert reply.ok and rt.cards["gpu2"].swap_state in ("loading", "unloading")
        await step(rt, clock, 30)
        await rt.drain_alerts()
        assert [s for s, _ in sent] == ["faulted"]        # no false all-clear while reconciling
        assert rt._fault_since["gpu2"] == since           # the clock keeps running
    run(go())


def test_a_fault_that_survives_a_pool_restart_keeps_its_reason_and_start():
    from datetime import timedelta

    from app.store import MemoryStore

    async def go():
        store = MemoryStore()
        rt0, clock0 = make(mode="enforce")
        faulted_at = clock0() - timedelta(hours=3)
        await store.upsert_card({"card": "gpu2", "lent": False, "swapped_in": [], "swap_state": "fault",
                                 "swap_role": SEAT, "swap_generation": 365,
                                 "swap_action": {"action": "load", "action_id": "agent-gpu2:load:g365:x",
                                                 "reason": "seat_and_evicted_both_running",
                                                 "outcome": "failed", "finished_at": faulted_at.isoformat()}})
        rt, clock = make(mode="enforce", store=store)
        rt._world.up |= {SEAT, "diffusion"}
        sent = []

        async def alert(card, state, view):
            sent.append(view)
            return True
        rt.fault_alert = alert
        await rt.start()
        await announce(rt)
        await rt.tick()
        await rt.drain_alerts()
        [view] = sent                                    # 3 h old: alerts on the first tick
        assert view["reason"] == "seat_and_evicted_both_running" and view["minutes"] >= 180
    run(go())
