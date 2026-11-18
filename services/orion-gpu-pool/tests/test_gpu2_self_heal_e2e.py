"""End to end: a rebuild starts the wrong role on gpu2, and the card heals with no human action.

Real pool runtime (enforce, real scheduler, lease graph, in-memory store) wired to the REAL lane
controller code (services/orion-gpu-lane-controller/app: actuator_bus -> pool_fence -> launch_exec)
over a bridged fake bus. Only circe itself is simulated (`Circe`): which containers run, what each
port answers, and what nvidia-smi shows. The pool probes it through the real prober
(app.main.make_prober); the controller drives it through `docker compose`, `docker top` and
`docker exec ... nvidia-smi`.

Incident 2026-10-16 -> 10-20: a post-pull rebuild restarted diffusion-host (and the pool) while the
agent-gpu2 seat held gpu2. No images for 4 days.
"""
from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

pytest.importorskip("loguru")   # the controller's logger; CI installs it (orion-gpu-pool-tests.yml)

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app.main import make_prober
from app.runtime import PoolRuntime
from app.store import MemoryStore
from orion.gpu_pool.config import launch_digest, load_pool_config
from orion.gpu_pool.lease_graph import build_lease_graph
from orion.schemas.gpu_pool import GpuActuateV1, GpuLeaseRequestV1, LlmWorkerAnnounceV1
from tests.test_holds_and_actuation import LIVE2, SEAT, hold
from tests.test_runtime import CFG, CLEAR_GUARDS, PROFILES, SEAT_WAIT, Clock, FakeBus, acq_r, run

REPO = Path(__file__).resolve().parents[3]
CONTROLLER = REPO / "services" / "orion-gpu-lane-controller"
ACTUATE = "orion:gpu_pool:actuate:request"
SEAT_SERVICE, DIFF_SERVICE = "atlas-agent-burst", "diffusion-host"
SEAT_PID, DIFF_PID, LLAMA_MIB = 3846941, 26001, 23828.0


def _load_controller():
    """The controller's app modules under their own package name (as its own tests do), so they
    never collide with this service's ``app`` package."""
    pkg, app_pkg = "orion_gpu_lane_controller", "orion_gpu_lane_controller.app"
    for name, path in ((pkg, CONTROLLER), (app_pkg, CONTROLLER / "app")):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    out = {}
    for name in ("settings", "compose", "pool_fence", "launch_exec", "actuator_bus"):
        full = f"{app_pkg}.{name}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(full, CONTROLLER / "app" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            spec.loader.exec_module(module)
        out[name] = sys.modules[full]
    return out


ctl = _load_controller()


class Circe:
    """circe's docker daemon, ports and nvidia-smi, as far as gpu2 is concerned."""

    def __init__(self):
        self.running: set[str] = set()
        self.calls: list[tuple[str, str]] = []       # compose mutations: (verb, service)
        self.draining = False                         # diffusion-host's drain latch

    # -- what the GPU holds: the seat's llama-server holds 23.8 GB; offloaded FLUX ~0.3 GB idle.
    def gpu_mib(self) -> dict[int, float]:
        out = {3932600: 872.0}                        # world-model, always on gpu2
        if SEAT_SERVICE in self.running:
            out[SEAT_PID] = LLAMA_MIB
        if DIFF_SERVICE in self.running:
            out[DIFF_PID] = 310.0
        return out

    def diffusion_can_generate(self) -> bool:
        # Honest /ready: loaded (CPU RAM) and the card has room (the seat is not on it).
        return DIFF_SERVICE in self.running and SEAT_SERVICE not in self.running

    # -- docker (the controller's SafeCommandRunner stand-in)
    def run(self, command, *, cwd=None, env=None):
        if command[1] == "top":
            service = command[2].removesuffix("-id")
            pid = {SEAT_SERVICE: SEAT_PID, DIFF_SERVICE: DIFF_PID}[service]
            ok = service in self.running
            return subprocess.CompletedProcess(command, 0 if ok else 1, f"PID\n{pid}\n" if ok else "", "")
        if command[1] == "exec":
            body = "".join(f"{p}, {m:.0f}\n" for p, m in self.gpu_mib().items())
            return subprocess.CompletedProcess(command, 0, body, "")
        rest = command[command.index("-f") + 2:]
        if rest[0] == "ps":
            service = rest[1]
            state = "running" if service in self.running else ("exited" if service in self.seen else None)
            out = "" if state is None else f'{{"ID":"{service}-id","Name":"{service}","State":"{state}"}}'
            return subprocess.CompletedProcess(command, 0, out, "")
        verb, service = rest[0], rest[-1]
        self.calls.append((verb, service))
        self.seen.add(service)
        if verb == "stop":
            self.running.discard(service)
            if service == DIFF_SERVICE:
                self.draining = False                 # a fresh process starts undrained
        elif verb == "up":
            self.running.add(service)
        return subprocess.CompletedProcess(command, 0, "", "")

    seen: set[str] = set()

    # -- HTTP the controller asks (diffusion drain/ready, llama /health and /slots)
    async def controller_http(self, url, payload=None):
        port = int(url.split("://", 1)[1].split("/", 1)[0].split(":")[1])
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        service = SEAT_SERVICE if port == CFG.roles[SEAT].port else DIFF_SERVICE
        if service not in self.running:
            raise ConnectionError("down")
        if service == SEAT_SERVICE:
            return [{"id": 0, "is_processing": False}] if path == "/slots" else {"status": "ok"}
        if path.startswith("/v1/lifecycle"):
            if payload is not None:
                self.draining = bool(payload.get("draining"))
            return {"draining": self.draining, "in_flight": False}
        return {"ready": self.diffusion_can_generate() and not self.draining}

    # -- HTTP the pool's discovery asks
    def pool_http(self, request: httpx.Request) -> httpx.Response:
        role = next(r for r, s in CFG.roles.items() if s.port == request.url.port)
        if role == SEAT:
            if SEAT_SERVICE not in self.running:
                raise httpx.ConnectError("refused")
        elif role == "diffusion":
            if DIFF_SERVICE not in self.running:
                raise httpx.ConnectError("refused")
            if request.url.path == "/health":
                return httpx.Response(200, json={"ok": True, "model_loaded": True})
            ready = self.diffusion_can_generate() and not self.draining
            return httpx.Response(200 if ready else 503, json={"ready": ready})
        if CFG.roles[role].kind == "service":
            return httpx.Response(200, json={"ok": True})
        _, file, slots, ctx = LIVE2[role]
        return httpx.Response(200, json={"model_path": f"/models/gguf/{file}", "total_slots": slots,
                                         "default_generation_settings": {"n_ctx": ctx},
                                         "modalities": {"vision": False}})


class Bridge(FakeBus):
    """The pool's bus: actuation requests are handed to the real controller code, its results
    come back through rt.on_actuate_result."""

    def __init__(self):
        super().__init__()
        self.pending: list[dict] = []

    async def publish(self, channel, env):
        await super().publish(channel, env)
        if channel == ACTUATE:
            self.pending.append(env.payload)


class World:
    def __init__(self, tmp_path: Path, monkeypatch):
        root = tmp_path / "circe-repo"
        (root / "config").mkdir(parents=True)
        (root / "config" / "gpu_pool.yaml").write_text((REPO / "config" / "gpu_pool.yaml").read_text())
        assert launch_digest(load_pool_config(root / "config" / "gpu_pool.yaml"), SEAT) == launch_digest(CFG, SEAT)
        s, lx, bus = ctl["settings"].settings, ctl["launch_exec"], ctl["actuator_bus"]
        monkeypatch.setattr(s, "GPU_LANE_REPO_ROOT", str(root))
        monkeypatch.setattr(s, "GPU_POOL_FENCE_STATE_PATH", str(tmp_path / "fence.json"))
        monkeypatch.setattr(s, "GPU_POOL_ACTUATOR_NAME", "circe")
        monkeypatch.setattr(s, "GPU_LANE_DRAIN_TIMEOUT_SEC", 0.5)
        monkeypatch.setattr(bus, "_task", None)
        monkeypatch.setattr(bus, "_current", None)
        monkeypatch.setattr(lx, "POLL_SEC", 0.001)
        self.circe = Circe()
        self.circe.seen = set()
        monkeypatch.setattr(lx, "runner", lambda timeout_sec: self.circe)
        monkeypatch.setattr(lx, "request", self.circe.controller_http)
        self.store = MemoryStore()
        self.clock = Clock()
        self.clock.t = datetime.now(timezone.utc)   # the controller checks deadlines on the real clock
        self.rt: PoolRuntime | None = None

    def boot_pool(self) -> PoolRuntime:
        """A (re)started pool process on the same store -- what a post-pull rebuild does."""
        rt = PoolRuntime(cfg=CFG, profiles=PROFILES, store=self.store,
                         graph=build_lease_graph(lambda: CFG, MemorySaver()), bus=Bridge(),
                         prober=make_prober(httpx.AsyncClient(transport=httpx.MockTransport(self.circe.pool_http)),
                                            CFG),
                         now=self.clock, probe_interval_sec=0, mode="enforce")
        rt.guard_states = dict(CLEAR_GUARDS)
        self.rt = rt
        return rt

    async def announce(self):
        for role, (profile, _, _, _) in LIVE2.items():
            if role == SEAT and SEAT_SERVICE not in self.circe.running:
                continue
            await self.rt.on_announce(LlmWorkerAnnounceV1(host="circe", role=role, profile_name=profile,
                                                          port=CFG.roles[role].port, announced_at=self.rt.now()))

    async def pump(self):
        """Hand every pending actuation to the controller and feed its results back."""
        bus = ctl["actuator_bus"]
        while self.rt.bus.pending:
            payload = self.rt.bus.pending.pop(0)

            async def publish(result, corr=None):
                await self.rt.on_actuate_result(result)
            await bus.handle(payload, publish)
            if bus._task is not None:
                await bus._task

    async def step(self, sec, every=30, beat=()):
        left = sec
        while left > 0:
            dt = min(every, left)
            self.clock.advance(dt)
            left -= dt
            for lid in beat:
                await self.rt.heartbeat(lid)
            await self.announce()
            await self.rt.tick()
            await self.pump()

    async def start(self):
        await self.rt.start()
        await self.announce()
        await self.rt.tick()
        await self.pump()

    def actuations(self):
        return [GpuActuateV1.model_validate(e.payload) for c, e in self.rt.bus.published if c == ACTUATE]

    def card(self):
        return self.rt.cards["gpu2"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


async def lend_gpu2_to_the_seat(w: World):
    """Diffusion owns gpu2; agent work waits past the seat's after_wait; the pool loads the seat."""
    w.circe.running |= {DIFF_SERVICE}
    w.circe.seen |= {DIFF_SERVICE}
    w.boot_pool()
    await w.start()
    assert w.card().swap_state == "idle" and SEAT not in w.card().swapped_in
    home = await w.rt.acquire(hold("home:1"))
    waiting = await w.rt.acquire(hold("wait:1"))
    await w.step(SEAT_WAIT + 60, beat=[home.lease_id])
    assert SEAT in w.card().swapped_in and w.card().swap_state == "idle", w.card().swap_state
    assert w.circe.running == {SEAT_SERVICE}
    for lid in (home.lease_id, waiting.lease_id):
        await w.rt.release(lid, "ok", None)


async def paint(w: World):
    """Diffusion work arrives; returns the hold's final status after the pool acted on it."""
    h = await w.rt.acquire(acq_r("diffusion", "paint:1", kind="hold", priority="background"))
    for _ in range(80):
        if h.status == "granted":
            break
        await w.step(30)
        h = await w.rt.status(h.lease_id)
    return h


def test_rebuild_restarts_diffusion_and_the_pool_while_gpu2_is_lent__heals_and_paints(world):
    """The exact 10-16 chain: seat holds gpu2, a rebuild brings diffusion-host up beside it (empty:
    no room) and restarts the pool. Boot reconcile sees both running -> fault. Discovery heals it
    to idle (seat loaded), and the next painting unloads the seat through the controller, which
    stops the empty diffusion, unloads the seat, and starts diffusion fresh."""
    async def go():
        w = world
        await lend_gpu2_to_the_seat(w)
        w.circe.running.add(DIFF_SERVICE)              # docker compose up -d (post-pull rebuild)
        w.boot_pool()                                  # ... which also restarted the pool
        await w.start()
        assert w.card().swap_state == "fault"
        assert w.rt.bus.events("swap_failed")[0]["reason"] == "reconcile_ambiguous:boot"
        await w.step(60)
        assert w.card().swap_state == "idle" and SEAT in w.card().swapped_in   # healed, no human
        h = await paint(w)
        assert h.status == "granted", h
        assert w.circe.running == {DIFF_SERVICE} and SEAT not in w.card().swapped_in
        assert w.card().swap_state == "idle"
        assert ("stop", DIFF_SERVICE) in w.circe.calls and ("up", DIFF_SERVICE) in w.circe.calls
        assert not [e for e in w.rt.bus.events("swap_failed")
                    if e["reason"].startswith("seat_and_evicted_both_running")]
    run(go())


def test_rebuild_restarts_diffusion_beside_a_lent_seat_without_a_pool_restart(world):
    """No fault this time (the pool never restarted): the next painting's unload meets both
    running, and the controller resolves it instead of refusing."""
    async def go():
        w = world
        await lend_gpu2_to_the_seat(w)
        w.circe.running.add(DIFF_SERVICE)
        await w.step(60)
        h = await paint(w)
        assert h.status == "granted", h
        assert w.circe.running == {DIFF_SERVICE} and w.card().swap_state == "idle"
    run(go())


def test_rebuild_restarts_the_seat_mid_lend(world):
    """The seat container restarts while it holds gpu2 (down a few probes, then back). The card
    must not fault, must not thrash, and painting afterwards still works."""
    async def go():
        w = world
        await lend_gpu2_to_the_seat(w)
        sent = len(w.actuations())
        w.circe.running.discard(SEAT_SERVICE)          # recreate: container gone briefly
        await w.step(45, every=15)
        w.circe.running.add(SEAT_SERVICE)              # ... and back
        await w.step(120)
        assert w.card().swap_state == "idle" and SEAT in w.card().swapped_in
        assert len(w.actuations()) == sent              # nothing re-actuated by the blip
        h = await paint(w)
        assert h.status == "granted", h
        assert w.circe.running == {DIFF_SERVICE}
    run(go())


def test_rebuild_starts_the_seat_on_a_card_diffusion_owns(world):
    """The mirror: diffusion owns gpu2 and a rebuild brings the seat container up beside it. The
    seat takes the memory, so diffusion honestly reads not-ready. Agent demand then loads the seat
    through the controller (empty diffusion stopped), and a later painting gets the card back."""
    async def go():
        w = world
        w.circe.running |= {DIFF_SERVICE}
        w.circe.seen |= {DIFF_SERVICE}
        w.boot_pool()
        await w.start()
        w.circe.running.add(SEAT_SERVICE)              # rebuild ups the seat container
        w.circe.seen.add(SEAT_SERVICE)
        await w.step(60)
        home = await w.rt.acquire(hold("home:1"))
        waiting = await w.rt.acquire(hold("wait:1"))
        await w.step(SEAT_WAIT + 60, beat=[home.lease_id])
        assert SEAT in w.card().swapped_in and w.card().swap_state == "idle", w.card().swap_state
        assert w.circe.running == {SEAT_SERVICE}
        for lid in (home.lease_id, waiting.lease_id):
            await w.rt.release(lid, "ok", None)
        h = await paint(w)
        assert h.status == "granted", h
        assert w.circe.running == {DIFF_SERVICE}
    run(go())
