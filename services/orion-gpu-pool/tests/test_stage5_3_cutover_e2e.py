"""Stage 5.3 cutover, end to end: the real pool runtime drives the real circe controller.

Spec: docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md
("Corrections from building 5.2" 6, acceptance checks 1-4). Runbook:
docs/runbooks/2026-09-29-gpu-pool-stage5-3-cutover.md.

Pool side: PoolRuntime with the real scheduler and lease graph (tests/test_holds_and_actuation.py's
harness). Controller side: services/orion-gpu-lane-controller's actuator_bus.handle() ->
pool_fence.resolve() -> launch_exec.execute(), reading its own copy of the committed
config/gpu_pool.yaml. Only `docker` (FakeDocker) and HTTP to the workers (FakeHttp) are faked. Each
GpuActuateV1 the pool publishes is handed to the controller, and every GpuActuateResultV1 it
publishes goes back into the pool -- the bus hop, minus Redis.
"""
from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import shutil
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("loguru")   # the controller's logger; CI installs it for this file

from app.runtime import MAX_ACTION_TIMEOUTS
from orion.schemas.gpu_pool import GpuActuateResultV1
from tests.test_holds_and_actuation import SEAT, actuations, boot, demand_gpu2, make, step
from tests.test_runtime import CFG

REPO = Path(__file__).resolve().parents[3]
CTL_DIR = REPO / "services" / "orion-gpu-lane-controller"
# agent-gpu2's committed default (first launch.profiles entry): Ternary-Bonsai since stage 7.2.
SEAT_DEFAULT = "ternary-bonsai2-27b-pq2-v100-32gb-circe-agent"


def _controller():
    """Load the controller's app package under its own name (the pool's is `app`)."""
    pkg, app_pkg = "orion_gpu_lane_controller", "orion_gpu_lane_controller.app"
    for name, path in ((pkg, CTL_DIR), (app_pkg, CTL_DIR / "app")):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            sys.modules[name] = mod
    out = {}
    for name in ("settings", "compose", "pool_fence", "launch_exec", "actuator_bus"):
        full = f"{app_pkg}.{name}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(full, CTL_DIR / "app" / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
        out[name] = sys.modules[full]
    return types.SimpleNamespace(**out)


CTL = _controller()


class FakeDocker:
    """SafeCommandRunner stand-in: compose service state, every mutating call recorded."""

    def __init__(self, running=()):
        self.state = {s: "running" for s in running}
        self.calls: list[dict] = []

    def run(self, command, *, cwd=None, env=None):
        assert command[:2] == ["docker", "compose"], command
        rest = command[command.index("-f") + 2:]
        if rest[0] == "ps":
            st = self.state.get(rest[1])
            out = "" if st is None else f'{{"ID":"{rest[1]}","Name":"{rest[1]}","State":"{st}"}}'
            return subprocess.CompletedProcess(command, 0, out, "")
        verb, service = rest[0], rest[-1]
        profile = command[command.index("--profile") + 1] if "--profile" in command else None
        self.calls.append({"verb": verb, "service": service, "profile": profile, "env": dict(env or {})})
        self.state[service] = "exited" if verb == "stop" else "running"
        return subprocess.CompletedProcess(command, 0, "", "")

    def mutations(self):
        return [(c["verb"], c["service"]) for c in self.calls]


class FakeHttp:
    """Workers by port: up iff their compose service runs; /slots busy on request."""

    def __init__(self, docker: FakeDocker):
        self.docker = docker
        self.by_port = {str(s.port): s.launch.service for s in CFG.roles.values() if s.launch}
        self.busy: set[str] = set()
        self.never_ready: set[str] = set()
        self.draining: dict[str, bool] = {}

    async def __call__(self, url, payload=None):
        host, path = url.split("://", 1)[1].split("/", 1)
        service = self.by_port[host.split(":")[1]]
        if self.docker.state.get(service) != "running":
            raise ConnectionError("down")
        path = "/" + path
        if payload is not None:
            self.draining[service] = payload.get("draining") is True
            return {"ok": True}
        if path == "/slots":
            return [{"id": 0, "is_processing": service in self.busy}]
        if path.endswith("/status"):
            return {"draining": self.draining.get(service, False), "in_flight": False}
        if service in self.never_ready:
            return {"status": "loading"}
        return {"ready": True} if path == "/ready" else {"status": "ok"}


class Circe:
    """The controller on its own checkout, wired to the pool through `pump`."""

    ROLE_OF = {"atlas-agent-burst": SEAT, "diffusion-host": "diffusion"}

    def __init__(self, tmp_path, monkeypatch, running=("diffusion-host",)):
        root = tmp_path / "circe"
        (root / "config").mkdir(parents=True)
        # The SAME committed file the pool parsed (CFG): pool and actuator must agree on the digest.
        shutil.copy(REPO / "config" / "gpu_pool.yaml", root / "config" / "gpu_pool.yaml")
        s = CTL.settings.settings
        monkeypatch.setattr(s, "GPU_LANE_REPO_ROOT", str(root))
        monkeypatch.setattr(s, "GPU_POOL_FENCE_STATE_PATH", str(tmp_path / "fence.json"))
        monkeypatch.setattr(s, "GPU_POOL_ACTUATOR_NAME", "circe")
        monkeypatch.setattr(s, "GPU_LANE_DRAIN_TIMEOUT_SEC", 1.0)
        monkeypatch.setattr(CTL.actuator_bus, "_task", None)
        monkeypatch.setattr(CTL.actuator_bus, "_current", None)
        monkeypatch.setattr(CTL.launch_exec, "POLL_SEC", 0.001)

        self.docker = FakeDocker(running)
        self.http = FakeHttp(self.docker)
        monkeypatch.setattr(CTL.launch_exec, "runner", lambda timeout_sec: self.docker)
        monkeypatch.setattr(CTL.launch_exec, "request", self.http)
        self.seen = 0
        self.results: list[GpuActuateResultV1] = []

    def sync(self, world) -> None:
        """What the pool's probes see follows what really runs."""
        for service, role in self.ROLE_OF.items():
            (world.up.add if self.docker.state.get(service) == "running" else world.up.discard)(role)

    async def pump(self, rt) -> list[GpuActuateResultV1]:
        """Deliver every new actuation request, then every result back to the pool."""
        out: list[GpuActuateResultV1] = []
        msgs = actuations(rt)[self.seen:]
        self.seen += len(msgs)
        for msg in msgs:
            got: list[GpuActuateResultV1] = []

            async def publish(result, corr):
                got.append(result)
            await CTL.actuator_bus.handle(msg.model_dump(mode="json"), publish)
            if CTL.actuator_bus._task is not None:
                await CTL.actuator_bus._task
            self.sync(rt._world)
            for res in got:
                await rt.on_actuate_result(res)
            out += got
        self.results += out
        return out


def _real_now(clock) -> None:
    # The controller refuses a request whose deadline_at is already past on ITS clock (real UTC).
    clock.t = datetime.now(timezone.utc).replace(microsecond=0)


def _run(coro):
    return asyncio.run(coro)


async def _loaded(rt, clock, circe):
    """Demand -> the pool loads agent-gpu2 through the controller's generic path."""
    home, waiting = await demand_gpu2(rt, clock)
    await circe.pump(rt)
    return home, waiting


def test_load_then_busy_unload_then_unload_through_the_generic_path(tmp_path, monkeypatch):
    """Acceptance 1 + 2 (and the reason format): load sends the default profile and lands the 27B on
    card index 2; an unload while the 27B is busy fails with `upstream_not_idle:agent-gpu2` and
    changes nothing; a later idle unload puts diffusion back on index 2."""
    circe = Circe(tmp_path, monkeypatch)

    async def go():
        rt, clock = make()
        _real_now(clock)
        await boot(rt)
        home, waiting = await _loaded(rt, clock, circe)

        # --- load ---------------------------------------------------------------------------
        [load] = actuations(rt)
        assert (load.action, load.profile) == ("load", SEAT_DEFAULT)
        assert [(r.status, r.phase) for r in circe.results] == [
            ("accepted", None), ("progress", "draining"), ("progress", "stopping"),
            ("progress", "starting"), ("progress", "ready_wait"), ("succeeded", None)]
        assert circe.docker.mutations() == [("stop", "diffusion-host"), ("up", "atlas-agent-burst")]
        up = circe.docker.calls[-1]
        assert up["profile"] == "agent-burst"
        assert up["env"]["ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES"] == "2"
        assert up["env"]["ATLAS_AGENT_BURST_PROFILE_NAME"] == SEAT_DEFAULT
        card = rt.cards["gpu2"]
        assert SEAT in card.swapped_in and card.swap_state == "idle"
        [sw] = rt.bus.events("swapped")
        assert sw["detail"]["profile"] == SEAT_DEFAULT and sw["detail"]["observed"] == {
            SEAT: "running", "diffusion": "exited"}
        await step(rt, clock, 30, beat=[home.lease_id, waiting.lease_id])   # discovery confirms the 27B
        assert (await rt.store.lease(waiting.lease_id))["role"] == SEAT

        # --- unload refused by a busy 27B -----------------------------------------------------
        await rt.release(home.lease_id, "ok")
        await rt.release(waiting.lease_id, "ok")
        circe.http.busy.add("atlas-agent-burst")
        n = len(circe.docker.calls)
        for _ in range(40):
            await step(rt, clock, 30)
            await circe.pump(rt)
            if rt.bus.events("swap_failed"):
                break
        [failed] = rt.bus.events("swap_failed")
        assert failed["reason"] == "upstream_not_idle:agent-gpu2" and failed["detail"]["action"] == "unload"
        assert failed["detail"]["profile"] is None                     # an unload never names a model
        assert len(circe.docker.calls) == n                           # nothing was stopped
        # grammar carries the reason verbatim; nothing parses it
        assert any("reason=upstream_not_idle:agent-gpu2" in g["atom"]["summary"] for g in rt.bus.grammar())
        # An unload that failed leaves the card faulted until discovery sees a consistent card (as
        # with the bridge's burst_upstream_not_idle); here: seat up, diffusion down -> still loaded.
        for _ in range(4):
            await step(rt, clock, 30)
            await circe.pump(rt)
        assert card.swap_state == "idle" and SEAT in card.swapped_in

        # --- idle unload once the 27B is free -----------------------------------------------------
        circe.http.busy.clear()
        for _ in range(80):
            await step(rt, clock, 30)
            await circe.pump(rt)
            if SEAT not in card.swapped_in and card.swap_state == "idle":
                break
        assert SEAT not in card.swapped_in and card.swap_state == "idle"
        assert circe.docker.mutations()[-2:] == [("stop", "atlas-agent-burst"), ("up", "diffusion-host")]
        restore = circe.docker.calls[-1]
        assert restore["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        assert circe.docker.state == {"atlas-agent-burst": "exited", "diffusion-host": "running"}
        assert [e["detail"]["action"] for e in rt.bus.events("swapped")][-1] == "unload"

    _run(go())


def test_failed_load_is_rolled_back_and_the_pool_cools_down(tmp_path, monkeypatch):
    """Acceptance 3: the 27B never gets ready -> the controller stops it, restarts diffusion on
    index 2 and reports restored=true; the pool records swap_failed with the role-suffixed reason,
    keeps the card idle (not fault) and backs off."""
    circe = Circe(tmp_path, monkeypatch)
    circe.http.never_ready.add("atlas-agent-burst")
    real_build = CTL.pool_fence.build_plan

    def short_ready(cfg, role, profile):   # 900 s of wall clock is not a unit test
        plan = real_build(cfg, role, profile)
        return dataclasses.replace(plan, seat=dataclasses.replace(plan.seat, timeout_sec=0.05))
    monkeypatch.setattr(CTL.pool_fence, "build_plan", short_ready)

    async def go():
        rt, clock = make()
        _real_now(clock)
        await boot(rt)
        await _loaded(rt, clock, circe)
        final = circe.results[-1]
        assert (final.status, final.restored, final.reason) == ("failed", True, "model_readiness_timeout:agent-gpu2")
        assert circe.docker.mutations() == [("stop", "diffusion-host"), ("up", "atlas-agent-burst"),
                                            ("stop", "atlas-agent-burst"), ("up", "diffusion-host")]
        assert circe.docker.calls[-1]["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        [failed] = rt.bus.events("swap_failed")
        assert failed["reason"] == "model_readiness_timeout:agent-gpu2"
        assert failed["detail"]["restored"] is True and failed["detail"]["profile"] == SEAT_DEFAULT
        card = rt.cards["gpu2"]
        assert card.swap_state == "idle" and SEAT not in card.swapped_in
        assert card.cooldown_until is not None and card.cooldown_until > clock()

    _run(go())


def test_actuation_deadline_and_stuck_ceiling_cover_the_controllers_worst_case():
    """Stage 4.3 rule, re-checked for the 900 s ready wait (was 600 s on the bridge): the pool's
    first deadline is the sum of the launch timeouts the action touches, and it faults a card as
    stuck only after MAX_ACTION_TIMEOUTS x that. A failed load's realistic worst case on the
    controller -- two `docker ps` reads, diffusion's drain, the seat's ready wait, diffusion's
    restore ready wait, plus four docker stop/up calls of up to 15 s -- must fit inside that
    ceiling. Four docker calls each hanging near GPU_LANE_COMMAND_TIMEOUT_SEC do NOT fit: that is a
    wedged actuator, which the ceiling exists to fault (runtime._action_timeout docstring)."""
    rt, _ = make()
    budget = rt._action_timeout(SEAT)
    seat, diffusion = CFG.roles[SEAT].launch, CFG.roles["diffusion"].launch
    assert budget == seat.timeout_sec + diffusion.timeout_sec == 1500
    fields = CTL.settings.Settings.model_fields     # the shipped defaults, not whatever .env says
    drain, command = fields["GPU_LANE_DRAIN_TIMEOUT_SEC"].default, fields["GPU_LANE_COMMAND_TIMEOUT_SEC"].default
    reads = 2 * CTL.launch_exec.READ_TIMEOUT_SEC
    realistic = reads + drain + seat.timeout_sec + diffusion.timeout_sec + 4 * 15
    wedged = reads + drain + seat.timeout_sec + diffusion.timeout_sec + 4 * command
    assert budget < realistic < MAX_ACTION_TIMEOUTS * budget   # pool polls `status`, then stays patient
    assert wedged > MAX_ACTION_TIMEOUTS * budget                # a hung docker call faults the card


# --- scripts/gpu_pool_actuator_probe.py against the real controller ---------------------------------

def _probe_module():
    spec = importlib.util.spec_from_file_location("gpu_pool_actuator_probe", REPO / "scripts" / "gpu_pool_actuator_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ask(circe, msg) -> list[dict]:
    got: list[dict] = []

    async def go():
        async def publish(result, corr):
            got.append(result.model_dump(mode="json"))
        await CTL.actuator_bus.handle(msg.model_dump(mode="json"), publish)
        if CTL.actuator_bus._task is not None:
            await CTL.actuator_bus._task
    _run(go())
    return got


def test_probe_reads_status_and_digest_agreement_without_touching_anything(tmp_path, monkeypatch):
    """The runbook's pre/post-deploy probe: `status` answers with what is on the card; `digest` is
    refused profile_not_allowed when both checkouts agree -- no docker call, no generation spent."""
    probe = _probe_module()
    circe = Circe(tmp_path, monkeypatch)
    status = _ask(circe, probe.build(CFG, SEAT, "status"))
    assert probe.verdict("status", status).startswith("OK: observed={'agent-gpu2': 'absent', 'diffusion': 'running'}")
    digest = _ask(circe, probe.build(CFG, SEAT, "digest"))
    assert probe.verdict("digest", digest) == "OK: launch digests agree"
    assert circe.docker.calls == []
    assert CTL.pool_fence.read_state()["generations"] == {}


def test_probe_reports_a_controller_checkout_on_another_commit(tmp_path, monkeypatch):
    """circe's checkout on another launch block (here: a different ready timeout) than the pool's: MISMATCH."""
    import yaml
    probe = _probe_module()
    circe = Circe(tmp_path, monkeypatch)
    path = tmp_path / "circe" / "config" / "gpu_pool.yaml"
    data = yaml.safe_load(path.read_text())
    data["roles"][SEAT]["launch"]["timeout_sec"] = 901
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    got = _ask(circe, probe.build(CFG, SEAT, "digest"))
    assert probe.verdict("digest", got) == "MISMATCH: controller checkout/image is not on this commit"
    assert probe.verdict("digest", []).startswith("NO ANSWER")
    assert circe.docker.calls == []
