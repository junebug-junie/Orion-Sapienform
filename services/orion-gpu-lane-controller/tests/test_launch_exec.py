"""Stage 5.2: the generic launch actuator (app/launch_exec.py + pool_fence.resolve's LaunchPlan).

Spec: docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md
(Decision 2, worked examples A and B). Everything runs through the real actuator_bus.handle() ->
pool_fence.resolve() -> launch_exec.execute() path; only the docker binary (FakeDocker, a
SafeCommandRunner stand-in) and HTTP (FakeHttp) are faked, so these pin the exact compose calls and
env the controller would issue on circe.
"""
import asyncio
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from test_api import REPO_ROOT, main_module
from orion.gpu_pool.config import PoolConfig, launch_digest, load_pool_config
from orion.schemas.gpu_pool import GpuActuateResultV1

bus = main_module.actuator_bus
fence = main_module.pool_fence
settings = main_module.settings
lx = bus.launch_exec

VISION_PROFILE = "qwen3-vl-8b-vision-test"
ALT_27B = "gemma-27b-alt-test"
DEFAULT_27B = "qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex"
# The committed agent-gpu2 default (first launch.profiles entry) since stage 7.2.
LIVE_DEFAULT = "qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex"  # Bonsai rolled back 2026-10-07
LLAMA = "services/orion-llamacpp-host/docker-compose.atlas-workers.yml"
LLAMA_ENV = "services/orion-llamacpp-host/.env"


# --- config variants ---------------------------------------------------------------------------

def real_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "config" / "gpu_pool.yaml").read_text())


def example_a(data: dict) -> dict:
    """Worked example A: gpu4 hosts either an 8B (fast2, resident) or a vision model (vision4 seat)."""
    data["cards"]["gpu4"] = {"vram_gb": 32, "index": 4}
    data["roles"]["fast2"] = {
        "kind": "llm", "cards": ["gpu4"], "owner": ["metacog", "fast"], "port": 8017,
        "launch": {"actuator": "circe", "compose": LLAMA, "env_file": LLAMA_ENV, "service": "atlas-fast2",
                   "cuda_env": "ATLAS_FAST2_CUDA_VISIBLE_DEVICES", "ready": "/health", "timeout_sec": 0.05}}
    data["roles"]["vision4"] = {
        "kind": "llm", "cards": ["gpu4"], "owner": "vision", "port": 8018,
        "launch": {"actuator": "circe", "compose": LLAMA, "env_file": LLAMA_ENV, "service": "atlas-vision4",
                   "compose_profile": "vision4", "cuda_env": "ATLAS_VISION4_CUDA_VISIBLE_DEVICES",
                   "profile_var": "ATLAS_VISION4_PROFILE_NAME", "profiles": [VISION_PROFILE],
                   "ready": "/health", "timeout_sec": 0.05},
        "swap": {"evicts": ["fast2"], "guards": ["thermal"]}}
    data["classes"]["fast"]["roles"].insert(1, "fast2")
    data["classes"]["metacog"]["roles"].insert(2, "fast2")
    data["classes"]["vision"] = {"roles": ["vision4"], "on_unavailable": "backlog"}
    return data


def example_b(data: dict) -> dict:
    """Worked example B (the 5.3 end state): agent-gpu2 without bridge verbs, two model options."""
    seat = data["roles"]["agent-gpu2"]
    seat["swap"].pop("load", None)     # already gone from the committed file since 5.3
    seat["swap"].pop("unload", None)
    seat["launch"]["profiles"] = [DEFAULT_27B, ALT_27B]
    seat["launch"]["timeout_sec"] = 0.05
    data["roles"]["diffusion"]["launch"]["timeout_sec"] = 0.05
    return data


# --- fakes -------------------------------------------------------------------------------------

class FakeDocker:
    """Stands in for SafeCommandRunner: tracks compose service state, records every call."""

    def __init__(self, running=(), broken=()):
        self.state = {s: "running" for s in running}
        self.broken = set(broken)          # `ps` fails for these (unknown Docker state)
        self.never_start = set()           # `up` exits non-zero
        self.stop_fails = set()            # `stop` exits non-zero
        self.calls: list[dict] = []

    def run(self, command, *, cwd=None, env=None):
        assert command[:2] == ["docker", "compose"], command
        profile = command[command.index("--profile") + 1] if "--profile" in command else None
        compose = command[command.index("-f") + 1]
        rest = command[command.index("-f") + 2:]
        if rest[0] == "ps":
            service = rest[1]
            if service in self.broken:
                return subprocess.CompletedProcess(command, 1, "", "boom")
            st = self.state.get(service)
            out = "" if st is None else f'{{"ID":"{service}-id","Name":"{service}","State":"{st}"}}'
            return subprocess.CompletedProcess(command, 0, out, "")
        verb, service = rest[0], rest[-1]
        # Only what the controller changed relative to its own process env.
        interesting = {k: v for k, v in (env or {}).items() if os.environ.get(k) != v}
        self.calls.append({"verb": verb, "service": service, "args": rest[1:-1], "profile": profile,
                           "compose": compose, "env": interesting, "full_env": dict(env or {})})
        if verb == "stop":
            if service in self.stop_fails:
                return subprocess.CompletedProcess(command, 1, "", "no")
            if service in self.state:
                self.state[service] = "exited"
            return subprocess.CompletedProcess(command, 0, "", "")
        if verb == "up":
            if service in self.never_start:
                return subprocess.CompletedProcess(command, 1, "", "no")
            self.state[service] = "running"
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected compose verb {verb}")

    def mutations(self):
        return [(c["verb"], c["service"]) for c in self.calls]


class FakeHttp:
    """HTTP by port: ready iff the service is running (unless listed in `never_ready`)."""

    def __init__(self, docker: FakeDocker, cfg):
        self.docker = docker
        self.by_port = {str(spec.port): spec.launch.service for spec in cfg.roles.values() if spec.launch}
        self.never_ready: set[str] = set()
        self.busy: set[str] = set()
        self.stuck_in_flight: set[str] = set()   # drain status never reports in_flight=false
        self.on_status = None                     # called on each drain-status poll
        self.on_ready = None                      # called on each ready poll
        self.on_slots = None                      # called on each /slots read
        self.draining: dict[str, bool] = {}
        self.posts: list[tuple[str, str, dict]] = []

    async def __call__(self, url, payload=None):
        port, path = url.split("://", 1)[1].split("/", 1)[0].split(":")[1], "/" + url.split("://", 1)[1].split("/", 1)[1]
        service = self.by_port[port]
        up = self.docker.state.get(service) == "running"
        if payload is not None:
            self.posts.append((service, path, payload))
            if not up:
                raise ConnectionError("down")
            self.draining[service] = payload.get("draining") is True
            return {"ok": True}
        if not up:
            raise ConnectionError("down")
        if path == "/slots":
            if self.on_slots is not None:
                self.on_slots()
            return [{"id": 0, "is_processing": service in self.busy}]
        if path.endswith("/status"):
            if self.on_status is not None:
                self.on_status()
            return {"draining": self.draining.get(service, False), "in_flight": service in self.stuck_in_flight}
        if self.on_ready is not None:
            self.on_ready()
        if service in self.never_ready:
            return {"status": "loading", "ready": False}
        return {"ready": True} if path == "/ready" else {"status": "ok"}


# --- harness -----------------------------------------------------------------------------------

@pytest.fixture
def world(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    monkeypatch.setattr(settings, "GPU_LANE_REPO_ROOT", str(root))
    monkeypatch.setattr(settings, "GPU_POOL_FENCE_STATE_PATH", str(tmp_path / "state" / "fence.json"))
    monkeypatch.setattr(settings, "GPU_LANE_DRAIN_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(settings, "GPU_POOL_ACTUATOR_NAME", "circe")
    monkeypatch.setattr(bus, "_task", None)
    monkeypatch.setattr(bus, "_current", None)
    monkeypatch.setattr(lx, "POLL_SEC", 0.001)

    class World:
        def write(self, data):
            (root / "config" / "gpu_pool.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            self.cfg = load_pool_config(root / "config" / "gpu_pool.yaml")

        def start(self, running=()):
            self.docker = FakeDocker(running=running)
            self.http = FakeHttp(self.docker, self.cfg)
            monkeypatch.setattr(lx, "runner", lambda timeout_sec: self.docker)
            monkeypatch.setattr(lx, "request", self.http)
            return self

        def payload(self, role, cards, **over):
            body = {"action_id": f"pool:{role}:1", "generation": 1, "actuator": "circe", "role": role,
                    "action": "load", "cards": cards, "profile": None,
                    "launch_digest": launch_digest(self.cfg, role),
                    "deadline_at": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
                    "reason": "demand"}
            body.update(over)
            return body

        def handle(self, body):
            sink: list[GpuActuateResultV1] = []

            async def publish(result, corr):
                sink.append(result)

            async def scenario():
                await bus.handle(body, publish)
                if bus._task is not None:
                    await bus._task
            asyncio.run(scenario())
            return sink

    w = World()
    w.root = root
    return w


def statuses(sink):
    return [(r.status, r.phase) for r in sink]


# --- worked example A: gpu4 hosting an 8B or a vision model ------------------------------------

def test_example_a_load_vision_on_gpu4_evicts_the_8b(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-fast2", "diffusion-host"])
    sink = world.handle(world.payload("vision4", ["gpu4"], profile=VISION_PROFILE))
    assert statuses(sink) == [("accepted", None), ("progress", "stopping"), ("progress", "starting"),
                              ("progress", "ready_wait"), ("succeeded", None)]
    assert world.docker.mutations() == [("stop", "atlas-fast2"), ("up", "atlas-vision4")]
    up = world.docker.calls[-1]
    # The card index and the allow-listed profile are the only values the controller sets.
    assert up["env"] == {"ATLAS_VISION4_CUDA_VISIBLE_DEVICES": "4", "ATLAS_VISION4_PROFILE_NAME": VISION_PROFILE}
    assert up["profile"] == "vision4" and up["compose"] == LLAMA
    assert up["args"] == ["-d", "--no-build", "--no-deps"]
    final = sink[-1]
    # observed covers every launch role on this actuator, not a fixed gpu2 pair.
    assert final.observed == {"agent-gpu2": "absent", "diffusion": "running", "fast2": "exited",
                              "vision4": "running"}
    assert final.restored is None


def test_example_a_unload_restores_the_8b_on_its_card(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-vision4"])
    sink = world.handle(world.payload("vision4", ["gpu4"], action="unload", reason="idle"))
    assert sink[-1].status == "succeeded"
    assert world.docker.mutations() == [("stop", "atlas-vision4"), ("up", "atlas-fast2")]
    assert world.docker.calls[-1]["env"] == {"ATLAS_FAST2_CUDA_VISIBLE_DEVICES": "4"}   # no profile var


def test_example_a_unload_refuses_busy_llm_slots(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-vision4"])
    world.http.busy.add("atlas-vision4")
    sink = world.handle(world.payload("vision4", ["gpu4"], action="unload"))
    assert sink[-1].status == "failed" and sink[-1].reason == "upstream_not_idle:vision4"
    assert sink[-1].restored is None
    assert world.docker.mutations() == []


def test_example_a_readiness_timeout_rolls_back_to_the_8b(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-fast2"])
    world.http.never_ready.add("atlas-vision4")
    sink = world.handle(world.payload("vision4", ["gpu4"], profile=VISION_PROFILE))
    assert statuses(sink) == [("accepted", None), ("progress", "stopping"), ("progress", "starting"),
                              ("progress", "ready_wait"), ("progress", "rolling_back"),
                              ("progress", "ready_wait"), ("failed", None)]   # 2nd ready_wait: fast2 back
    assert world.docker.mutations() == [("stop", "atlas-fast2"), ("up", "atlas-vision4"),
                                        ("stop", "atlas-vision4"), ("up", "atlas-fast2")]
    assert sink[-1].restored is True and sink[-1].reason == "model_readiness_timeout:vision4"
    assert sink[-1].observed["fast2"] == "running" and sink[-1].observed["vision4"] == "exited"


def test_example_a_failed_rollback_reports_not_restored(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-fast2"])
    world.docker.never_start.add("atlas-vision4")
    world.docker.never_start.add("atlas-fast2")
    sink = world.handle(world.payload("vision4", ["gpu4"], profile=VISION_PROFILE))
    assert sink[-1].status == "failed" and sink[-1].restored is False
    assert sink[-1].reason == "startup_failed:vision4:restoration_failed"


def test_example_a_noop_when_already_loaded(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-vision4"])
    sink = world.handle(world.payload("vision4", ["gpu4"], profile=VISION_PROFILE))
    assert statuses(sink) == [("accepted", None), ("succeeded", None)] and sink[-1].reason == "noop"
    assert world.docker.mutations() == []


# --- worked example B: a second model option for gpu2's 27B seat --------------------------------

def test_example_b_second_profile_drains_diffusion_then_loads_it(world):
    world.write(example_b(real_config()))
    world.start(running=["diffusion-host"])
    sink = world.handle(world.payload("agent-gpu2", ["gpu2"], profile=ALT_27B))
    assert statuses(sink) == [("accepted", None), ("progress", "draining"), ("progress", "stopping"),
                              ("progress", "starting"), ("progress", "ready_wait"), ("succeeded", None)]
    assert world.http.posts[0] == ("diffusion-host", "/v1/lifecycle/drain", {"draining": True})
    assert world.docker.mutations() == [("stop", "diffusion-host"), ("up", "atlas-agent-burst")]
    up = world.docker.calls[-1]
    assert up["profile"] == "agent-burst"
    assert up["env"] == {"ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES": "2", "ATLAS_AGENT_BURST_PROFILE_NAME": ALT_27B}
    assert sink[-1].observed == {"agent-gpu2": "running", "diffusion": "exited"}


def test_example_b_no_profile_leaves_the_compose_default(world):
    world.write(example_b(real_config()))
    world.start(running=["diffusion-host"])
    sink = world.handle(world.payload("agent-gpu2", ["gpu2"]))
    assert sink[-1].status == "succeeded"
    assert world.docker.calls[-1]["env"] == {"ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES": "2"}


def test_example_b_failed_load_restores_diffusion_on_its_card(world):
    world.write(example_b(real_config()))
    world.start(running=["diffusion-host"])
    world.http.never_ready.add("atlas-agent-burst")
    sink = world.handle(world.payload("agent-gpu2", ["gpu2"], profile=ALT_27B))
    assert sink[-1].status == "failed" and sink[-1].restored is True
    assert world.docker.mutations() == [("stop", "diffusion-host"), ("up", "atlas-agent-burst"),
                                        ("stop", "atlas-agent-burst"), ("up", "diffusion-host")]
    restart = world.docker.calls[-1]
    assert restart["env"] == {"CUDA_VISIBLE_DEVICES": "2"}   # from the card index, as the bridge hard-coded
    assert ("diffusion-host", "/v1/lifecycle/drain", {"draining": False}) in world.http.posts


def test_example_b_drain_timeout_undrains_without_stopping(world):
    world.write(example_b(real_config()))
    world.start(running=["diffusion-host"])
    world.http.stuck_in_flight.add("diffusion-host")
    sink = world.handle(world.payload("agent-gpu2", ["gpu2"]))
    assert sink[-1].status == "failed" and sink[-1].reason == "drain_timeout:diffusion"
    assert sink[-1].restored is True
    assert world.docker.mutations() == []
    assert world.http.posts[-1] == ("diffusion-host", "/v1/lifecycle/drain", {"draining": False})


def test_example_b_unload_restores_diffusion(world):
    world.write(example_b(real_config()))
    world.start(running=["atlas-agent-burst"])
    sink = world.handle(world.payload("agent-gpu2", ["gpu2"], action="unload", reason="idle"))
    assert sink[-1].status == "succeeded"
    assert world.docker.mutations() == [("stop", "atlas-agent-burst"), ("up", "diffusion-host")]
    assert ("diffusion-host", "/v1/lifecycle/drain", {"draining": False}) in world.http.posts


# --- refusals and safety -----------------------------------------------------------------------

def test_profile_outside_allow_list_refused_before_any_docker_call(world):
    world.write(example_b(real_config()))
    world.start(running=["diffusion-host"])
    for bad in ("../../etc/passwd", "llama-70b", DEFAULT_27B.upper()):
        bus._task = None
        sink = world.handle(world.payload("agent-gpu2", ["gpu2"], profile=bad, action_id=f"x-{bad}"))
        assert statuses(sink) == [("refused", None)] and sink[0].reason == "profile_not_allowed"
    assert world.docker.calls == []
    assert fence.read_state()["generations"] == {}   # a refusal never spends a generation


def test_unknown_container_state_starts_nothing(world):
    world.write(example_b(real_config()))
    world.start(running=["diffusion-host"])
    world.docker.broken.add("diffusion-host")
    sink = world.handle(world.payload("agent-gpu2", ["gpu2"]))
    assert sink[-1].status == "failed" and sink[-1].reason == "container_state_not_safe:diffusion"
    assert sink[-1].restored is None and world.docker.mutations() == []


def test_seat_and_evicted_both_running_is_a_failure_not_a_noop(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-fast2", "atlas-vision4"])
    sink = world.handle(world.payload("vision4", ["gpu4"], profile=VISION_PROFILE))
    assert sink[-1].reason == "seat_and_evicted_both_running"
    assert world.docker.mutations() == []


def test_evicted_role_on_another_actuator_is_a_named_refusal(world):
    data = example_a(real_config())
    data["actuators"]["circe-b"] = {"host": "circe"}
    data["roles"]["fast2"]["launch"]["actuator"] = "circe-b"
    world.write(data)
    world.start(running=["atlas-fast2"])
    sink = world.handle(world.payload("vision4", ["gpu4"], profile=VISION_PROFILE))
    assert statuses(sink) == [("refused", None)] and sink[0].reason == "no_launch_block:fast2"


def test_resident_role_is_not_directly_loadable(world):
    world.write(example_a(real_config()))
    world.start()
    sink = world.handle(world.payload("fast2", ["gpu4"]))
    assert sink[0].reason == "not_a_swap_seat" and world.docker.calls == []


# --- generation fencing on the generic path ----------------------------------------------------

def test_generic_generation_fence_and_idempotent_replay(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-fast2"])
    first = world.handle(world.payload("vision4", ["gpu4"], profile=VISION_PROFILE, generation=5, action_id="a5"))
    assert first[-1].status == "succeeded"
    assert fence.read_state()["generations"] == {"gpu4": 5}
    calls = len(world.docker.calls)
    replay = world.handle(world.payload("vision4", ["gpu4"], profile=VISION_PROFILE, generation=5, action_id="a5"))
    assert statuses(replay) == [("succeeded", None)] and len(world.docker.calls) == calls
    stale = world.handle(world.payload("vision4", ["gpu4"], action="unload", generation=4, action_id="old"))
    assert stale[0].reason == "stale_generation" and len(world.docker.calls) == calls


def test_checkout_edited_mid_load_stops_forward_progress_but_still_restores(world):
    world.write(example_b(real_config()))
    world.start(running=["diffusion-host"])
    path = world.root / "config" / "gpu_pool.yaml"
    # A `git pull` lands on circe while diffusion drains.
    world.http.on_status = lambda: path.write_text(
        path.read_text().replace("timeout_sec: 0.05", "timeout_sec: 0.06", 1))
    sink = world.handle(world.payload("agent-gpu2", ["gpu2"]))
    assert sink[-1].status == "failed" and sink[-1].reason == "launch_digest_changed"
    assert sink[-1].restored is True
    assert world.docker.mutations() == []            # nothing stopped after the checkpoint
    assert world.http.posts[-1] == ("diffusion-host", "/v1/lifecycle/drain", {"draining": False})


def test_status_observes_every_launch_role(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-fast2", "diffusion-host"])
    world.docker.broken.add("atlas-agent-burst")
    sink = world.handle(world.payload("vision4", ["gpu4"], action="status", launch_digest="x"))
    assert sink[-1].status == "succeeded" and sink[-1].in_flight is False
    assert sink[-1].observed == {"agent-gpu2": "unknown", "diffusion": "running", "fast2": "running",
                                 "vision4": "absent"}


def test_example_configs_pass_the_pool_validator():
    """The worked examples are config-only: they must load with the unchanged pool parser."""
    PoolConfig.model_validate(example_a(real_config()))
    PoolConfig.model_validate(example_b(real_config()))


def test_example_b_plan_names_only_yaml_values():
    cfg = PoolConfig.model_validate(example_b(real_config()))
    plan = fence.resolve(cfg, role="agent-gpu2", action="load", cards=["gpu2"], digest=None, profile=ALT_27B)
    assert isinstance(plan, fence.LaunchPlan)
    assert plan.seat.service == "atlas-agent-burst" and plan.seat.compose == LLAMA
    assert plan.seat.env_text() == ("ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES=2 "
                                    f"ATLAS_AGENT_BURST_PROFILE_NAME={ALT_27B}")
    assert [p.role for p in plan.evicts] == ["diffusion"] and plan.evicts[0].drain is not None
    assert plan.evicts[0].base_url == "http://100.112.254.99:8014"


# --- review follow-ups: multi-evictee order, fence during rollback, unload edges, busy ----------

def example_multi(data: dict) -> dict:
    """gpu4: vision4 evicts two residents -- fast2 (llm) and embed4 (a service with a drain)."""
    data = example_a(data)
    data["roles"]["embed4"] = {
        "kind": "service", "cards": ["gpu4"], "port": 8019, "slots": 1, "vram_gb": 2,
        "launch": {"actuator": "circe", "compose": "services/orion-embed/docker-compose.yml",
                   "service": "embed-host", "cuda_env": "EMBED_CUDA_VISIBLE_DEVICES",
                   "drain": {"set": "/v1/lifecycle/drain", "status": "/v1/lifecycle/status"},
                   "ready": "/ready", "timeout_sec": 0.05}}
    data["roles"]["vision4"]["swap"]["evicts"] = ["fast2", "embed4"]
    return data


def test_multi_evictee_rollback_restarts_in_reverse_stop_order(world):
    world.write(example_multi(real_config()))
    world.start(running=["atlas-fast2", "embed-host"])
    world.http.never_ready.add("atlas-vision4")
    sink = world.handle(world.payload("vision4", ["gpu4"], profile=VISION_PROFILE))
    assert world.docker.mutations() == [("stop", "atlas-fast2"), ("stop", "embed-host"), ("up", "atlas-vision4"),
                                        ("stop", "atlas-vision4"), ("up", "embed-host"), ("up", "atlas-fast2")]
    assert sink[-1].restored is True


def test_drained_but_not_stopped_evictee_is_undrained_on_rollback(world):
    world.write(example_multi(real_config()))
    world.start(running=["atlas-fast2", "embed-host"])
    world.http.stuck_in_flight.add("embed-host")
    sink = world.handle(world.payload("vision4", ["gpu4"], profile=VISION_PROFILE))
    assert sink[-1].reason == "drain_timeout:embed4" and sink[-1].restored is True
    assert world.docker.mutations() == [("stop", "atlas-fast2"), ("stop", "atlas-vision4"), ("up", "atlas-fast2")]
    assert world.http.posts[-1] == ("embed-host", "/v1/lifecycle/drain", {"draining": False})


def test_superseded_generation_during_rollback_reports_not_restored(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-fast2"])
    world.http.never_ready.add("atlas-vision4")
    state_path = world.root.parent / "state" / "fence.json"

    def supersede():   # a newer generation lands while the seat fails to come up
        state = json.loads(state_path.read_text())
        state["generations"]["gpu4"] = 99
        state_path.write_text(json.dumps(state))
    world.http.on_ready = supersede
    sink = world.handle(world.payload("vision4", ["gpu4"], profile=VISION_PROFILE))
    assert sink[-1].restored is False
    assert sink[-1].reason == "model_readiness_timeout:vision4:restoration_failed"
    assert ("up", "atlas-fast2") not in world.docker.mutations()   # the old intent may not restore


def test_unload_evictee_start_failure_is_failed_without_rollback(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-vision4"])
    world.docker.never_start.add("atlas-fast2")
    sink = world.handle(world.payload("vision4", ["gpu4"], action="unload"))
    assert sink[-1].status == "failed" and sink[-1].reason == "startup_failed:fast2"
    assert sink[-1].restored is None
    assert world.docker.mutations() == [("stop", "atlas-vision4"), ("up", "atlas-fast2")]


def test_unload_of_a_draining_seat_undrains_it_when_the_stop_fails(world):
    data = example_a(real_config())
    data["roles"]["vision4"]["launch"]["drain"] = {"set": "/v1/lifecycle/drain", "status": "/v1/lifecycle/status"}
    world.write(data)
    world.start(running=["atlas-vision4"])
    world.docker.stop_fails.add("atlas-vision4")
    sink = world.handle(world.payload("vision4", ["gpu4"], action="unload"))
    assert sink[-1].reason == "stop_failed:vision4"
    assert world.http.posts == [("atlas-vision4", "/v1/lifecycle/drain", {"draining": True}),
                                ("atlas-vision4", "/v1/lifecycle/drain", {"draining": False})]


def test_busy_while_the_generic_lock_is_held(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-fast2"])

    async def scenario():
        sink = []

        async def publish(result, corr):
            sink.append(result)
        async with lx.lock:
            await bus.handle(world.payload("vision4", ["gpu4"], profile=VISION_PROFILE), publish)
        return sink
    sink = asyncio.run(scenario())
    assert [(r.status, r.reason) for r in sink] == [("refused", "busy")]
    assert world.docker.calls == []


def test_no_profile_strips_an_inherited_profile_var(world, monkeypatch):
    monkeypatch.setenv("ATLAS_AGENT_BURST_PROFILE_NAME", "leaked-from-controller-env")
    world.write(example_b(real_config()))
    world.start(running=["diffusion-host"])
    sink = world.handle(world.payload("agent-gpu2", ["gpu2"]))
    assert sink[-1].status == "succeeded"
    assert "ATLAS_AGENT_BURST_PROFILE_NAME" not in world.docker.calls[-1]["full_env"]


def test_config_unloadable_observe_reports_nothing_not_a_fixed_pair(world):
    """5.6: the fixed gpu2 pair fallback is gone with the bridge. An unparseable checkout names no
    roles, so observe() reports none (the pool reconciles), and never touches docker."""
    (world.root / "config").mkdir(exist_ok=True)
    (world.root / "config" / "gpu_pool.yaml").write_text("version: 1\nnot_a_key: true\n")
    world.cfg = load_pool_config(REPO_ROOT / "config" / "gpu_pool.yaml")
    world.start(running=["diffusion-host"])
    assert asyncio.run(bus.observe()) == {}
    assert world.docker.calls == []


def test_unload_rechecks_the_fence_between_idle_check_and_stop(world):
    world.write(example_a(real_config()))
    world.start(running=["atlas-vision4"])
    state_path = world.root.parent / "state" / "fence.json"

    def supersede():
        state = json.loads(state_path.read_text())
        state["generations"]["gpu4"] = 99
        state_path.write_text(json.dumps(state))
    world.http.on_slots = supersede
    sink = world.handle(world.payload("vision4", ["gpu4"], action="unload"))
    assert sink[-1].reason == "stale_or_unknown_intent" and world.docker.mutations() == []


# --- stage 5.3: the committed config/gpu_pool.yaml takes the generic path ------------------------

def test_live_config_load_is_generic_with_the_default_profile(world, monkeypatch):
    """Acceptance 1 (controller side): the committed YAML loads the seat's default profile (stage 7.2:
    Ternary-Bonsai) through launch_exec -- drain and stop diffusion, then `up` atlas-agent-burst with
    the card index and the allow-listed default profile as compose interpolation variables -- and
    never through the gpu2 bridge."""
    world.write(real_config())
    assert world.cfg.load_profile("agent-gpu2") == LIVE_DEFAULT
    world.start(running=["diffusion-host"])
    sink = world.handle(world.payload("agent-gpu2", ["gpu2"], profile=LIVE_DEFAULT))
    assert statuses(sink) == [("accepted", None), ("progress", "draining"), ("progress", "stopping"),
                              ("progress", "starting"), ("progress", "ready_wait"), ("succeeded", None)]
    assert world.docker.mutations() == [("stop", "diffusion-host"), ("up", "atlas-agent-burst")]
    up = world.docker.calls[-1]
    assert up["compose"] == LLAMA and up["profile"] == "agent-burst"
    assert up["args"] == ["-d", "--no-build", "--no-deps"]
    assert {k: v for k, v in up["env"].items() if k.startswith("ATLAS_AGENT_BURST_")} == {
        "ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES": "2", "ATLAS_AGENT_BURST_PROFILE_NAME": LIVE_DEFAULT}
    assert ("diffusion-host", "/v1/lifecycle/drain", {"draining": True}) in world.http.posts
    assert sink[-1].observed == {"agent-gpu2": "running", "diffusion": "exited"}


def test_live_config_unload_refuses_a_busy_seat_with_the_role_suffixed_reason(world, monkeypatch):
    """The bridge's `burst_upstream_not_idle` is now `upstream_not_idle:agent-gpu2`: nothing is
    stopped, and `observed` still says the seat is up (the pool reads that, never the reason)."""
    world.write(real_config())
    world.start(running=["atlas-agent-burst"])
    world.http.busy.add("atlas-agent-burst")
    sink = world.handle(world.payload("agent-gpu2", ["gpu2"], action="unload", reason="idle"))
    assert sink[-1].status == "failed" and sink[-1].reason == "upstream_not_idle:agent-gpu2"
    assert sink[-1].restored is None and world.docker.mutations() == []
    assert sink[-1].observed == {"agent-gpu2": "running", "diffusion": "absent"}


def test_live_config_unload_restores_diffusion_on_card_index_2(world, monkeypatch):
    """Acceptance 2 (controller side): unload stops the seat and starts diffusion with
    CUDA_VISIBLE_DEVICES=2 set by the actuator (the bridge hard-coded it in gpu2.targets())."""
    world.write(real_config())
    world.start(running=["atlas-agent-burst"])
    sink = world.handle(world.payload("agent-gpu2", ["gpu2"], action="unload", reason="idle"))
    assert sink[-1].status == "succeeded"
    assert world.docker.mutations() == [("stop", "atlas-agent-burst"), ("up", "diffusion-host")]
    up = world.docker.calls[-1]
    assert up["env"].get("CUDA_VISIBLE_DEVICES") == "2" and "ATLAS_AGENT_BURST_PROFILE_NAME" not in up["env"]


def test_live_config_refuses_a_profile_outside_its_allow_list(world, monkeypatch):
    """Acceptance 4 (first half), on the committed YAML: a hand-crafted request naming another
    llm_profiles.yaml profile is refused before any docker call or generation is spent."""
    world.write(real_config())
    world.start(running=["diffusion-host"])
    sink = world.handle(world.payload("agent-gpu2", ["gpu2"], profile="qwen3-8b-q4km-v100-16gb-balanced"))
    assert statuses(sink) == [("refused", None)] and sink[0].reason == "profile_not_allowed"
    assert world.docker.calls == [] and fence.read_state()["generations"] == {}
