"""Stage 4.2: pool actuation bridge (app/actuator_bus.py + app/pool_fence.py).

Spec: docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md.
The Docker/HTTP steps are gpu2's own (covered in test_gpu2.py) and are mocked here; these tests pin
who may ask, the fence, the refusals and the published result sequence.
"""
import asyncio
import json
import shutil
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from test_api import REPO_ROOT, main_module
from orion.gpu_pool.config import launch_digest, load_pool_config
from orion.schemas.gpu_pool import GpuActuateResultV1
from orion.schemas.gpu_slot import GpuSlotRequestV1

gpu = main_module.gpu2
bus = main_module.actuator_bus
fence = main_module.pool_fence
settings = main_module.settings


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A private copy of config/gpu_pool.yaml as the controller's own checkout."""
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "config" / "gpu_pool.yaml", root / "config" / "gpu_pool.yaml")
    monkeypatch.setattr(settings, "GPU_LANE_REPO_ROOT", str(root))
    monkeypatch.setattr(settings, "GPU2_POOL_FENCE_STATE_PATH", str(tmp_path / "state" / "fence.json"))
    monkeypatch.setattr(settings, "GPU2_AUTHORITY", "pool")
    monkeypatch.setattr(settings, "GPU2_ENABLED", True)
    monkeypatch.setattr(settings, "GPU_POOL_ACTUATOR_NAME", "circe")
    monkeypatch.setattr(bus, "_task", None)
    monkeypatch.setattr(bus, "_current", None)
    monkeypatch.setattr(bus, "observe", AsyncMock(return_value={"agent-gpu2": "running", "diffusion": "exited"}))
    return root


def digest(root, role="agent-gpu2"):
    return launch_digest(load_pool_config(root / "config" / "gpu_pool.yaml"), role)


def payload(root, **over):
    body = {"action_id": "pool:gpu2:1", "generation": 1, "actuator": "circe", "role": "agent-gpu2",
            "action": "load", "cards": ["gpu2"], "profile": None, "launch_digest": digest(root),
            "deadline_at": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
            "reason": "demand"}
    body.update(over)
    return body


class Sink:
    def __init__(self):
        self.results: list[GpuActuateResultV1] = []

    async def __call__(self, result, corr):
        assert isinstance(result, GpuActuateResultV1)
        self.results.append(result)

    @property
    def statuses(self):
        return [(r.status, r.phase) for r in self.results]


def run(coro_fn):
    """Run handle() and then let the background transition task finish."""
    async def scenario():
        await coro_fn()
        if bus._task is not None:
            await bus._task
    asyncio.run(scenario())


def fake_transition(outcome, phases=("draining", "stopping", "starting", "ready_wait"), seen=None):
    async def transition(req):
        if seen is not None:
            seen.append(req)
        await gpu.authority(req, require_drained=False)  # the real fence, as transition() calls it
        for p in phases:
            await gpu.phase(p)
        return dict(outcome)
    return transition


# --- request validation ------------------------------------------------------------------------

def test_other_actuator_is_ignored_silently(repo):
    sink = Sink()
    run(lambda: bus.handle(payload(repo, actuator="athena"), sink))
    assert sink.results == []


def test_invalid_request_without_actuator_is_not_answered(repo):
    body = payload(repo, cards=[])
    del body["actuator"]
    sink = Sink()
    run(lambda: bus.handle(body, sink))
    assert sink.results == []


def test_naive_deadline_refused_not_crashed(repo, monkeypatch):
    transition = AsyncMock()
    monkeypatch.setattr(gpu, "transition", transition)
    sink = Sink()
    naive = (datetime.now(timezone.utc) + timedelta(minutes=5)).replace(tzinfo=None).isoformat()
    run(lambda: bus.handle(payload(repo, deadline_at=naive), sink))
    assert sink.statuses == [("refused", None)] and sink.results[0].reason == "invalid_request:deadline_at_naive"
    transition.assert_not_called()


def test_invalid_request_is_refused_when_addressable(repo):
    sink = Sink()
    run(lambda: bus.handle(payload(repo, cards=[]), sink))
    assert [r.status for r in sink.results] == ["refused"]
    assert sink.results[0].reason.startswith("invalid_request:")
    assert sink.results[0].action_id == "pool:gpu2:1"


def test_unaddressable_garbage_publishes_nothing(repo):
    sink = Sink()
    run(lambda: bus.handle({"hello": "world"}, sink))
    run(lambda: bus.handle("not a dict", sink))
    assert sink.results == []


# --- authority switch --------------------------------------------------------------------------

def test_default_authority_is_durable_and_refuses_pool_requests(repo, monkeypatch):
    monkeypatch.setattr(settings, "GPU2_AUTHORITY", "durable")
    transition = AsyncMock()
    monkeypatch.setattr(gpu, "transition", transition)
    sink = Sink()
    run(lambda: bus.handle(payload(repo), sink))
    assert sink.statuses == [("refused", None)] and sink.results[0].reason == "authority_durable"
    transition.assert_not_called()
    assert not (repo.parent / "state" / "fence.json").exists()  # nothing persisted, nothing fenced


def test_settings_default_is_durable():
    from pydantic_settings import BaseSettings  # noqa: F401 -- settings class, fresh instance
    assert type(settings)(_env_file=None).GPU2_AUTHORITY == "durable"


def test_durable_authority_uses_durable_callback_not_pool_fence(repo, monkeypatch):
    monkeypatch.setattr(settings, "GPU2_AUTHORITY", "durable")
    pool = AsyncMock()
    monkeypatch.setattr(fence, "authority", pool)
    monkeypatch.setattr(gpu, "request", AsyncMock(return_value={
        "operation_id": "d:1", "generation": 1, "desired_target": "agent-burst",
        "can_transition": True, "activation_eligible": True}))
    req = GpuSlotRequestV1(slot="circe-gpu2", target="agent-burst", operation_id="d:1", generation=1)
    assert asyncio.run(gpu.authority(req))["can_transition"] is True
    pool.assert_not_called()


def test_pool_authority_refuses_http_activate(repo, monkeypatch):
    transition = AsyncMock()
    monkeypatch.setattr(gpu, "transition", transition)
    body = {"slot": "circe-gpu2", "target": "agent-burst", "operation_id": "d:9", "generation": 9}
    response = TestClient(main_module.app).post("/v1/gpu-slots/activate", json=body)
    assert response.status_code == 503 and response.json()["error"] == "authority_pool"
    transition.assert_not_called()


def test_pool_authority_status_never_calls_durable(repo, monkeypatch):
    request = AsyncMock()
    monkeypatch.setattr(gpu, "request", request)
    monkeypatch.setattr(gpu, "snapshots", lambda: {})
    gpu._state.clear()
    gpu._state.update(state="neither", error=None)
    asyncio.run(gpu.status())
    request.assert_not_called()


def test_gpu2_disabled_refuses(repo, monkeypatch):
    monkeypatch.setattr(settings, "GPU2_ENABLED", False)
    sink = Sink()
    run(lambda: bus.handle(payload(repo), sink))
    assert sink.results[0].reason == "gpu2_disabled"


# --- refusal paths -----------------------------------------------------------------------------

@pytest.mark.parametrize("over,reason", [
    ({"role": "nope"}, "unknown_role"),
    ({"role": "agent"}, "role_not_on_this_actuator"),
    ({"launch_digest": "0" * 64}, "launch_digest_mismatch"),
    ({"cards": ["gpu1"]}, "cards_mismatch"),
    ({"profile": "qwen27b"}, "profile_unsupported"),
    ({"role": "diffusion"}, "not_a_bridge_role"),
])
def test_refusals(repo, monkeypatch, over, reason):
    if over.get("role") == "diffusion":
        over = {**over, "launch_digest": digest(repo, "diffusion")}
    transition = AsyncMock()
    monkeypatch.setattr(gpu, "transition", transition)
    sink = Sink()
    run(lambda: bus.handle(payload(repo, **over), sink))
    assert sink.statuses == [("refused", None)]
    assert sink.results[0].reason == reason
    transition.assert_not_called()


def test_deadline_passed_refused(repo, monkeypatch):
    transition = AsyncMock()
    monkeypatch.setattr(gpu, "transition", transition)
    sink = Sink()
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    run(lambda: bus.handle(payload(repo, deadline_at=past), sink))
    assert sink.results[0].reason == "deadline_passed"
    transition.assert_not_called()


def test_digest_follows_own_checkout(repo, monkeypatch):
    """A pool on a different launch block (here: the controller's checkout changed the timeout)
    must be refused -- the controller only starts what its own YAML says."""
    sent = digest(repo)
    path = repo / "config" / "gpu_pool.yaml"
    path.write_text(path.read_text().replace("ready: /health, timeout_sec: 900", "ready: /health, timeout_sec: 901"))
    assert digest(repo) != sent
    sink = Sink()
    run(lambda: bus.handle(payload(repo, launch_digest=sent), sink))
    assert sink.results[0].reason == "launch_digest_mismatch"


def test_unreadable_fence_state_fails_closed(repo, monkeypatch):
    state = repo.parent / "state" / "fence.json"
    state.parent.mkdir(parents=True)
    state.write_text("{not json")
    transition = AsyncMock()
    monkeypatch.setattr(gpu, "transition", transition)
    sink = Sink()
    run(lambda: bus.handle(payload(repo), sink))
    assert sink.results[0].reason.startswith("fence_state_unreadable")
    transition.assert_not_called()


# --- generation fencing ------------------------------------------------------------------------

def test_generation_persisted_before_transition_and_stale_refused(repo, monkeypatch):
    persisted = []

    async def transition(req):
        persisted.append(json.loads((repo.parent / "state" / "fence.json").read_text()))
        return {"status": "success"}
    monkeypatch.setattr(gpu, "transition", transition)
    sink = Sink()
    run(lambda: bus.handle(payload(repo, generation=5, action_id="a5"), sink))
    assert persisted[0]["generations"] == {"gpu2": 5}
    assert persisted[0]["in_flight"]["action_id"] == "a5"
    for gen in (5, 4):
        stale = Sink()
        run(lambda: bus.handle(payload(repo, generation=gen, action_id=f"b{gen}"), stale))
        assert stale.results[0].reason == "stale_generation"
    newer = Sink()
    run(lambda: bus.handle(payload(repo, generation=6, action_id="a6", action="unload"), newer))
    assert newer.results[-1].status == "succeeded"


def test_generation_survives_controller_restart(repo, monkeypatch):
    monkeypatch.setattr(gpu, "transition", AsyncMock(return_value={"status": "success"}))
    run(lambda: bus.handle(payload(repo, generation=3, action_id="a3"), Sink()))
    monkeypatch.setattr(bus, "_task", None)   # a fresh process: only the file remembers
    monkeypatch.setattr(bus, "_current", None)
    sink = Sink()
    run(lambda: bus.handle(payload(repo, generation=2, action_id="old"), sink))
    assert sink.results[0].reason == "stale_generation"


def test_refused_request_does_not_advance_generation(repo, monkeypatch):
    monkeypatch.setattr(gpu, "transition", AsyncMock(return_value={"status": "success"}))
    run(lambda: bus.handle(payload(repo, generation=9, launch_digest="bad"), Sink()))
    sink = Sink()
    run(lambda: bus.handle(payload(repo, generation=2, action_id="ok"), sink))
    assert sink.results[-1].status == "succeeded"


def test_pool_fence_rejects_superseded_in_flight(repo):
    """transition()'s own authority() checkpoints: a request that is not the in-flight generation
    (e.g. a newer one was persisted) must stop before the next mutation."""
    state = fence._empty()
    state["generations"] = {"gpu2": 7}
    state["in_flight"] = {"action_id": "a7", "generation": 7, "role": "agent-gpu2", "action": "load",
                          "cards": ["gpu2"], "launch_digest": digest(repo)}
    fence.write_state(state)
    ok = GpuSlotRequestV1(slot="circe-gpu2", target="agent-burst", operation_id="a7", generation=7)
    assert asyncio.run(gpu.authority(ok))["can_transition"] is True
    old = GpuSlotRequestV1(slot="circe-gpu2", target="agent-burst", operation_id="a6", generation=6)
    with pytest.raises(RuntimeError, match="stale_or_unknown_intent"):
        asyncio.run(gpu.authority(old))
    path = repo / "config" / "gpu_pool.yaml"
    path.write_text(path.read_text().replace("timeout_sec: 900", "timeout_sec: 901"))
    with pytest.raises(RuntimeError, match="launch_digest_changed"):
        asyncio.run(gpu.authority(ok))


def test_busy_while_transition_running(repo, monkeypatch):
    async def scenario():
        release = asyncio.Event()

        async def transition(req):
            await release.wait()
            return {"status": "success"}
        monkeypatch.setattr(gpu, "transition", transition)
        first, second = Sink(), Sink()
        await bus.handle(payload(repo, generation=1, action_id="a1"), first)
        await bus.handle(payload(repo, generation=2, action_id="a2"), second)
        assert second.results[0].reason == "busy"
        replay = Sink()
        await bus.handle(payload(repo, generation=1, action_id="a1"), replay)
        assert replay.statuses == [("progress", None)]   # in-flight replay: no second transition
        release.set()
        await bus._task
        assert first.results[-1].status == "succeeded"
    asyncio.run(scenario())


# --- result publishing sequence ----------------------------------------------------------------

def test_success_sequence_and_real_transition_target(repo, monkeypatch):
    seen = []
    monkeypatch.setattr(gpu, "transition", fake_transition({"status": "success"}, seen=seen))
    sink = Sink()
    run(lambda: bus.handle(payload(repo), sink))
    assert sink.statuses == [("accepted", None), ("progress", "draining"), ("progress", "stopping"),
                             ("progress", "starting"), ("progress", "ready_wait"), ("succeeded", None)]
    assert seen[0].target == "agent-burst" and seen[0].operation_id == "pool:gpu2:1" and seen[0].generation == 1
    final = sink.results[-1]
    assert final.observed == {"agent-gpu2": "running", "diffusion": "exited"}
    assert final.elapsed_ms is not None and final.restored is None
    state = fence.read_state()
    assert state["in_flight"] is None and state["actions"]["pool:gpu2:1"]["status"] == "succeeded"


def test_unload_maps_to_diffusion_restore(repo, monkeypatch):
    seen = []
    monkeypatch.setattr(gpu, "transition", fake_transition({"status": "noop"}, phases=(), seen=seen))
    sink = Sink()
    run(lambda: bus.handle(payload(repo, action="unload", reason="idle"), sink))
    assert seen[0].target == "diffusion"
    assert sink.statuses == [("accepted", None), ("succeeded", None)] and sink.results[-1].reason == "noop"


def test_replayed_action_id_returns_recorded_result_without_second_transition(repo, monkeypatch):
    transition = AsyncMock(return_value={"status": "success"})
    monkeypatch.setattr(gpu, "transition", transition)
    run(lambda: bus.handle(payload(repo), Sink()))
    replay = Sink()
    run(lambda: bus.handle(payload(repo), replay))
    assert replay.statuses == [("succeeded", None)]
    assert transition.await_count == 1


def test_status_republishes_last_result_then_observed(repo, monkeypatch):
    monkeypatch.setattr(gpu, "transition", AsyncMock(return_value={"status": "success"}))
    run(lambda: bus.handle(payload(repo, generation=4, action_id="a4"), Sink()))
    sink = Sink()
    # status is a read: no digest check, no generation fence, nothing persisted.
    run(lambda: bus.handle(payload(repo, action="status", action_id="s1", generation=1,
                                   launch_digest="whatever", reason="reconcile"), sink))
    assert [(r.action_id, r.status) for r in sink.results] == [("a4", "succeeded"), ("s1", "succeeded")]
    assert sink.results[1].observed == {"agent-gpu2": "running", "diffusion": "exited"}
    assert sink.results[1].in_flight is False and sink.results[1].last_action_id == "a4"
    assert "last generation 4" in sink.results[1].reason
    assert fence.read_state()["generations"] == {"gpu2": 4}


def test_restart_mid_action_is_recorded_as_interrupted(repo, monkeypatch):
    state = fence._empty()
    state["generations"] = {"gpu2": 2}
    state["in_flight"] = {"action_id": "a2", "generation": 2, "role": "agent-gpu2", "action": "load",
                          "cards": ["gpu2"], "launch_digest": digest(repo)}
    fence.write_state(state)
    recovered = fence.recover_interrupted()
    assert recovered["reason"] == "interrupted_by_controller_restart"
    assert recovered["restored"] is False   # a cut-off load may have evicted diffusion: fault, not "untouched"
    transition = AsyncMock()
    monkeypatch.setattr(gpu, "transition", transition)
    replay = Sink()
    run(lambda: bus.handle(payload(repo, generation=2, action_id="a2"), replay))
    assert replay.statuses == [("failed", None)]
    assert replay.results[0].reason == "interrupted_by_controller_restart"
    assert replay.results[0].restored is False
    transition.assert_not_called()


# --- rollback on failed readiness --------------------------------------------------------------

def test_failed_readiness_rolls_back_and_reports_restored(repo, monkeypatch):
    """Through the real gpu2.transition(): diffusion drained+stopped, the 27B never gets ready,
    so agent-burst is stopped and diffusion restarted; the pool sees failed + restored=true."""
    calls = []

    async def record(name):
        calls.append(name)
    snap = {"active": "diffusion", "targets": {
        "diffusion": {"state": "running", "containers": [{"state": "running"}]},
        "agent-burst": {"state": "exited", "containers": []}}}
    monkeypatch.setattr(gpu, "status", AsyncMock(return_value=snap))
    monkeypatch.setattr(gpu, "model_ready", AsyncMock(return_value=False))
    monkeypatch.setattr(gpu, "drain_diffusion", lambda: record("drain"))
    monkeypatch.setattr(gpu, "stop", lambda t: record("stop:" + t))

    async def start(target):
        calls.append("start:" + target)
        if target == "agent-burst":
            await gpu.phase("ready_wait")
            raise RuntimeError("model_readiness_timeout")
    monkeypatch.setattr(gpu, "start", start)
    sink = Sink()
    run(lambda: bus.handle(payload(repo), sink))
    assert calls == ["drain", "stop:diffusion", "start:agent-burst", "stop:agent-burst", "start:diffusion"]
    assert sink.statuses == [("accepted", None), ("progress", "draining"), ("progress", "stopping"),
                             ("progress", "starting"), ("progress", "ready_wait"),
                             ("progress", "rolling_back"), ("failed", None)]
    final = sink.results[-1]
    assert final.restored is True and final.reason == "model_readiness_timeout"


def test_failed_rollback_reports_not_restored(repo, monkeypatch):
    snap = {"active": "diffusion", "targets": {
        "diffusion": {"state": "running", "containers": [{"state": "running"}]},
        "agent-burst": {"state": "exited", "containers": []}}}
    monkeypatch.setattr(gpu, "status", AsyncMock(return_value=snap))
    monkeypatch.setattr(gpu, "model_ready", AsyncMock(return_value=False))
    monkeypatch.setattr(gpu, "drain_diffusion", AsyncMock())
    monkeypatch.setattr(gpu, "stop", AsyncMock())
    monkeypatch.setattr(gpu, "start", AsyncMock(side_effect=RuntimeError("model_readiness_timeout")))
    sink = Sink()
    run(lambda: bus.handle(payload(repo), sink))
    final = sink.results[-1]
    assert final.status == "failed" and final.restored is False
    assert final.reason == "model_readiness_timeout:restoration_failed"


def test_failed_unload_never_claims_restored(repo, monkeypatch):
    monkeypatch.setattr(gpu, "transition", AsyncMock(return_value={"status": "failed", "error": "burst_upstream_not_idle",
                                                                   "restored": True}))
    sink = Sink()
    run(lambda: bus.handle(payload(repo, action="unload"), sink))
    assert sink.results[-1].status == "failed" and sink.results[-1].restored is None


def test_progress_publish_failure_does_not_change_outcome(repo, monkeypatch):
    monkeypatch.setattr(gpu, "transition", fake_transition({"status": "success"}))

    class Flaky(Sink):
        async def __call__(self, result, corr):
            if result.status == "progress":
                raise ConnectionError("bus down")
            await super().__call__(result, corr)
    sink = Flaky()
    run(lambda: bus.handle(payload(repo), sink))
    assert [r.status for r in sink.results] == ["accepted", "succeeded"]


def test_real_config_launch_blocks_resolve_to_bridge_targets():
    """The committed YAML is what circe runs: agent-gpu2 load/unload must map onto the two fixed
    gpu2 transitions, and diffusion (evicted resident, no swap verbs) is not directly actuatable."""
    cfg = load_pool_config(REPO_ROOT / "config" / "gpu_pool.yaml")
    assert fence.resolve(cfg, role="agent-gpu2", action="load", cards=["gpu2"], digest=None) == "agent-burst"
    assert fence.resolve(cfg, role="agent-gpu2", action="unload", cards=["gpu2"], digest=None) == "diffusion"
    with pytest.raises(fence.Refusal, match="not_a_bridge_role"):
        fence.resolve(cfg, role="diffusion", action="load", cards=["gpu2"], digest=None)


# --- review regressions ------------------------------------------------------------------------

def test_accepted_publish_failure_still_runs_and_clears_in_flight(repo, monkeypatch):
    transition = AsyncMock(return_value={"status": "success"})
    monkeypatch.setattr(gpu, "transition", transition)

    class DropsAck(Sink):
        async def __call__(self, result, corr):
            if result.status == "accepted":
                raise ConnectionError("bus hiccup")
            await super().__call__(result, corr)
    sink = DropsAck()
    run(lambda: bus.handle(payload(repo), sink))
    transition.assert_awaited_once()
    assert [r.status for r in sink.results] == ["succeeded"]
    assert fence.read_state()["in_flight"] is None and bus._current is None


def test_rollback_not_blocked_by_checkout_edited_mid_load(repo, monkeypatch):
    """git pull on circe during a load changes the digest: the next forward checkpoint stops, but
    rollback (require_drained=False) must still put diffusion back."""
    calls = []
    snap = {"active": "diffusion", "targets": {
        "diffusion": {"state": "running", "containers": [{"state": "running"}]},
        "agent-burst": {"state": "exited", "containers": []}}}
    monkeypatch.setattr(gpu, "status", AsyncMock(return_value=snap))
    monkeypatch.setattr(gpu, "model_ready", AsyncMock(return_value=False))

    async def drain():
        calls.append("drain")
        path = repo / "config" / "gpu_pool.yaml"
        path.write_text(path.read_text().replace("timeout_sec: 900", "timeout_sec: 901"))
    monkeypatch.setattr(gpu, "drain_diffusion", drain)
    monkeypatch.setattr(gpu, "request", AsyncMock(return_value={}))

    async def stop(t):
        calls.append("stop:" + t)
    monkeypatch.setattr(gpu, "stop", stop)
    sink = Sink()
    run(lambda: bus.handle(payload(repo), sink))
    final = sink.results[-1]
    assert "stop:diffusion" not in calls          # forward progress stopped at the checkpoint
    assert final.status == "failed" and final.reason == "launch_digest_changed"
    assert final.restored is True                  # diffusion un-drained, not stranded


def test_finished_result_not_erased_by_concurrent_admission(repo, monkeypatch):
    """B reads the fence, awaits config load; A finishes meanwhile and records its result. B must
    not write its older snapshot back over A's record."""
    async def scenario():
        loop = asyncio.get_running_loop()
        release = asyncio.Event()

        async def transition(req):
            if req.operation_id == "a1":
                await release.wait()
            return {"status": "success"}
        monkeypatch.setattr(gpu, "transition", transition)
        await bus.handle(payload(repo, generation=1, action_id="a1"), Sink())
        task_a = bus._task
        real_load = fence.load_config

        def slow_load():
            loop.call_soon_threadsafe(release.set)
            import time as _t
            _t.sleep(0.3)   # A finishes (and records) while B sits in this await
            return real_load()
        monkeypatch.setattr(fence, "load_config", slow_load)
        second = Sink()
        await bus.handle(payload(repo, generation=2, action_id="b2", action="unload"), second)
        await task_a
        if bus._task is not None:
            await bus._task
        replay = Sink()
        await bus.handle(payload(repo, generation=1, action_id="a1"), replay)
        assert replay.statuses == [("succeeded", None)]
        assert second.results[-1].status == "succeeded"
    asyncio.run(scenario())


def test_status_observes_outside_admit_lock(repo, monkeypatch):
    held = []

    async def observe():
        held.append(bus._admit_lock.locked())
        return {"agent-gpu2": "absent", "diffusion": "running"}
    monkeypatch.setattr(bus, "observe", observe)
    sink = Sink()
    run(lambda: bus.handle(payload(repo, action="status", action_id="s1"), sink))
    assert held == [False] and sink.results[-1].status == "succeeded"


def test_lifespan_runs_one_heartbeat_chassis(monkeypatch):
    """The actuator Hunter already heartbeats; HeartbeatOnly is only its fallback."""
    started = []

    class Fake:
        def __init__(self, name):
            self.name = name

        async def start_background(self):
            started.append(self.name)

        async def stop(self):
            pass
    monkeypatch.setattr(settings, "ORION_BUS_ENABLED", True)
    monkeypatch.setattr(settings, "GPU2_AUTHORITY", "durable")
    monkeypatch.setattr(main_module, "build_actuator_chassis", lambda: Fake("actuator"))
    monkeypatch.setattr(main_module, "build_heartbeat_chassis", lambda: Fake("heartbeat"))
    with TestClient(main_module.app):
        pass
    assert started == ["actuator"]

    started.clear()

    def broken():
        raise RuntimeError("bus down")
    monkeypatch.setattr(main_module, "build_actuator_chassis", broken)
    with TestClient(main_module.app):
        pass
    assert started == ["heartbeat"]


def test_status_reports_in_flight_structurally(repo, monkeypatch):
    async def scenario():
        release = asyncio.Event()

        async def transition(req):
            await gpu.phase("draining")
            await release.wait()
            return {"status": "success"}
        monkeypatch.setattr(gpu, "transition", transition)
        await bus.handle(payload(repo, generation=3, action_id="a3"), Sink())
        await asyncio.sleep(0)
        sink = Sink()
        await bus.handle(payload(repo, action="status", action_id="s1"), sink)
        # While a load runs: no re-published last result, in_flight=True, phase of the running action.
        assert [(r.action_id, r.in_flight, r.phase) for r in sink.results] == [("s1", True, "draining")]
        assert sink.results[0].last_action_id is None
        release.set()
        await bus._task
    asyncio.run(scenario())


def test_status_on_fresh_controller_says_nothing_ran(repo):
    sink = Sink()
    run(lambda: bus.handle(payload(repo, action="status", action_id="s0"), sink))
    assert len(sink.results) == 1
    assert sink.results[0].in_flight is False and sink.results[0].last_action_id is None


def test_non_status_results_never_carry_status_fields(repo, monkeypatch):
    monkeypatch.setattr(gpu, "transition", fake_transition({"status": "success"}))
    sink = Sink()
    run(lambda: bus.handle(payload(repo), sink))
    assert all(r.in_flight is None and r.last_action_id is None for r in sink.results)
