"""Pool actuation intake (app/actuator_bus.py + app/pool_fence.py), stage 4.2 onward.

Spec: docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md.
The Docker/HTTP steps are launch_exec's own (covered in test_launch_exec.py) and are mocked here;
these tests pin who may ask, the fence, the refusals and the published result sequence. Since 5.6
(bridge deleted) they run on the committed config/gpu_pool.yaml.
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi.testclient import TestClient

from test_api import REPO_ROOT, main_module
from orion.gpu_pool.config import launch_digest, load_pool_config
from orion.gpu_pool.config import PoolConfig
from orion.schemas.gpu_pool import GpuActuateResultV1

bus = main_module.actuator_bus
lx = bus.launch_exec
fence = main_module.pool_fence
settings = main_module.settings


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """The controller's own checkout: a copy of the committed config/gpu_pool.yaml."""
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config" / "gpu_pool.yaml").write_text((REPO_ROOT / "config" / "gpu_pool.yaml").read_text())
    monkeypatch.setattr(settings, "GPU_LANE_REPO_ROOT", str(root))
    monkeypatch.setattr(settings, "GPU_POOL_FENCE_STATE_PATH", str(tmp_path / "state" / "fence.json"))
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


def fake_execute(outcome, phases=("draining", "stopping", "starting", "ready_wait"), seen=None):
    async def execute(plan, action, intent):
        if seen is not None:
            seen.append((plan, action, intent))
        await fence.authority(intent, require_drained=False)  # the real fence, as execute() calls it
        for p in phases:
            await lx.phase(p)
        return dict(outcome)
    return execute


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
    monkeypatch.setattr(lx, "execute", transition)
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


# --- the pool is the only gpu2 authority (stage 4.6) -------------------------------------------

def test_the_durable_authority_switch_and_its_callback_url_are_gone():
    fresh = type(settings)(_env_file=None)
    assert not hasattr(fresh, "GPU2_AUTHORITY") and not hasattr(fresh, "GPU2_AUTHORITY_URL")


def test_stage5_6_no_enable_switch_a_launch_block_is_the_only_gate(repo, monkeypatch):
    """GPU2_ENABLED (gpu2_disabled) is gone: a role is actuated iff it has a launch block for this
    actuator; one without is refused by name, never silently run."""
    assert not hasattr(settings, "GPU2_ENABLED")
    execute = AsyncMock(return_value={"status": "success"})
    monkeypatch.setattr(lx, "execute", execute)
    sink = Sink()
    run(lambda: bus.handle(payload(repo), sink))
    assert sink.results[-1].status == "succeeded" and execute.await_count == 1
    refused = Sink()
    run(lambda: bus.handle(payload(repo, role="experiment", action_id="x1", generation=2,
                                   cards=["gpu0", "gpu1", "gpu2", "gpu3"]), refused))
    assert refused.results[0].reason == "role_not_on_this_actuator"


# --- refusal paths -----------------------------------------------------------------------------

@pytest.mark.parametrize("over,reason", [
    ({"role": "nope"}, "unknown_role"),
    ({"role": "agent"}, "role_not_on_this_actuator"),
    ({"launch_digest": "0" * 64}, "launch_digest_mismatch"),
    ({"cards": ["gpu1"]}, "cards_mismatch"),
    ({"profile": "qwen27b"}, "profile_not_allowed"),   # stage 5.2: was profile_unsupported
    ({"role": "diffusion"}, "not_a_swap_seat"),        # stage 5.2: was not_a_bridge_role
])
def test_refusals(repo, monkeypatch, over, reason):
    if over.get("role") == "diffusion":
        over = {**over, "launch_digest": digest(repo, "diffusion")}
    transition = AsyncMock()
    monkeypatch.setattr(lx, "execute", transition)
    sink = Sink()
    run(lambda: bus.handle(payload(repo, **over), sink))
    assert sink.statuses == [("refused", None)]
    assert sink.results[0].reason == reason
    transition.assert_not_called()


def test_deadline_passed_refused(repo, monkeypatch):
    transition = AsyncMock()
    monkeypatch.setattr(lx, "execute", transition)
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
    assert path.read_text().count("timeout_sec: 900") == 1   # agent-gpu2 only
    path.write_text(path.read_text().replace("timeout_sec: 900", "timeout_sec: 901"))
    assert digest(repo) != sent
    sink = Sink()
    run(lambda: bus.handle(payload(repo, launch_digest=sent), sink))
    assert sink.results[0].reason == "launch_digest_mismatch"


def test_unreadable_fence_state_fails_closed(repo, monkeypatch):
    state = repo.parent / "state" / "fence.json"
    state.parent.mkdir(parents=True)
    state.write_text("{not json")
    transition = AsyncMock()
    monkeypatch.setattr(lx, "execute", transition)
    sink = Sink()
    run(lambda: bus.handle(payload(repo), sink))
    assert sink.results[0].reason.startswith("fence_state_unreadable")
    transition.assert_not_called()


# --- generation fencing ------------------------------------------------------------------------

def test_generation_persisted_before_transition_and_stale_refused(repo, monkeypatch):
    persisted = []

    async def transition(plan, action, intent):
        persisted.append(json.loads((repo.parent / "state" / "fence.json").read_text()))
        return {"status": "success"}
    monkeypatch.setattr(lx, "execute", transition)
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
    monkeypatch.setattr(lx, "execute", AsyncMock(return_value={"status": "success"}))
    run(lambda: bus.handle(payload(repo, generation=3, action_id="a3"), Sink()))
    monkeypatch.setattr(bus, "_task", None)   # a fresh process: only the file remembers
    monkeypatch.setattr(bus, "_current", None)
    sink = Sink()
    run(lambda: bus.handle(payload(repo, generation=2, action_id="old"), sink))
    assert sink.results[0].reason == "stale_generation"


def test_refused_request_does_not_advance_generation(repo, monkeypatch):
    monkeypatch.setattr(lx, "execute", AsyncMock(return_value={"status": "success"}))
    run(lambda: bus.handle(payload(repo, generation=9, launch_digest="bad"), Sink()))
    sink = Sink()
    run(lambda: bus.handle(payload(repo, generation=2, action_id="ok"), sink))
    assert sink.results[-1].status == "succeeded"


def test_pool_fence_rejects_superseded_in_flight(repo):
    """execute()'s own authority() checkpoints: a request that is not the in-flight generation
    (e.g. a newer one was persisted) must stop before the next mutation."""
    state = fence._empty()
    state["generations"] = {"gpu2": 7}
    state["in_flight"] = {"action_id": "a7", "generation": 7, "role": "agent-gpu2", "action": "load",
                          "cards": ["gpu2"], "launch_digest": digest(repo)}
    fence.write_state(state)
    ok = lx.Intent("a7", 7)
    assert asyncio.run(fence.authority(ok))["can_transition"] is True
    old = lx.Intent("a6", 6)
    with pytest.raises(RuntimeError, match="stale_or_unknown_intent"):
        asyncio.run(fence.authority(old))
    path = repo / "config" / "gpu_pool.yaml"
    path.write_text(path.read_text().replace("timeout_sec: 900", "timeout_sec: 901"))
    with pytest.raises(RuntimeError, match="launch_digest_changed"):
        asyncio.run(fence.authority(ok))


def test_busy_while_transition_running(repo, monkeypatch):
    async def scenario():
        release = asyncio.Event()

        async def transition(plan, action, intent):
            await release.wait()
            return {"status": "success"}
        monkeypatch.setattr(lx, "execute", transition)
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

def test_success_sequence_and_real_launch_plan(repo, monkeypatch):
    seen = []
    monkeypatch.setattr(lx, "execute", fake_execute({"status": "success"}, seen=seen))
    sink = Sink()
    run(lambda: bus.handle(payload(repo), sink))
    assert sink.statuses == [("accepted", None), ("progress", "draining"), ("progress", "stopping"),
                             ("progress", "starting"), ("progress", "ready_wait"), ("succeeded", None)]
    plan, action, intent = seen[0]
    assert isinstance(plan, fence.LaunchPlan) and plan.seat.role == "agent-gpu2" and action == "load"
    assert [p.role for p in plan.evicts] == ["diffusion"]
    assert intent.operation_id == "pool:gpu2:1" and intent.generation == 1
    final = sink.results[-1]
    assert final.observed == {"agent-gpu2": "running", "diffusion": "exited"}
    assert final.elapsed_ms is not None and final.restored is None
    state = fence.read_state()
    assert state["in_flight"] is None and state["actions"]["pool:gpu2:1"]["status"] == "succeeded"


def test_unload_runs_the_seat_plan_that_restores_diffusion(repo, monkeypatch):
    seen = []
    monkeypatch.setattr(lx, "execute", fake_execute({"status": "noop"}, phases=(), seen=seen))
    sink = Sink()
    run(lambda: bus.handle(payload(repo, action="unload", reason="idle"), sink))
    plan, action, _ = seen[0]
    assert action == "unload" and plan.seat.role == "agent-gpu2" and [p.role for p in plan.evicts] == ["diffusion"]
    assert sink.statuses == [("accepted", None), ("succeeded", None)] and sink.results[-1].reason == "noop"


def test_replayed_action_id_returns_recorded_result_without_second_transition(repo, monkeypatch):
    transition = AsyncMock(return_value={"status": "success"})
    monkeypatch.setattr(lx, "execute", transition)
    run(lambda: bus.handle(payload(repo), Sink()))
    replay = Sink()
    run(lambda: bus.handle(payload(repo), replay))
    assert replay.statuses == [("succeeded", None)]
    assert transition.await_count == 1


def test_status_republishes_last_result_then_observed(repo, monkeypatch):
    monkeypatch.setattr(lx, "execute", AsyncMock(return_value={"status": "success"}))
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
    monkeypatch.setattr(lx, "execute", transition)
    replay = Sink()
    run(lambda: bus.handle(payload(repo, generation=2, action_id="a2"), replay))
    assert replay.statuses == [("failed", None)]
    assert replay.results[0].reason == "interrupted_by_controller_restart"
    assert replay.results[0].restored is False
    transition.assert_not_called()


# --- rollback on failed readiness --------------------------------------------------------------

def test_failed_unload_never_claims_restored(repo, monkeypatch):
    monkeypatch.setattr(lx, "execute", AsyncMock(return_value={"status": "failed", "error": "burst_upstream_not_idle",
                                                                   "restored": True}))
    sink = Sink()
    run(lambda: bus.handle(payload(repo, action="unload"), sink))
    assert sink.results[-1].status == "failed" and sink.results[-1].restored is None


def test_progress_publish_failure_does_not_change_outcome(repo, monkeypatch):
    monkeypatch.setattr(lx, "execute", fake_execute({"status": "success"}))

    class Flaky(Sink):
        async def __call__(self, result, corr):
            if result.status == "progress":
                raise ConnectionError("bus down")
            await super().__call__(result, corr)
    sink = Flaky()
    run(lambda: bus.handle(payload(repo), sink))
    assert [r.status for r in sink.results] == ["accepted", "succeeded"]


def test_real_config_launch_blocks_resolve_to_launch_plans():
    """Stage 5.3: the committed YAML is what circe runs. agent-gpu2 has no bridge verbs, so load and
    unload both resolve to a generic LaunchPlan built from its launch block and diffusion's;
    diffusion (evicted resident) is still not directly actuatable."""
    from orion.gpu_pool.config import PoolConfig
    cfg = load_pool_config(REPO_ROOT / "config" / "gpu_pool.yaml")
    profile = cfg.load_profile("agent-gpu2")
    assert profile == "ternary-bonsai2-27b-pq2-v100-32gb-circe-agent"   # stage 7.2 (Q4 is the 2nd entry)
    for action in ("load", "unload"):
        plan = fence.resolve(cfg, role="agent-gpu2", action=action, cards=["gpu2"], digest=None,
                             profile=profile if action == "load" else None)
        assert isinstance(plan, fence.LaunchPlan)
        assert (plan.seat.service, plan.seat.compose_profile) == ("atlas-agent-burst", "agent-burst")
        assert [p.service for p in plan.evicts] == ["diffusion-host"]
        assert plan.evicts[0].env == {"CUDA_VISIBLE_DEVICES": "2"}   # the bridge's fixed extra_env, now derived
    load = fence.resolve(cfg, role="agent-gpu2", action="load", cards=["gpu2"], digest=None, profile=profile)
    assert load.seat.env == {"ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES": "2", "ATLAS_AGENT_BURST_PROFILE_NAME": profile}
    assert load.seat.timeout_sec == 900.0
    with pytest.raises(fence.Refusal, match="not_a_swap_seat"):
        fence.resolve(cfg, role="diffusion", action="load", cards=["gpu2"], digest=None)
    # 5.6: the fixed gpu2 transitions are gone; the old rollback shape (bridge verbs) no longer parses.
    data = yaml.safe_load((REPO_ROOT / "config" / "gpu_pool.yaml").read_text())
    data["roles"]["agent-gpu2"]["swap"].update(load="gpu2/agent", unload="gpu2/restore")
    with pytest.raises(ValueError, match="Extra inputs"):
        PoolConfig.model_validate(data)
    assert not hasattr(fence, "BRIDGE_TARGETS")


# --- review regressions ------------------------------------------------------------------------

def test_accepted_publish_failure_still_runs_and_clears_in_flight(repo, monkeypatch):
    transition = AsyncMock(return_value={"status": "success"})
    monkeypatch.setattr(lx, "execute", transition)

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


def test_finished_result_not_erased_by_concurrent_admission(repo, monkeypatch):
    """B reads the fence, awaits config load; A finishes meanwhile and records its result. B must
    not write its older snapshot back over A's record."""
    async def scenario():
        loop = asyncio.get_running_loop()
        release = asyncio.Event()

        async def transition(plan, action, intent):
            if intent.operation_id == "a1":
                await release.wait()
            return {"status": "success"}
        monkeypatch.setattr(lx, "execute", transition)
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


def test_lifespan_retries_actuator_start_until_bus_is_up(monkeypatch):
    """A bus that is slow at boot must not leave the controller permanently deaf: retry with a fresh chassis."""
    attempts = []

    class Fake:
        def __init__(self, ok):
            self.ok = ok

        async def start_background(self):
            attempts.append(self.ok)
            if not self.ok:
                raise TimeoutError("bus down")

        async def stop(self):
            pass
    chassis = iter([Fake(False), Fake(False), Fake(True)])
    monkeypatch.setattr(settings, "ORION_BUS_ENABLED", True)
    monkeypatch.setattr(main_module, "BUS_RETRY_DELAY_SEC", 0.01)
    monkeypatch.setattr(main_module, "build_actuator_chassis", lambda: next(chassis))

    async def scenario():
        async with main_module.lifespan(main_module.app):
            for _ in range(200):
                if main_module.actuator_chassis is not None:
                    break
                await asyncio.sleep(0.01)
    run(scenario)
    assert attempts == [False, False, True]


def test_status_reports_in_flight_structurally(repo, monkeypatch):
    async def scenario():
        release = asyncio.Event()

        async def transition(plan, action, intent):
            await lx.phase("draining")
            await release.wait()
            return {"status": "success"}
        monkeypatch.setattr(lx, "execute", transition)
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
    monkeypatch.setattr(lx, "execute", fake_execute({"status": "success"}))
    sink = Sink()
    run(lambda: bus.handle(payload(repo), sink))
    assert all(r.in_flight is None and r.last_action_id is None for r in sink.results)
