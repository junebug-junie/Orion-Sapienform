"""GPU actuation watch: pure evaluation (app/gpu_watch.py) and the service wiring."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from orion.gpu_pool.actuator_probe import Verdict, classify

from app import gpu_watch
from app.gpu_watch import (
    NO_ANSWER_STREAK,
    REFUSAL_BURST,
    REFUSAL_WINDOW_SEC,
    ActiveProbeTracker,
    RefusalWatch,
    RoleTarget,
    role_targets,
)
from app.service import MeshGuardianService
from app.settings import Settings
from app.stability import REALERT_AFTER_SEC, AlertGate

T0 = 1_791_590_000.0  # ~2026-10-09
TARGET = RoleTarget(role="agent-gpu2", actuator="circe", host="circe")
TARGETS = {"agent-gpu2": TARGET, "diffusion": RoleTarget("diffusion", "circe", "circe")}
REPO = Path(__file__).resolve().parents[3]


def _v(check: str, status: str | None, reason: str | None = None) -> Verdict:
    results = [] if status is None else [{"status": status, "reason": reason}]
    return classify(check, results, role="agent-gpu2")


def _refused(reason: str, role: str = "agent-gpu2") -> dict:
    return {"event": "actuate_refused", "role": role, "reason": reason}


class TestActive:
    def test_incident_status_config_unloadable_is_one_critical_naming_host_and_fix(self) -> None:
        alerts = ActiveProbeTracker().observe(TARGET, _v("status", "refused", "config_unloadable:ValidationError"))
        assert [(a.kind, a.severity, a.key) for a in alerts] == [
            ("gpu_controller_config_unloadable", "critical", "gpu_config_unloadable:agent-gpu2")]
        msg = alerts[0].message
        assert "circe" in msg and "agent-gpu2" in msg
        assert "rebuild orion-gpu-lane-controller on circe from main" in msg
        assert "predates config/gpu_pool.yaml" in msg

    def test_digest_mismatch_is_error(self) -> None:
        alerts = ActiveProbeTracker().observe(TARGET, _v("digest", "refused", "launch_digest_mismatch"))
        assert [(a.kind, a.severity) for a in alerts] == [("gpu_controller_digest_mismatch", "error")]

    def test_healthy_and_transient_refusals_do_not_alert(self) -> None:
        tracker = ActiveProbeTracker()
        assert tracker.observe(TARGET, _v("status", "succeeded")) == []
        assert tracker.observe(TARGET, _v("digest", "refused", "profile_not_allowed")) == []
        assert tracker.observe(TARGET, _v("digest", "refused", "busy")) == []
        assert tracker.observe(TARGET, _v("digest", "refused", "deadline_passed")) == []

    def test_no_answer_alerts_only_on_second_consecutive_cycle(self) -> None:
        assert NO_ANSWER_STREAK == 2
        tracker = ActiveProbeTracker()
        assert tracker.observe(TARGET, _v("status", None)) == []
        alerts = tracker.observe(TARGET, _v("status", None))
        assert [(a.kind, a.severity) for a in alerts] == [("gpu_controller_no_answer", "error")]
        assert "not answering" in alerts[0].message and "circe" in alerts[0].message

    def test_persistent_refusal_alerts_on_second_cycle_transients_never(self) -> None:
        tracker = ActiveProbeTracker()
        assert tracker.observe(TARGET, _v("digest", "refused", "role_not_on_this_actuator")) == []
        alerts = tracker.observe(TARGET, _v("digest", "refused", "role_not_on_this_actuator"))
        assert [(a.key, a.severity) for a in alerts] == [("gpu_controller_refusing:agent-gpu2", "error")]
        for _ in range(3):
            assert tracker.observe(TARGET, _v("digest", "refused", "busy")) == []

    def test_a_transient_resets_the_refusing_streak(self) -> None:
        tracker = ActiveProbeTracker()
        assert tracker.observe(TARGET, _v("digest", "refused", "cards_mismatch")) == []
        assert tracker.observe(TARGET, _v("digest", "refused", "busy")) == []
        assert tracker.observe(TARGET, _v("digest", "refused", "cards_mismatch")) == []

    def test_probe_digest_mismatch_does_not_blame_only_the_controller(self) -> None:
        alerts = ActiveProbeTracker().observe(TARGET, _v("digest", "refused", "launch_digest_mismatch"))
        assert "rebuild orion-mesh-guardian" in alerts[0].message

    def test_an_answer_resets_the_no_answer_streak(self) -> None:
        tracker = ActiveProbeTracker()
        assert tracker.observe(TARGET, _v("status", None)) == []
        assert tracker.observe(TARGET, _v("status", "succeeded")) == []
        assert tracker.observe(TARGET, _v("status", None)) == []


class TestPassive:
    def test_first_config_unloadable_refusal_is_critical(self) -> None:
        alerts = RefusalWatch().observe(_refused("config_unloadable:ValidationError"), T0, TARGETS)
        assert [(a.key, a.severity) for a in alerts] == [("gpu_config_unloadable:agent-gpu2", "critical")]
        assert alerts[0].context["latest_reason"] == "config_unloadable:ValidationError"
        assert alerts[0].context["refusals_in_window"] == 1

    def test_digest_mismatch_refusal_is_critical(self) -> None:
        alerts = RefusalWatch().observe(_refused("launch_digest_mismatch"), T0, TARGETS)
        assert [(a.key, a.severity) for a in alerts] == [("gpu_digest_mismatch:agent-gpu2", "critical")]

    def test_burst_of_ordinary_refusals_alerts_at_threshold(self) -> None:
        assert REFUSAL_BURST == 3
        watch = RefusalWatch()
        assert watch.observe(_refused("busy"), T0, TARGETS) == []
        assert watch.observe(_refused("busy"), T0 + 60, TARGETS) == []
        alerts = watch.observe(_refused("stale_generation"), T0 + 120, TARGETS)
        assert [(a.kind, a.severity) for a in alerts] == [("gpu_actuate_refusal_burst", "error")]
        assert alerts[0].context["refusals_in_window"] == 3
        assert alerts[0].context["latest_reason"] == "stale_generation"

    def test_refusals_spread_past_the_window_do_not_alert(self) -> None:
        watch = RefusalWatch()
        step = REFUSAL_WINDOW_SEC / 2 + 1
        for i in range(6):
            assert watch.observe(_refused("busy"), T0 + i * step, TARGETS) == []

    def test_window_is_per_role_and_ignores_other_events(self) -> None:
        watch = RefusalWatch()
        assert watch.observe(_refused("busy", "agent-gpu2"), T0, TARGETS) == []
        assert watch.observe(_refused("busy", "diffusion"), T0 + 1, TARGETS) == []
        assert watch.observe({"event": "swapped", "role": "agent-gpu2"}, T0 + 2, TARGETS) == []
        assert watch.observe(_refused("busy", "agent-gpu2"), T0 + 3, TARGETS) == []

    def test_incident_155_refusals_make_one_card_per_gate_window(self) -> None:
        """The real incident: 155 config_unloadable refusals over 28 h, all role agent-gpu2."""
        watch, gate = RefusalWatch(), AlertGate()
        span = 28 * 3600
        cards = []
        for i in range(155):
            now = T0 + i * span / 154
            cards += gate.admit(watch.observe(_refused("config_unloadable:ValidationError"), now, TARGETS), now)
        windows = int(span // REALERT_AFTER_SEC) + 1
        assert len(cards) == windows  # 5 cards over 28 h at a 6 h re-alert, not 155
        assert all(c.key == "gpu_config_unloadable:agent-gpu2" for c in cards)


def test_role_targets_reads_launch_blocks_from_the_live_config() -> None:
    from orion.gpu_pool.config import load_pool_config

    targets = role_targets(load_pool_config(REPO / "config" / "gpu_pool.yaml"))
    assert set(targets) == {"agent-gpu2", "diffusion"}
    assert targets["agent-gpu2"] == RoleTarget("agent-gpu2", "circe", "circe")


# --- service wiring ---------------------------------------------------------


class FakePubSub:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=0.01)
        except asyncio.TimeoutError:
            return None


class FakeBus:
    """Answers actuate requests from ``answers[(role, action)]`` -> (status, reason) | None."""

    def __init__(self, answers: dict) -> None:
        self.answers = answers
        self.published: list[tuple[str, dict]] = []
        self._subs: list[FakePubSub] = []
        self.redis = None

    @asynccontextmanager
    async def subscribe(self, *channels):
        ps = FakePubSub()
        self._subs.append(ps)
        yield ps

    async def publish(self, channel, env) -> None:
        import json

        req = env.payload
        self.published.append((channel, req))
        answer = self.answers.get((req["role"], req["action"]))
        if isinstance(answer, Exception):
            raise answer
        if answer is None:
            return
        status, reason = answer
        res = {"action_id": req["action_id"], "status": status, "reason": reason}
        for ps in self._subs:
            ps.queue.put_nowait({"data": json.dumps({"payload": res})})


def _service(answers: dict) -> tuple[MeshGuardianService, list[dict]]:
    settings = Settings(ORION_REPO_ROOT=str(REPO), MESH_GUARDIAN_GPU_PROBE_WAIT_SEC=0.05)
    service = MeshGuardianService(settings)
    service.bus = FakeBus(answers)
    cards: list[dict] = []
    service.attention.publish_transition = lambda **kw: cards.append(kw)
    return service, cards


HEALTHY = {("agent-gpu2", "status"): ("succeeded", None), ("agent-gpu2", "load"): ("refused", "profile_not_allowed"),
           ("diffusion", "status"): ("succeeded", None), ("diffusion", "load"): ("refused", "profile_not_allowed")}


@pytest.mark.asyncio
async def test_gpu_probe_cycle_publishes_one_critical_card_through_the_gate() -> None:
    answers = {**HEALTHY, ("agent-gpu2", "load"): ("refused", "config_unloadable:ValidationError")}
    service, cards = _service(answers)
    await service.run_gpu_probes(T0)
    assert len(service.bus.published) == 2  # 2 roles x digest
    assert [(c["service_id"], c["event"]["severity"]) for c in cards] == [("gpu-lane:agent-gpu2", "critical")]
    assert cards[0]["heartbeat_name"] == "gpu_watch"
    assert cards[0]["event"]["context"]["event"] == "gpu_controller_config_unloadable"
    await service.run_gpu_probes(T0 + 300)
    assert len(cards) == 1  # same key inside the AlertGate window


@pytest.mark.asyncio
async def test_healthy_cycle_is_silent() -> None:
    service, cards = _service(HEALTHY)
    assert await service.run_gpu_probes(T0) == []
    assert cards == []


@pytest.mark.asyncio
async def test_one_failing_probe_does_not_skip_the_others() -> None:
    answers = {**HEALTHY, ("agent-gpu2", "load"): RuntimeError("bus hiccup"),
               ("diffusion", "load"): ("refused", "launch_digest_mismatch")}
    service, cards = _service(answers)
    await service.run_gpu_probes(T0)
    assert len(service.bus.published) == 2
    assert [c["event"]["context"]["event"] for c in cards] == ["gpu_controller_digest_mismatch"]


@pytest.mark.asyncio
async def test_gpu_loop_survives_a_cycle_that_raises() -> None:
    service, _ = _service(HEALTHY)
    calls = 0

    async def boom(now):
        nonlocal calls
        calls += 1
        raise RuntimeError("cycle blew up")

    async def stop_after_two(_delay):
        if calls >= 2:
            service._stop.set()

    service.run_gpu_probes = boom
    with patch("app.service.asyncio.sleep", side_effect=stop_after_two):
        await asyncio.wait_for(service._gpu_loop(), timeout=2)
    assert calls == 2


@pytest.mark.asyncio
async def test_pool_refusal_event_and_active_probe_share_one_card() -> None:
    answers = {**HEALTHY, ("agent-gpu2", "load"): ("refused", "config_unloadable:ValidationError")}
    service, cards = _service(answers)
    await service.handle_gpu_pool_event(_refused("config_unloadable:ValidationError"), T0)
    assert len(cards) == 1 and cards[0]["event"]["context"]["source"] == "pool_event"
    assert cards[0]["event"]["context"]["host"] == "circe"  # targets loaded lazily from live config
    await service.run_gpu_probes(T0 + 60)
    assert len(cards) == 1


@pytest.mark.asyncio
async def test_unloadable_live_config_raises_a_guardian_card(tmp_path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "gpu_pool.yaml").write_text("roles: {x: {unknown_field: 1}}\n")
    service, cards = _service(HEALTHY)
    service.settings = Settings(ORION_REPO_ROOT=str(tmp_path), MESH_GUARDIAN_GPU_PROBE_WAIT_SEC=0.05)
    assert await service.run_gpu_probes(T0) == []
    assert service.bus.published == []
    assert [c["event"]["context"]["event"] for c in cards] == ["gpu_watch_config_unloadable"]
    assert "rebuild orion-mesh-guardian" in cards[0]["event"]["message"]


@pytest.mark.asyncio
async def test_stability_cycle_still_publishes_through_shared_path() -> None:
    from app.stability import StabilityAlert

    service, cards = _service(HEALTHY)

    async def one_alert(now):
        return [StabilityAlert(key="k", subject="s", kind="crash_loop", severity="error", message="m")], {}

    service._check_crash_loops = one_alert

    async def nothing(now):
        return [], {}

    service._check_bus_redis = nothing
    service._check_falkordb = nothing
    await service.run_stability_checks(T0)
    assert [(c["service_id"], c["heartbeat_name"]) for c in cards] == [("s", "stability")]


@pytest.mark.asyncio
async def test_guardian_never_sends_status_probes() -> None:
    """status makes the controller replay its last result under the pool's action_id, which can
    flip the pool's belief (late_result); the periodic watch sends only the digest load."""
    from orion.gpu_pool.actuator_probe import PROBE_PROFILE

    service, _ = _service(HEALTHY)
    await service.run_gpu_probes(T0)
    assert service.bus.published
    assert {(r["action"], r["profile"]) for _, r in service.bus.published} == {("load", PROBE_PROFILE)}


def test_alert_gate_escalates_severity_inside_the_window() -> None:
    from app.stability import StabilityAlert

    gate = AlertGate()
    err = StabilityAlert(key="k", subject="s", kind="x", severity="error", message="m")
    crit = StabilityAlert(key="k", subject="s", kind="x", severity="critical", message="m")
    assert gate.admit([err], T0) == [err]
    assert gate.admit([err], T0 + 60) == []
    assert gate.admit([crit], T0 + 120) == [crit]
    assert gate.admit([crit, err], T0 + 180) == []


@pytest.mark.asyncio
async def test_unparseable_config_is_retried_at_most_once_per_interval(tmp_path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "gpu_pool.yaml").write_text("roles: {x: {unknown_field: 1}}\n")
    service, _ = _service(HEALTHY)
    service.settings = Settings(ORION_REPO_ROOT=str(tmp_path))
    loads = 0
    real = service._load_gpu_config

    async def counting(now):
        nonlocal loads
        loads += 1
        return await real(now)

    service._load_gpu_config = counting
    for i in range(20):
        await service.handle_gpu_pool_event(_refused("busy"), T0 + i)
    assert loads == 1
    await service.handle_gpu_pool_event(_refused("busy"), T0 + service.settings.gpu_probe_interval_sec)
    assert loads == 2


def test_settings_ship_gpu_watch_on() -> None:
    fields = Settings.model_fields
    assert fields["gpu_watch_enabled"].default is True
    assert fields["gpu_probe_interval_sec"].default == 300
    assert fields["gpu_probe_wait_sec"].default == 90.0
    assert gpu_watch.CONTROLLER_SERVICE == "orion-gpu-lane-controller"
