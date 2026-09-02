"""Tests for app/lane_control.py -- the actual GPU1 flip mechanism.

Mocks SafeCommandRunner entirely (no real docker calls) via a fake runner
class that branches on command content, mirroring
services/orion-cortex-exec/tests/test_docker_compose_service_bringup.py's
mocking convention for the same kind of subprocess-shelling code.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = SERVICE_DIR / "app"
PACKAGE_NAME = "orion_gpu_lane_controller"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"
if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(SERVICE_DIR)]
    sys.modules[PACKAGE_NAME] = pkg
if APP_PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(APP_PACKAGE_NAME)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[APP_PACKAGE_NAME] = pkg

import importlib.util

REPO_ROOT = SERVICE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{APP_PACKAGE_NAME}.{name}", APP_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


settings_module = _load("settings")
sys.modules[f"{APP_PACKAGE_NAME}.settings"] = settings_module
lane_control = _load("lane_control")


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    services_dir = tmp_path / "services"
    affect_dir = services_dir / "orion-affectgpt-worker"
    affect_dir.mkdir(parents=True)
    (affect_dir / "docker-compose.yml").write_text("services: {}\n")
    agent_dir = services_dir / "orion-llamacpp-host"
    agent_dir.mkdir(parents=True)
    (agent_dir / "docker-compose.atlas-workers.yml").write_text("services: {}\n")

    monkeypatch.setattr(lane_control.settings, "GPU_LANE_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(lane_control.settings, "GPU_LANE_COMMAND_TIMEOUT_SEC", 5.0)
    monkeypatch.setattr(lane_control.settings, "GPU_LANE_HEALTH_POLL_SEC", 0.0)
    return tmp_path


def _row(cid="abc123def456", name="c", state="running", health=None):
    d = {"ID": cid, "Name": name, "State": state}
    if health is not None:
        d["Health"] = health
    import json

    return json.dumps(d)


def _install_fake_runner(monkeypatch, *, ps_stdout_by_service=None, stop_rc=0, build_rc=0, up_rc=0):
    ps_stdout_by_service = ps_stdout_by_service or {}
    calls: list[list[str]] = []

    class _FakeRunner:
        def __init__(self, *, allowed_commands, timeout_sec):
            self.allowed_commands = allowed_commands
            self.timeout_sec = timeout_sec

        def run(self, command, *, cwd=None, env=None):
            calls.append(list(command))
            if "ps" in command:
                service = next((s for s in ps_stdout_by_service if s in command), None)
                return SimpleNamespace(returncode=0, stdout=ps_stdout_by_service.get(service, ""), stderr="")
            if "stop" in command:
                return SimpleNamespace(returncode=stop_rc, stdout="", stderr="" if stop_rc == 0 else "stop failed")
            if "build" in command:
                return SimpleNamespace(returncode=build_rc, stdout="", stderr="" if build_rc == 0 else "build failed")
            if "up" in command:
                return SimpleNamespace(returncode=up_rc, stdout="", stderr="" if up_rc == 0 else "up failed")
            return SimpleNamespace(returncode=1, stdout="", stderr="unrecognized command in test double")

    monkeypatch.setattr(lane_control, "SafeCommandRunner", _FakeRunner)
    return calls


def test_status_neither_running(fake_repo, monkeypatch):
    _install_fake_runner(monkeypatch, ps_stdout_by_service={})
    status = lane_control.get_status()
    assert status["active"] == "neither"
    assert status["affect"]["running"] is False
    assert status["agent"]["running"] is False


def test_status_agent_running(fake_repo, monkeypatch):
    _install_fake_runner(
        monkeypatch,
        ps_stdout_by_service={"atlas-agent": _row(state="running", health="healthy")},
    )
    status = lane_control.get_status()
    assert status["active"] == "agent"
    assert status["agent"]["running"] is True
    assert status["affect"]["running"] is False


def test_status_both_running_is_detected_not_hidden(fake_repo, monkeypatch):
    _install_fake_runner(
        monkeypatch,
        ps_stdout_by_service={
            "atlas-agent": _row(state="running", health="healthy"),
            "affectgpt-worker": _row(state="running"),
        },
    )
    status = lane_control.get_status()
    assert status["active"] == "both"


def test_flip_noop_when_already_sole_target(fake_repo, monkeypatch):
    calls = _install_fake_runner(
        monkeypatch,
        ps_stdout_by_service={"atlas-agent": _row(state="running", health="healthy")},
    )
    result = asyncio.run(lane_control.flip("agent"))
    assert result["status"] == "noop"
    # Only the two status-check `ps` calls -- no stop/build/up was ever invoked.
    assert not any("stop" in c or "build" in c or "up" in c for c in calls)


def test_flip_success_stops_other_and_brings_target_up(fake_repo, monkeypatch):
    calls = _install_fake_runner(
        monkeypatch,
        ps_stdout_by_service={
            "affectgpt-worker": _row(state="running"),
            "atlas-agent": _row(state="running", health="healthy"),
        },
    )
    result = asyncio.run(lane_control.flip("agent"))
    assert result["status"] == "success"
    assert result["stop_other"]["ok"] is True
    assert result["bring_up"]["ok"] is True
    assert any("stop" in c and "affectgpt-worker" in c for c in calls)
    assert any("up" in c and "atlas-agent" in c for c in calls)
    # Every command against the 4-service atlas-workers compose file names
    # atlas-agent explicitly -- never a bare stop/build/up.
    for c in calls:
        if "orion-llamacpp-host/docker-compose.atlas-workers.yml" in " ".join(c):
            assert "atlas-agent" in c


def test_flip_stop_failure_short_circuits_before_bringup(fake_repo, monkeypatch):
    calls = _install_fake_runner(monkeypatch, ps_stdout_by_service={}, stop_rc=1)
    result = asyncio.run(lane_control.flip("agent"))
    assert result["status"] == "stop_failed"
    assert not any("build" in c or "up" in c for c in calls)


def test_flip_build_failure_leaves_both_down_honestly(fake_repo, monkeypatch):
    _install_fake_runner(monkeypatch, ps_stdout_by_service={}, build_rc=1)
    result = asyncio.run(lane_control.flip("agent"))
    assert result["status"] == "build_failed"
    assert "may now be idle" in result["user_facing_summary"]


def test_flip_up_failure_after_build_ok(fake_repo, monkeypatch):
    _install_fake_runner(monkeypatch, ps_stdout_by_service={}, up_rc=1)
    result = asyncio.run(lane_control.flip("agent"))
    assert result["status"] == "up_failed"


def test_flip_unhealthy_when_never_settles(fake_repo, monkeypatch):
    # up "succeeds" (rc=0) but ps never reports a running container for
    # atlas-agent -- poll window is 0s (fixture), so this returns fast.
    _install_fake_runner(monkeypatch, ps_stdout_by_service={})
    result = asyncio.run(lane_control.flip("agent"))
    assert result["status"] == "unhealthy"
    assert result["settled"]["running"] is False


def test_flip_invalid_target(fake_repo, monkeypatch):
    _install_fake_runner(monkeypatch)
    result = asyncio.run(lane_control.flip("bogus"))
    assert result["status"] == "invalid_target"


def test_flip_returns_busy_without_racing_when_already_in_progress(fake_repo, monkeypatch):
    calls = _install_fake_runner(monkeypatch, ps_stdout_by_service={})

    async def _run():
        await lane_control._FLIP_LOCK.acquire()
        try:
            return await lane_control.flip("agent")
        finally:
            lane_control._FLIP_LOCK.release()

    result = asyncio.run(_run())
    assert result["status"] == "busy"
    # Held lock means _flip_locked's body never ran -- no docker commands at all.
    assert calls == []


def test_agent_invocation_forces_gpu1_env_override(fake_repo, monkeypatch):
    seen_envs: list[dict] = []

    class _EnvCapturingRunner:
        def __init__(self, *, allowed_commands, timeout_sec):
            pass

        def run(self, command, *, cwd=None, env=None):
            if env is not None:
                seen_envs.append(env)
            if "ps" in command:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lane_control, "SafeCommandRunner", _EnvCapturingRunner)
    monkeypatch.setattr(lane_control.settings, "AGENT_GPU1_CUDA_VISIBLE_DEVICES", "1")
    asyncio.run(lane_control.flip("agent"))
    assert any(e.get("ATLAS_AGENT_CUDA_VISIBLE_DEVICES") == "1" for e in seen_envs)
