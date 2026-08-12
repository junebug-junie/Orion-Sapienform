"""Tests for skills.docker.compose_service_bringup.v1 (DockerComposeServiceBringupVerb).

Covers: service discovery/validation, policy gate, mocked build/up success and failure paths, and
health-poll classification (healthy / running_no_healthcheck / unhealthy). Mirrors the module-loading
and SafeCommandRunner-mocking conventions already used by test_skill_verbs.py.
"""

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = SERVICE_DIR / "app"
PACKAGE_NAME = "orion_cortex_exec"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"
if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(SERVICE_DIR)]
    sys.modules[PACKAGE_NAME] = pkg
if APP_PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(APP_PACKAGE_NAME)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[APP_PACKAGE_NAME] = pkg
spec = importlib.util.spec_from_file_location(f"{APP_PACKAGE_NAME}.verb_adapters", APP_DIR / "verb_adapters.py")
verb_adapters = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = verb_adapters
spec.loader.exec_module(verb_adapters)

REPO_ROOT = SERVICE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.verbs.base import VerbContext  # noqa: E402
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest  # noqa: E402


def _plan_request(*, skill_args: dict | None = None) -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(verb_name="skills.docker.compose_service_bringup.v1", steps=[]),
        args=PlanExecutionArgs(request_id=str(uuid4()), extra={"skill_args": skill_args or {}}),
        context={"metadata": {}},
    )


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    services_dir = tmp_path / "services"
    (services_dir / "orion-widget").mkdir(parents=True)
    (services_dir / "orion-widget" / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / ".env").write_text("FOO=bar\n")
    monkeypatch.setattr(verb_adapters.self_study_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(verb_adapters.settings, "skills_allow_docker_compose_bringup", True)
    monkeypatch.setattr(verb_adapters.settings, "skills_docker_compose_bringup_timeout_sec", 5.0)
    monkeypatch.setattr(verb_adapters.settings, "skills_docker_compose_bringup_health_poll_sec", 5.0)
    monkeypatch.setattr(verb_adapters, "_HEALTH_POLL_CONFIRM_DELAY_SEC", 0)
    return tmp_path


def _run(payload: PlanExecutionRequest):
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, effects = asyncio.run(verb_adapters.DockerComposeServiceBringupVerb().execute(ctx, payload))
    return json.loads(out.final_text), out


def test_policy_gate_blocks_when_disabled(fake_repo, monkeypatch):
    monkeypatch.setattr(verb_adapters.settings, "skills_allow_docker_compose_bringup", False)
    data, out = _run(_plan_request(skill_args={"service": "orion-widget"}))
    assert data["status"] == "blocked"
    assert data["policy_blocked"] is True
    assert out.ok is False


def test_rejects_path_traversal_service_name(fake_repo):
    data, out = _run(_plan_request(skill_args={"service": "../etc"}))
    assert data["status"] == "not_found"
    assert out.ok is False


def test_rejects_service_name_with_slash(fake_repo):
    data, out = _run(_plan_request(skill_args={"service": "a/b"}))
    assert data["status"] == "not_found"
    assert out.ok is False


def test_unknown_service_returns_not_found(fake_repo):
    data, out = _run(_plan_request(skill_args={"service": "does-not-exist"}))
    assert data["status"] == "not_found"
    assert out.ok is False
    assert "available_services" in data
    assert "orion-widget" in data["available_services"]
    assert data["diagnostics"]["repo_root_exists"] is True
    assert data["diagnostics"]["services_dir_exists"] is True
    assert data["diagnostics"]["compose_file_exists"] is False


def test_denylisted_service_is_blocked(fake_repo):
    (fake_repo / "services" / "orion-cortex-exec").mkdir(parents=True)
    (fake_repo / "services" / "orion-cortex-exec" / "docker-compose.yml").write_text("services: {}\n")
    data, out = _run(_plan_request(skill_args={"service": "orion-cortex-exec"}))
    assert data["status"] == "blocked"
    assert data["policy_blocked"] is True
    assert out.ok is False


def _container_row(cid: str, name: str) -> str:
    return json.dumps({"ID": cid, "Name": name})


def _runner_factory(*, build_rc=0, up_rc=0, ps_stdout="", inspect_body=None):
    inspect_body = inspect_body if inspect_body is not None else []

    class _Runner:
        def __init__(self, **kwargs):
            pass

        def run(self, command, cwd=None, env=None):
            if "build" in command and command[:2] == ["docker", "compose"]:
                return SimpleNamespace(returncode=build_rc, stdout="Building...\n", stderr="")
            if "up" in command and command[:2] == ["docker", "compose"]:
                return SimpleNamespace(returncode=up_rc, stdout="Starting...\n", stderr="")
            if "ps" in command and command[:2] == ["docker", "compose"]:
                return SimpleNamespace(returncode=0, stdout=ps_stdout, stderr="")
            if command[:3] == ["docker", "container", "inspect"]:
                return SimpleNamespace(returncode=0, stdout=json.dumps(inspect_body), stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr=f"unexpected:{command!r}")

    return _Runner


def test_build_failure_short_circuits(fake_repo, monkeypatch):
    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _runner_factory(build_rc=1))
    data, out = _run(_plan_request(skill_args={"service": "orion-widget"}))
    assert data["status"] == "build_failed"
    assert data["build_exit_code"] == 1
    assert "up_exit_code" not in data
    assert out.ok is False


def test_up_failure_after_build_success(fake_repo, monkeypatch):
    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _runner_factory(build_rc=0, up_rc=1))
    data, out = _run(_plan_request(skill_args={"service": "orion-widget"}))
    assert data["status"] == "up_failed"
    assert data["build_exit_code"] == 0
    assert data["up_exit_code"] == 1
    assert out.ok is False


def test_healthy_classification(fake_repo, monkeypatch):
    ps_stdout = _container_row("abc123", "orion-widget-1") + "\n"
    inspect_body = [
        {
            "Id": "abc123",
            "State": {"Status": "running", "Health": {"Status": "healthy"}},
            "Config": {"Healthcheck": {"Test": ["CMD", "curl", "-f", "http://localhost/health"]}},
        }
    ]
    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _runner_factory(ps_stdout=ps_stdout, inspect_body=inspect_body))
    data, out = _run(_plan_request(skill_args={"service": "orion-widget"}))
    assert data["status"] == "healthy"
    assert out.ok is True
    assert data["containers"][0]["has_healthcheck"] is True
    assert data["containers"][0]["health"] == "healthy"


def test_running_no_healthcheck_classification(fake_repo, monkeypatch):
    ps_stdout = _container_row("abc123", "orion-widget-1") + "\n"
    inspect_body = [
        {
            "Id": "abc123",
            "State": {"Status": "running"},
            "Config": {},
        }
    ]
    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _runner_factory(ps_stdout=ps_stdout, inspect_body=inspect_body))
    data, out = _run(_plan_request(skill_args={"service": "orion-widget"}))
    assert data["status"] == "running_no_healthcheck"
    assert out.ok is True
    assert data["containers"][0]["has_healthcheck"] is False


def test_healthy_classification_with_realistic_short_vs_full_container_ids(fake_repo, monkeypatch):
    """Real docker output shape, not the same-string-both-places shortcut the other classification
    tests use: `docker compose ps --format json` reports a 12-char short id; `docker container
    inspect` always returns the full 64-char Id regardless of what id form it was invoked with.
    Confirmed live 2026-08-12: keying inspect results only by the full id made every poll-loop
    lookup by the short id miss, silently defaulting every container to state="unknown" for the
    entire poll window even though it had built, started, and was actually running -- misclassified
    as unhealthy. This must resolve to healthy just like the same-id-everywhere test above.
    (Fixed via `_resolve_container_state_key`'s read-time prefix match -- the same mechanism
    skills.runtime.docker_prune_stopped_containers.v1 already used for this identical id-form
    mismatch, not a second bespoke fix.)
    """
    short_id = "23292279c2e4"
    full_id = (short_id + "d676c39adb044c785861e2b4abef94fd3e8707e4e3202e9bc6c" * 2)[:64]
    assert len(full_id) == 64
    ps_stdout = _container_row(short_id, "orion-widget-1") + "\n"
    inspect_body = [
        {
            "Id": full_id,
            "State": {"Status": "running", "Health": {"Status": "healthy"}},
            "Config": {"Healthcheck": {"Test": ["CMD", "curl", "-f", "http://localhost/health"]}},
        }
    ]
    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _runner_factory(ps_stdout=ps_stdout, inspect_body=inspect_body))
    data, out = _run(_plan_request(skill_args={"service": "orion-widget"}))
    assert data["status"] == "healthy"
    assert out.ok is True
    assert data["containers"][0]["state"] == "running"
    assert data["containers"][0]["has_healthcheck"] is True
    assert data["containers"][0]["health"] == "healthy"


def test_healthy_classification_with_multiple_containers_distinct_short_ids(fake_repo, monkeypatch):
    """Review finding 2026-08-12: every other classification test (including the short-vs-full-id
    one above) inspects exactly one container, so per-container attribution when 2+ containers are
    inspected in a single call was entirely unexercised. Two containers with distinct short/full ids
    must each resolve to their own state, not bleed into each other."""
    short_id_a = "23292279c2e4"
    full_id_a = (short_id_a + "d676c39adb044c785861e2b4abef94fd3e8707e4e3202e9bc6c" * 2)[:64]
    short_id_b = "af0011223344"
    full_id_b = (short_id_b + "e787c39adb044c785861e2b4abef94fd3e8707e4e3202e9bc6d" * 2)[:64]
    assert len({full_id_a, full_id_b}) == 2
    ps_stdout = "\n".join(
        [_container_row(short_id_a, "orion-widget-1"), _container_row(short_id_b, "orion-widget-2")]
    )
    inspect_body = [
        {
            "Id": full_id_a,
            "State": {"Status": "running", "Health": {"Status": "healthy"}},
            "Config": {"Healthcheck": {"Test": ["CMD", "true"]}},
        },
        {
            "Id": full_id_b,
            "State": {"Status": "running"},
            "Config": {},
        },
    ]
    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _runner_factory(ps_stdout=ps_stdout, inspect_body=inspect_body))
    data, out = _run(_plan_request(skill_args={"service": "orion-widget"}))
    assert out.ok is True
    by_name = {c["name"]: c for c in data["containers"]}
    assert by_name["orion-widget-1"]["has_healthcheck"] is True
    assert by_name["orion-widget-1"]["health"] == "healthy"
    assert by_name["orion-widget-2"]["has_healthcheck"] is False
    assert by_name["orion-widget-2"]["health"] is None


def test_unhealthy_classification(fake_repo, monkeypatch):
    ps_stdout = _container_row("abc123", "orion-widget-1") + "\n"
    inspect_body = [
        {
            "Id": "abc123",
            "State": {"Status": "running", "Health": {"Status": "unhealthy"}},
            "Config": {"Healthcheck": {"Test": ["CMD", "true"]}},
        }
    ]
    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _runner_factory(ps_stdout=ps_stdout, inspect_body=inspect_body))
    data, out = _run(_plan_request(skill_args={"service": "orion-widget"}))
    assert data["status"] == "unhealthy"
    assert out.ok is False


def test_no_containers_reports_unhealthy(fake_repo, monkeypatch):
    monkeypatch.setattr(verb_adapters.settings, "skills_docker_compose_bringup_health_poll_sec", 0.01)
    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _runner_factory(ps_stdout=""))
    data, out = _run(_plan_request(skill_args={"service": "orion-widget"}))
    assert data["status"] == "unhealthy"
    assert out.ok is False
    assert data["containers"] == []


def test_build_output_redacts_secret_like_tokens(fake_repo, monkeypatch):
    class _Runner:
        def __init__(self, **kwargs):
            pass

        def run(self, command, cwd=None, env=None):
            if "build" in command and command[:2] == ["docker", "compose"]:
                return SimpleNamespace(
                    returncode=1,
                    stdout="cloning with token ghp_abcdefghijklmnopqrstuvwxyz0123456789\n",
                    stderr="Authorization: Bearer sk-live-abcdefghijklmnop\n",
                )
            return SimpleNamespace(returncode=1, stdout="", stderr=f"unexpected:{command!r}")

    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _Runner)
    data, out = _run(_plan_request(skill_args={"service": "orion-widget"}))
    assert data["status"] == "build_failed"
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in data["build_stdout_stderr_tail"]
    assert "sk-live-abcdefghijklmnop" not in data["build_stdout_stderr_tail"]
    assert "[REDACTED]" in data["build_stdout_stderr_tail"]


def test_flapping_container_fails_debounce_confirmation(fake_repo, monkeypatch):
    """A container that looks settled on the first snapshot but has flipped away from running by the
    confirming re-poll must not be reported healthy (review finding 2026-08-12: debounce guard)."""
    ps_stdout = _container_row("abc123", "orion-widget-1") + "\n"
    calls = {"n": 0}
    healthy_body = [
        {
            "Id": "abc123",
            "State": {"Status": "running", "Health": {"Status": "healthy"}},
            "Config": {"Healthcheck": {"Test": ["CMD", "true"]}},
        }
    ]
    flapped_body = [
        {
            "Id": "abc123",
            "State": {"Status": "restarting", "Health": {"Status": "unhealthy"}},
            "Config": {"Healthcheck": {"Test": ["CMD", "true"]}},
        }
    ]

    class _Runner:
        def __init__(self, **kwargs):
            pass

        def run(self, command, cwd=None, env=None):
            if "build" in command and command[:2] == ["docker", "compose"]:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if "up" in command and command[:2] == ["docker", "compose"]:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if "ps" in command and command[:2] == ["docker", "compose"]:
                return SimpleNamespace(returncode=0, stdout=ps_stdout, stderr="")
            if command[:3] == ["docker", "container", "inspect"]:
                calls["n"] += 1
                body = healthy_body if calls["n"] == 1 else flapped_body
                return SimpleNamespace(returncode=0, stdout=json.dumps(body), stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr=f"unexpected:{command!r}")

    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _Runner)
    monkeypatch.setattr(verb_adapters.settings, "skills_docker_compose_bringup_health_poll_sec", 0.01)
    data, out = _run(_plan_request(skill_args={"service": "orion-widget"}))
    # First snapshot looked healthy, but the confirming re-poll caught the flap -- and the poll window
    # (0.01s) expires before a third attempt can settle, so the loop must exit unsettled rather than
    # trusting the first, stale-by-the-time-it's-reported snapshot.
    assert data["status"] == "unhealthy"
    assert out.ok is False
    assert calls["n"] >= 2
