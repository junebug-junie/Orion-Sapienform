"""Regression coverage for power-guard's host-shutdown wiring.

power-guard runs in a container, so a local `shutdown` command only kills the
container, not the host it's meant to protect -- the shipped default was
`/sbin/shutdown -h +1 ...`, a binary that `docker exec orion-athena-power-guard
which shutdown` showed does not exist in the container image (python:3.12-slim,
nothing in the Dockerfile installs one). The container already mounts a
dedicated SSH key (docker-compose.yml -> /etc/powerguard/ssh_key) for exactly
this purpose -- it just was never wired into POWER_GUARD_SHUTDOWN_CMD. These
tests pin the default so it stays an actual host-reaching command, and don't
attempt to run it (that would shut down whatever host runs the suite).
"""

from __future__ import annotations

import pytest

from app.settings import Settings

# Every key POWER_GUARD_SHUTDOWN_CMD/POWER_GUARD_ENABLE_SHUTDOWN/
# POWER_GUARD_ONBATTERY_GRACE_SEC-adjacent that a real deployment's .env (or,
# same thing, a container's injected environment) may set. Cleared before
# each "pure defaults" test below.
_OVERRIDABLE_KEYS = (
    "POWER_GUARD_ENABLE_SHUTDOWN",
    "POWER_GUARD_ONBATTERY_GRACE_SEC",
    "POWER_GUARD_SHUTDOWN_CMD",
)


@pytest.fixture
def pure_defaults(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings() reflecting only the code-level defaults -- not the
    ".env file" pydantic-settings reads, AND not real process environment
    variables (which pydantic-settings reads with *higher* priority than
    the .env file, and which is exactly how docker-compose injects this
    service's live config into the running container). `_env_file=None`
    alone only blocks the former: confirmed live, `Settings(_env_file=None)`
    under `POWER_GUARD_ENABLE_SHUTDOWN=true` in the ambient environment
    still returns `True`. Athena's real (gitignored) .env for this service
    sets POWER_GUARD_ENABLE_SHUTDOWN=true -- if these tests ever ran inside
    that container's environment without this fixture, they'd mask a
    regression in the actual code defaults instead of catching one."""
    for key in _OVERRIDABLE_KEYS:
        monkeypatch.delenv(key, raising=False)
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_default_shutdown_cmd_uses_ssh_key_not_local_shutdown(pure_defaults: Settings) -> None:
    """The default must reach the host via the mounted key, not run locally."""
    cmd = pure_defaults.POWER_GUARD_SHUTDOWN_CMD
    assert cmd.startswith("ssh "), "default shutdown command must SSH to the host"
    assert "/etc/powerguard/ssh_key" in cmd, "must use the key docker-compose.yml mounts"
    assert "root@host.docker.internal" in cmd, "must target the host over the docker bridge"
    assert "-o BatchMode=yes" in cmd, "must fail fast on auth problems, never prompt"
    assert "shutdown -h now" in cmd


def test_shutdown_disabled_by_default(pure_defaults: Settings) -> None:
    """Arming a real host shutdown is a deliberate per-deployment opt-in."""
    assert pure_defaults.POWER_GUARD_ENABLE_SHUTDOWN is False


def test_default_grace_period_is_five_minutes(pure_defaults: Settings) -> None:
    """5 min: long enough to ride out a breaker flip, short enough to leave
    real shutdown margin on the UPS's ~20min runtime."""
    assert pure_defaults.POWER_GUARD_ONBATTERY_GRACE_SEC == 300.0


def test_run_shutdown_reports_failure_instead_of_only_logging_it(monkeypatch) -> None:
    """A failed attempt must be distinguishable from a successful one by the
    caller (so it can be published to the bus), not just logged."""
    import subprocess

    from app.main import _run_shutdown

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=255, cmd="ssh", stderr="Permission denied")

    monkeypatch.setattr(subprocess, "run", fake_run)
    success, detail = _run_shutdown("ssh ... 'shutdown -h now'")
    assert success is False
    assert "Permission denied" in detail or "255" in detail


def test_run_shutdown_reports_success(monkeypatch) -> None:
    import subprocess

    from app.main import _run_shutdown

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    success, _detail = _run_shutdown("ssh ... 'shutdown -h now'")
    assert success is True
