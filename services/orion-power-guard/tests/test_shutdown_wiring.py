"""Regression coverage for power-guard's host-shutdown wiring.

power-guard runs in a container, so a local `shutdown` command only kills the
container, not the host it's meant to protect (confirmed live 2026-09-08 on
athena: the shipped default was `/sbin/shutdown -h +1 ...`, a binary that
does not exist in the container image and would not have touched the host
even if it did). The container already mounts a dedicated SSH key
(docker-compose.yml -> /etc/powerguard/ssh_key) for exactly this purpose --
it just was never wired into POWER_GUARD_SHUTDOWN_CMD. These tests pin the
default so it stays an actual host-reaching command, and don't attempt to
run it (that would shut down whatever host runs the suite).
"""

from __future__ import annotations

from app.settings import Settings


def _pure_defaults() -> Settings:
    """Settings() with no .env picked up -- the actual code-level defaults,
    independent of whatever a given deployment's local .env overrides them
    to (this repo's own .env for this service intentionally sets
    POWER_GUARD_ENABLE_SHUTDOWN=true on athena; that must not make this
    test pass for the wrong reason)."""
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_default_shutdown_cmd_uses_ssh_key_not_local_shutdown() -> None:
    """The default must reach the host via the mounted key, not run locally."""
    cmd = _pure_defaults().POWER_GUARD_SHUTDOWN_CMD
    assert cmd.startswith("ssh "), "default shutdown command must SSH to the host"
    assert "/etc/powerguard/ssh_key" in cmd, "must use the key docker-compose.yml mounts"
    assert "root@host.docker.internal" in cmd, "must target the host over the docker bridge"
    assert "shutdown -h now" in cmd


def test_shutdown_disabled_by_default() -> None:
    """Arming a real host shutdown is a deliberate per-deployment opt-in."""
    assert _pure_defaults().POWER_GUARD_ENABLE_SHUTDOWN is False


def test_default_grace_period_is_five_minutes() -> None:
    """5 min: long enough to ride out a breaker flip, short enough to leave
    real shutdown margin on the UPS's ~20min runtime."""
    assert _pure_defaults().POWER_GUARD_ONBATTERY_GRACE_SEC == 300.0
