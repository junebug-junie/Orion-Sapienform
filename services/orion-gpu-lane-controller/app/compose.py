"""The controller's only Docker surface: ``docker compose`` against one named service.

Kept from the retired GPU1 flip (lane_control.py, deleted in GPU pool stage 5.6) because
launch_exec.py runs every step through it. Nothing here chooses a service: callers pass a
``ComposeTarget`` built from this checkout's config/gpu_pool.yaml ``launch`` block.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .settings import settings


class SafeCommandRunner:
    """Restricted subprocess runner: only ever invokes an allowlisted binary,
    resolved by basename, never through a shell. Mirrors cortex-exec's
    verb_adapters.SafeCommandRunner -- reimplemented locally rather than
    imported cross-service (that class is module-private to cortex-exec's
    app, and CLAUDE.md's service-boundary rules say not to reach into
    another service's internals for a ~30-line helper)."""

    def __init__(self, *, allowed_commands: set[str], timeout_sec: float) -> None:
        self.allowed_commands = set(allowed_commands)
        self.timeout_sec = float(timeout_sec)

    def run(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if not command:
            raise PermissionError("empty_command")
        binary = str(command[0]).strip()
        base = os.path.basename(binary) or binary
        if base not in self.allowed_commands:
            raise PermissionError(f"command_not_allowlisted:{binary}")
        if os.path.isabs(binary) and os.path.isfile(binary) and os.access(binary, os.X_OK):
            resolved = binary
        else:
            resolved = shutil.which(base)
        if not resolved:
            raise FileNotFoundError(base)
        run_kw: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": self.timeout_sec,
            "check": False,
            "cwd": cwd,
        }
        if env is not None:
            run_kw["env"] = env
        return subprocess.run([resolved, *command[1:]], **run_kw)


@dataclass
class ComposeTarget:
    key: str  # the pool role
    compose_relpath: str
    env_relpath: str
    compose_service: str
    profile: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)


def repo_root() -> Path:
    return Path(settings.GPU_LANE_REPO_ROOT).resolve()


def base_cmd(target: ComposeTarget, root: Path) -> list[str]:
    cmd = ["docker", "compose"]
    if (root / ".env").is_file():
        cmd += ["--env-file", ".env"]
    if target.env_relpath and (root / target.env_relpath).is_file():
        cmd += ["--env-file", target.env_relpath]
    if target.profile:
        cmd += ["--profile", target.profile]
    cmd += ["-f", target.compose_relpath]
    return cmd


def _ps_rows(text: str) -> list[dict[str, Any]]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    rows: list[dict[str, Any]] = []
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            rows = [item for item in parsed if isinstance(item, dict)]
        elif isinstance(parsed, dict):
            rows = [parsed]
    except json.JSONDecodeError:
        for raw_line in stripped.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    out: list[dict[str, Any]] = []
    for row in rows:
        cid = str(row.get("ID") or row.get("Id") or "").strip()
        if not cid:
            continue
        out.append(
            {
                "id": cid,
                "name": str(row.get("Name") or row.get("Names") or cid),
                "state": str(row.get("State") or "unknown"),
                "health": str(row.get("Health") or "") or None,
            }
        )
    return out


def snapshot(runner: SafeCommandRunner, root: Path, target: ComposeTarget) -> dict[str, Any]:
    """Current container state for one target, via `docker compose ps`.

    Returns {"running": bool, "state": str, "containers": [...]} --
    "running" is true only when the named service's container is Up (a
    healthcheck-defined container additionally needs health == "healthy").
    """
    cmd = [*base_cmd(target, root), "ps", target.compose_service, "-a", "--format", "json"]
    try:
        proc = runner.run(cmd, cwd=str(root))
    except Exception as exc:  # noqa: BLE001
        return {"running": False, "state": "unknown", "containers": [], "error": str(exc)}
    if proc.returncode != 0:
        return {"running": False, "state": "unknown", "containers": [], "error": "docker_ps_failed"}
    rows = _ps_rows(proc.stdout)
    if not rows and proc.stdout.strip() not in {"", "[]"}:
        return {"running": False, "state": "unknown", "containers": [], "error": "docker_ps_invalid"}
    if not rows:
        return {"running": False, "state": "absent", "containers": []}
    running = all(
        row["state"] == "running" and (row["health"] in (None, "healthy")) for row in rows
    )
    state = "running" if running else rows[0]["state"]
    return {"running": running, "state": state, "containers": rows}
