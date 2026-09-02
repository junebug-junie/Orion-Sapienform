"""Exclusive GPU1 flip between orion-affectgpt-worker and the atlas-agent
llama.cpp worker, both on circe.

Deliberately narrow: this module only ever runs `docker compose` against the
two fixed compose targets named in Settings, never a caller-supplied service
name (contrast cortex-exec's skills.docker.compose_service_bringup.v1, which
is a generic any-service bringup). The command surface is small enough to be
read in one sitting.

Health-poll/container-state shape mirrors
services/orion-cortex-exec/app/verb_adapters.py's
_run_docker_compose_service_bringup as closely as this simpler, two-target
case allows, so the result payloads read the same way operators are already
used to (status/diagnostics/user_facing_summary, healthy/
running_no_healthcheck/unhealthy classification).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .settings import settings


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
class LaneTarget:
    key: str  # "affect" | "agent"
    compose_relpath: str
    env_relpath: str
    compose_service: str
    profile: Optional[str] = None
    extra_env: Dict[str, str] = field(default_factory=dict)


def _targets() -> Dict[str, LaneTarget]:
    return {
        "affect": LaneTarget(
            key="affect",
            compose_relpath=settings.AFFECT_COMPOSE_RELPATH,
            env_relpath=settings.AFFECT_ENV_RELPATH,
            compose_service=settings.AFFECT_COMPOSE_SERVICE,
        ),
        "agent": LaneTarget(
            key="agent",
            compose_relpath=settings.AGENT_COMPOSE_RELPATH,
            env_relpath=settings.AGENT_ENV_RELPATH,
            compose_service=settings.AGENT_COMPOSE_SERVICE,
            profile=settings.AGENT_COMPOSE_PROFILE,
            extra_env={"ATLAS_AGENT_CUDA_VISIBLE_DEVICES": settings.AGENT_GPU1_CUDA_VISIBLE_DEVICES},
        ),
    }


def _repo_root() -> Path:
    return Path(settings.GPU_LANE_REPO_ROOT).resolve()


def _base_cmd(target: LaneTarget, repo_root: Path) -> List[str]:
    cmd = ["docker", "compose"]
    root_env = repo_root / ".env"
    if root_env.is_file():
        cmd += ["--env-file", ".env"]
    if (repo_root / target.env_relpath).is_file():
        cmd += ["--env-file", target.env_relpath]
    if target.profile:
        cmd += ["--profile", target.profile]
    cmd += ["-f", target.compose_relpath]
    return cmd


def _invoke_env(target: LaneTarget) -> dict[str, str]:
    if not target.extra_env:
        return dict(os.environ)
    return {**os.environ, **target.extra_env}


def _truncate(text: str, max_len: int = 8000) -> str:
    t = (text or "").strip()
    if len(t) > max_len:
        return t[: max_len - 40] + "\n...(stdout/stderr truncated)..."
    return t


def _compose_ps_rows(text: str) -> List[Dict[str, Any]]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    rows: List[Dict[str, Any]] = []
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
    out: List[Dict[str, Any]] = []
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


def _snapshot(runner: SafeCommandRunner, repo_root: Path, target: LaneTarget) -> Dict[str, Any]:
    """Current container state for one target, via `docker compose ps`.

    Returns {"running": bool, "state": str, "containers": [...]} --
    "running" is true only when the named service's container is Up (a
    healthcheck-defined container additionally needs health == "healthy").
    """
    cmd = [*_base_cmd(target, repo_root), "ps", target.compose_service, "-a", "--format", "json"]
    try:
        proc = runner.run(cmd, cwd=str(repo_root))
    except Exception as exc:  # noqa: BLE001
        return {"running": False, "state": "unknown", "containers": [], "error": str(exc)}
    rows = _compose_ps_rows(proc.stdout) if proc.returncode == 0 else []
    if not rows:
        return {"running": False, "state": "absent", "containers": []}
    running = all(
        row["state"] == "running" and (row["health"] in (None, "healthy")) for row in rows
    )
    state = "running" if running else rows[0]["state"]
    return {"running": running, "state": state, "containers": rows}


def get_status() -> Dict[str, Any]:
    repo_root = _repo_root()
    runner = SafeCommandRunner(allowed_commands={"docker"}, timeout_sec=30.0)
    targets = _targets()
    affect = _snapshot(runner, repo_root, targets["affect"])
    agent = _snapshot(runner, repo_root, targets["agent"])
    if affect["running"] and agent["running"]:
        active = "both"  # should never persist -- both lanes claiming GPU1 at once is a bug, not a state
    elif affect["running"]:
        active = "affect"
    elif agent["running"]:
        active = "agent"
    else:
        active = "neither"
    return {
        "observed_at_utc": _utc_now_iso(),
        "active": active,
        "affect": affect,
        "agent": agent,
    }


def _run_compose(
    runner: SafeCommandRunner, repo_root: Path, target: LaneTarget, *extra_args: str
) -> subprocess.CompletedProcess[str]:
    cmd = [*_base_cmd(target, repo_root), *extra_args, target.compose_service]
    return runner.run(cmd, cwd=str(repo_root), env=_invoke_env(target))


def _stop(runner: SafeCommandRunner, repo_root: Path, target: LaneTarget) -> Dict[str, Any]:
    try:
        proc = _run_compose(runner, repo_root, target, "stop")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "exit_code": None, "tail": str(exc)}
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "tail": _truncate((proc.stdout or "") + "\n" + (proc.stderr or "")),
    }


def _bring_up(runner: SafeCommandRunner, repo_root: Path, target: LaneTarget) -> Dict[str, Any]:
    try:
        build_proc = _run_compose(runner, repo_root, target, "build")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "phase": "build", "exit_code": None, "tail": str(exc)}
    if build_proc.returncode != 0:
        return {
            "ok": False,
            "phase": "build",
            "exit_code": build_proc.returncode,
            "tail": _truncate((build_proc.stdout or "") + "\n" + (build_proc.stderr or "")),
        }
    try:
        up_proc = _run_compose(runner, repo_root, target, "up", "-d")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "phase": "up", "exit_code": None, "tail": str(exc)}
    return {
        "ok": up_proc.returncode == 0,
        "phase": "up",
        "exit_code": up_proc.returncode,
        "tail": _truncate((up_proc.stdout or "") + "\n" + (up_proc.stderr or "")),
    }


async def _poll_until_settled(target: LaneTarget) -> Dict[str, Any]:
    repo_root = _repo_root()
    poll_runner = SafeCommandRunner(allowed_commands={"docker"}, timeout_sec=15.0)
    deadline = time.monotonic() + float(settings.GPU_LANE_HEALTH_POLL_SEC)
    snapshot = {"running": False, "state": "unknown", "containers": []}
    while True:
        snapshot = await asyncio.to_thread(_snapshot, poll_runner, repo_root, target)
        if snapshot["running"]:
            return snapshot
        if time.monotonic() >= deadline:
            return snapshot
        await asyncio.sleep(min(3.0, max(0.0, deadline - time.monotonic())))


# Guards the whole flip sequence (status read -> stop -> build -> up ->
# poll). Without this, two overlapping POST /v1/gpu-lane/flip calls (a
# double-click, or a client retry after its own timeout while the first
# call is still running) would both read a stale snapshot before either
# had acted, both conclude "not a no-op", and both start running `docker
# compose stop`/`build`/`up -d` against the same two compose targets
# concurrently -- review finding: exactly the "both"/"neither" state this
# module's docstrings already call out as a bug, not a valid outcome.
_FLIP_LOCK = asyncio.Lock()


async def flip(target_key: str) -> Dict[str, Any]:
    """Bring GPU1 to `target_key` exclusively.

    Serialized: only one flip runs at a time (see `_FLIP_LOCK`) -- a flip
    already in progress makes a concurrent call return `"busy"` immediately
    rather than queueing behind a sequence that can legitimately take
    several minutes (build + a cold GGUF/model load), or racing it.

    Idempotent: if `target` is already the sole running lane, this is a
    no-op (no stop/build/up/restart of an already-good, possibly-still-
    warming-up lane just because the same button got clicked twice). If
    `other` is also running (the "both" state get_status() can detect but
    should never actually see -- a bug, not a valid state), or if `other`
    is up while `target` is not, this stops `other` and brings `target` up,
    same as a normal flip -- self-healing rather than trusting the no-op
    path to cover every case.
    """
    if _FLIP_LOCK.locked():
        return {
            "observed_at_utc": _utc_now_iso(),
            "status": "busy",
            "target": target_key,
            "user_facing_summary": "Another GPU1 flip is already in progress -- try again once it settles.",
        }
    async with _FLIP_LOCK:
        return await _flip_locked(target_key)


async def _flip_locked(target_key: str) -> Dict[str, Any]:
    targets = _targets()
    if target_key not in targets:
        return {
            "observed_at_utc": _utc_now_iso(),
            "status": "invalid_target",
            "target": target_key,
            "user_facing_summary": f"target must be one of {sorted(targets)}, got {target_key!r}.",
        }

    other_key = "agent" if target_key == "affect" else "affect"
    target = targets[target_key]
    other = targets[other_key]
    repo_root = _repo_root()
    runner = SafeCommandRunner(
        allowed_commands={"docker"}, timeout_sec=float(settings.GPU_LANE_COMMAND_TIMEOUT_SEC)
    )

    status_runner = SafeCommandRunner(allowed_commands={"docker"}, timeout_sec=30.0)
    target_snapshot = await asyncio.to_thread(_snapshot, status_runner, repo_root, target)
    other_snapshot = await asyncio.to_thread(_snapshot, status_runner, repo_root, other)
    if target_snapshot["running"] and not other_snapshot["running"]:
        return {
            "observed_at_utc": _utc_now_iso(),
            "status": "noop",
            "target": target_key,
            "settled": target_snapshot,
            "user_facing_summary": f"GPU1 already on {target_key} ({target_snapshot['state']}) -- no action taken.",
        }

    stop_result = await asyncio.to_thread(_stop, runner, repo_root, other)
    if not stop_result["ok"]:
        return {
            "observed_at_utc": _utc_now_iso(),
            "status": "stop_failed",
            "target": target_key,
            "stop_other": stop_result,
            "user_facing_summary": f"Failed to stop {other_key} (exit {stop_result['exit_code']}) -- {target_key} was not started.",
        }

    bringup_result = await asyncio.to_thread(_bring_up, runner, repo_root, target)
    if not bringup_result["ok"]:
        return {
            "observed_at_utc": _utc_now_iso(),
            # Honest, not silently retried: other is down, target failed to come up --
            # GPU1 may now be idle. Surfaced explicitly rather than papered over.
            "status": f"{bringup_result['phase']}_failed",
            "target": target_key,
            "stop_other": stop_result,
            "bring_up": bringup_result,
            "user_facing_summary": (
                f"Stopped {other_key} ok, but {target_key} {bringup_result['phase']} failed "
                f"(exit {bringup_result['exit_code']}). GPU1 may now be idle -- neither lane is confirmed up."
            ),
        }

    settled = await _poll_until_settled(target)
    ok = bool(settled["running"])
    return {
        "observed_at_utc": _utc_now_iso(),
        "status": "success" if ok else "unhealthy",
        "target": target_key,
        "stop_other": stop_result,
        "bring_up": bringup_result,
        "settled": settled,
        "user_facing_summary": (
            f"GPU1 flipped to {target_key}: {other_key} stopped, {target_key} settled to {settled['state']}."
            if ok
            else (
                f"GPU1 flip to {target_key}: {other_key} stopped, {target_key} did not settle to running "
                f"within {settings.GPU_LANE_HEALTH_POLL_SEC:.0f}s (state={settled['state']})."
            )
        ),
    }
