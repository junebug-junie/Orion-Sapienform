from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from .roster import NEVER_REMEDIATE_IDS, RosterEntry


@dataclass(frozen=True)
class RemediationResult:
    ok: bool
    tier: int
    command: list[str]
    exit_code: int
    stderr_tail: str


def build_compose_command(entry: RosterEntry, *, repo_root: str, tier: int) -> list[str]:
    root = Path(repo_root)
    cmd: list[str] = ["docker", "compose", "--env-file", str(root / ".env")]
    if entry.include_bus_env:
        cmd.extend(["--env-file", str(root / "services" / "orion-bus" / ".env")])
    cmd.extend(["--env-file", str(root / "services" / entry.compose_dir / ".env")])
    cmd.extend(["-f", str(root / "services" / entry.compose_dir / "docker-compose.yml")])
    if tier == 1:
        cmd.extend(["up", "-d", "--force-recreate", entry.compose_service])
    elif tier == 2:
        raise ValueError("tier 2 uses build_compose_build_command + build_compose_up_command")
    else:
        raise ValueError(f"unsupported tier {tier}")
    return cmd


def build_compose_build_command(entry: RosterEntry, *, repo_root: str) -> list[str]:
    root = Path(repo_root)
    cmd: list[str] = ["docker", "compose", "--env-file", str(root / ".env")]
    if entry.include_bus_env:
        cmd.extend(["--env-file", str(root / "services" / "orion-bus" / ".env")])
    cmd.extend(["--env-file", str(root / "services" / entry.compose_dir / ".env")])
    cmd.extend(["-f", str(root / "services" / entry.compose_dir / "docker-compose.yml")])
    cmd.extend(["build", entry.compose_service])
    return cmd


def build_compose_up_command(entry: RosterEntry, *, repo_root: str) -> list[str]:
    root = Path(repo_root)
    cmd: list[str] = ["docker", "compose", "--env-file", str(root / ".env")]
    if entry.include_bus_env:
        cmd.extend(["--env-file", str(root / "services" / "orion-bus" / ".env")])
    cmd.extend(["--env-file", str(root / "services" / entry.compose_dir / ".env")])
    cmd.extend(["-f", str(root / "services" / entry.compose_dir / "docker-compose.yml")])
    cmd.extend(["up", "-d", entry.compose_service])
    return cmd


async def _run_command(cmd: list[str], *, repo_root: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=repo_root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    tail = (stderr or b"").decode("utf-8", "replace")[-2000:]
    return proc.returncode or 0, tail


async def execute_remediation(entry: RosterEntry, *, repo_root: str, tier: int) -> RemediationResult:
    if entry.id in NEVER_REMEDIATE_IDS or not entry.auto_remediate:
        return RemediationResult(ok=False, tier=tier, command=[], exit_code=1, stderr_tail="remediation_blocked")

    if tier == 1:
        cmd = build_compose_command(entry, repo_root=repo_root, tier=1)
        code, tail = await _run_command(cmd, repo_root=repo_root)
        return RemediationResult(ok=code == 0, tier=tier, command=cmd, exit_code=code, stderr_tail=tail)

    if tier == 2:
        build_cmd = build_compose_build_command(entry, repo_root=repo_root)
        code, tail = await _run_command(build_cmd, repo_root=repo_root)
        if code != 0:
            return RemediationResult(ok=False, tier=tier, command=build_cmd, exit_code=code, stderr_tail=tail)
        up_cmd = build_compose_up_command(entry, repo_root=repo_root)
        code, tail = await _run_command(up_cmd, repo_root=repo_root)
        return RemediationResult(ok=code == 0, tier=tier, command=up_cmd, exit_code=code, stderr_tail=tail)

    return RemediationResult(ok=False, tier=tier, command=[], exit_code=1, stderr_tail=f"unsupported tier {tier}")
