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


async def docker_cli_selfcheck(*, repo_root: str) -> str | None:
    """None if the docker CLI can reach the host daemon, else why not. The image
    build only proves the CLI exists; a client/daemon API-version gap (daemon
    upgrades raise the minimum client API) only shows when the daemon is asked."""
    try:
        code, tail = await _run_command(
            ["docker", "version", "--format", "{{.Server.APIVersion}}"], repo_root=repo_root
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None if code == 0 else (tail.strip() or f"exit {code}")


async def deployed_from_elsewhere(
    entry: RosterEntry, *, repo_root: str, socket_path: str = "/var/run/docker.sock", transport=None
) -> str | None:
    """Returns the compose working_dir the running container was deployed
    from when it is NOT this repo checkout, else None.

    Remediation always runs against the shared checkout. A service deployed
    from a worktree (scripts/safe_docker_build.sh) would be silently reverted
    to main by a recreate/rebuild -- the exact incident that wrapper exists to
    prevent -- so remediation must refuse instead. Fails closed: if the
    deploy origin cannot be read, that is reported as the reason to refuse.
    """
    import json

    import httpx

    expected = str(Path(repo_root) / "services" / entry.compose_dir)
    filters = json.dumps(
        {
            "label": [
                f"com.docker.compose.project={Path(entry.compose_dir).name}",
                f"com.docker.compose.service={entry.compose_service}",
            ]
        }
    )
    try:
        transport = transport or httpx.AsyncHTTPTransport(uds=socket_path)
        async with httpx.AsyncClient(transport=transport, base_url="http://docker", timeout=10.0) as client:
            resp = await client.get("/containers/json", params={"all": "true", "filters": filters})
            resp.raise_for_status()
            containers = resp.json()
    except Exception as exc:
        return f"<unreadable: {type(exc).__name__}: {exc}>"
    for container in containers:
        working_dir = (container.get("Labels") or {}).get("com.docker.compose.project.working_dir")
        if working_dir and working_dir.rstrip("/") != expected:
            return working_dir
    return None


async def execute_remediation(entry: RosterEntry, *, repo_root: str, tier: int) -> RemediationResult:
    if entry.id in NEVER_REMEDIATE_IDS or not entry.auto_remediate:
        return RemediationResult(ok=False, tier=tier, command=[], exit_code=1, stderr_tail="remediation_blocked")

    elsewhere = await deployed_from_elsewhere(entry, repo_root=repo_root)
    if elsewhere is not None:
        return RemediationResult(
            ok=False,
            tier=tier,
            command=[],
            exit_code=1,
            stderr_tail=(
                f"remediation_refused: {entry.compose_service} is deployed from {elsewhere}, not "
                f"{repo_root}; recreating from the shared checkout would revert that deploy. "
                f"Redeploy it from main (scripts/safe_docker_build.sh) or restart it by hand."
            ),
        )

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
