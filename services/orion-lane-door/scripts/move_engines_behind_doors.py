#!/usr/bin/env python3
"""Move circe's doored engines off their public lane ports, so the lane doors can take them.

Run by services/orion-lane-door/scripts/post_rebuild.sh after every post-pull rebuild of the doors on circe
(the doors are up by then). For each RUNNING engine of the atlas-workers compose project
(services/orion-llamacpp-host/docker-compose.atlas-workers.yml) that this door set fronts:

* it publishes only on loopback already -> nothing to do (every deploy after the first);
* it still publishes on a public address, and compose now resolves it to loopback (the default since the
  lane door shipped) -> recreate it (``up -d --no-build --no-deps``). The engine reloads its model; its
  door takes the lane port as soon as the engine answers on loopback;
* it publishes on a public address and compose still says so (an operator pinned ``*_ENGINE_PUBLISH`` to the
  lane port, the documented per-lane rollback) -> leave it. Its door keeps waiting.

Never touched: a stopped engine (whatever starts it next starts it on loopback), and atlas-agent-burst (the
GPU pool's: the lane controller's next load recreates it on loopback; recreating it here would drop the
profile and card the controller chose). Never runs anywhere but circe.

    python3 services/orion-lane-door/scripts/move_engines_behind_doors.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parents[3]
PROJECT = "orion-llamacpp-host"
COMPOSE_FILE = "docker-compose.atlas-workers.yml"
ENGINE_PORT = "8080/tcp"
LOOPBACK = {"127.0.0.1", "::1"}

# compose service -> (lane, compose profile or None). The engines a circe door fronts.
ENGINES: dict[str, tuple[str, Optional[str]]] = {
    "atlas-chat": ("chat", None),
    "atlas-chat-llamacpp": ("chat", "chat-llamacpp"),
    "atlas-fast": ("fast", None),
    "atlas-agent": ("agent", "agent-split"),
    "atlas-agent-llamacpp": ("agent", "agent-llamacpp"),
}
POOL_OWNED = {"atlas-agent-burst"}   # lane-door-agent-gpu2's engine; the lane controller moves it

Runner = Callable[[list[str]], "tuple[int, str]"]


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    if proc.returncode != 0 and proc.stderr:
        print(proc.stderr.strip(), file=sys.stderr)
    return proc.returncode, proc.stdout


def compose_base() -> list[str]:
    return ["docker", "compose", "--env-file", ".env", "--env-file", f"services/{PROJECT}/.env",
            "-f", f"services/{PROJECT}/{COMPOSE_FILE}"]


def public_bindings(inspect: dict) -> list[str]:
    """The engine port's host bindings that are NOT loopback, as ``ip:port``. Empty host ip = every address."""
    bindings = ((inspect.get("HostConfig") or {}).get("PortBindings") or {}).get(ENGINE_PORT) or []
    return [f"{b.get('HostIp') or '0.0.0.0'}:{b.get('HostPort')}" for b in bindings
            if (b.get("HostIp") or "") not in LOOPBACK]


def running_engines(runner: Runner) -> dict[str, dict]:
    """compose service -> docker inspect, for running containers of the atlas-workers project."""
    rc, out = runner(["docker", "ps", "-q", "--filter", f"label=com.docker.compose.project={PROJECT}"])
    ids = out.split() if rc == 0 else []
    if rc != 0:
        raise SystemExit("move_engines_behind_doors: docker ps failed")
    if not ids:
        return {}
    rc, out = runner(["docker", "inspect", *ids])
    if rc != 0:
        raise SystemExit("move_engines_behind_doors: docker inspect failed")
    found = {}
    for item in json.loads(out or "[]"):
        labels = (item.get("Config") or {}).get("Labels") or {}
        if (item.get("State") or {}).get("Running"):
            found[labels.get("com.docker.compose.service", "")] = item
    return found


def resolves_to_loopback(runner: Runner, service: str, profile: Optional[str]) -> bool:
    """What compose would publish for this service now (current files + .env): loopback only?"""
    cmd = compose_base() + (["--profile", profile] if profile else []) + ["config", "--format", "json"]
    rc, out = runner(cmd)
    if rc != 0:
        raise SystemExit(f"move_engines_behind_doors: compose config failed for {service}")
    ports = ((json.loads(out).get("services") or {}).get(service) or {}).get("ports") or []
    mine = [p for p in ports if str(p.get("target")) == ENGINE_PORT.split("/")[0]]
    return bool(mine) and all((p.get("host_ip") or "") in LOOPBACK for p in mine)


def plan(runner: Runner) -> tuple[list[tuple[str, Optional[str]]], list[str]]:
    """(services to recreate with their profile, report lines)."""
    to_move: list[tuple[str, Optional[str]]] = []
    lines: list[str] = []
    for service, item in sorted(running_engines(runner).items()):
        public = public_bindings(item)
        if service in POOL_OWNED:
            if public:
                lines.append(f"{service}: on {', '.join(public)}; the lane controller's next load moves it")
            continue
        if service not in ENGINES:
            continue
        lane, profile = ENGINES[service]
        if not public:
            lines.append(f"{service} ({lane}): already on loopback")
        elif resolves_to_loopback(runner, service, profile):
            lines.append(f"{service} ({lane}): on {', '.join(public)} -> recreating on loopback")
            to_move.append((service, profile))
        else:
            lines.append(f"{service} ({lane}): on {', '.join(public)}, and its *_ENGINE_PUBLISH keeps it there "
                         "(pinned); its door waits")
    return to_move, lines


def move_command(to_move: list[tuple[str, Optional[str]]]) -> list[str]:
    profiles = sorted({p for _, p in to_move if p})
    cmd = ["scripts/safe_docker_build.sh", PROJECT, "--compose-file", COMPOSE_FILE]
    for p in profiles:
        cmd += ["--profile", p]
    return cmd + ["up", "-d", "--no-build", "--no-deps", *[s for s, _ in to_move]]


def main(argv: Optional[list[str]] = None, *, runner: Runner = run, host: Optional[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--dry-run", action="store_true", help="print the plan, recreate nothing")
    args = ap.parse_args(argv)
    host = host if host is not None else socket.gethostname().split(".")[0]
    if host != "circe":
        print(f"move_engines_behind_doors: host {host} is not circe; nothing to move")
        return 0
    to_move, lines = plan(runner)
    for line in lines:
        print(f"move_engines_behind_doors: {line}")
    if not to_move:
        return 0
    cmd = move_command(to_move)
    print(f"move_engines_behind_doors: {'would run' if args.dry_run else 'running'}: {' '.join(cmd)}")
    if args.dry_run:
        return 0
    rc, out = runner(cmd)
    if out:
        print(out.rstrip())
    return rc


if __name__ == "__main__":
    sys.exit(main())
