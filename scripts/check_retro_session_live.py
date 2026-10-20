#!/usr/bin/env python3
"""Is a retro game session live right now? Read-only; used by the post-pull rebuild.

A retro game lives only in the orion-retro worker's memory. Recreating
orion-retro (or orion-hub, whose retro gateway carries the live session)
loses it. 2026-10-20 22:13Z: a post-pull rebuild did exactly that mid-game.

Signal: the retro service writes its session to ``status.json`` in its
evidence directory (services/orion-retro/app/session.py ``persist()``), with
``state`` one of starting/running/paused/stopped/failed. A file left behind
by a dead container can still say "running", so the container must also be
running. A container that restarts rewrites any live state to ``failed`` on
boot (session.py ``Sessions.__init__``), so running container + live state
means a session is really in memory.

Exit codes:
  0  not live (no container, container not running, no session, or ended)
  3  live (starting/running/paused, container running)
  4  unknown (docker or the status file could not be read) -- callers skip
     the rebuild: killing a game is worse than a late rebuild.
One line on stdout explains the verdict.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

NOT_LIVE, LIVE, UNKNOWN = 0, 3, 4
LIVE_STATES = frozenset({"starting", "running", "paused"})
CONTAINER = "orion-athena-retro"
EVIDENCE_MOUNT = "/evidence"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_evidence_dir(repo_root: Path) -> str | None:
    env_file = repo_root / "services" / "orion-retro" / ".env"
    try:
        lines = env_file.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == "RETRO_EVIDENCE_DIR":
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


def check(container: str, repo_root: Path, docker: str = "docker") -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [docker, "inspect", container],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return UNKNOWN, f"docker inspect {container} failed: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        if "no such object" in err.lower() or "no such container" in err.lower():
            return NOT_LIVE, f"no container {container}"
        return UNKNOWN, f"docker inspect {container} exit {proc.returncode}: {err[:200]}"
    try:
        info = json.loads(proc.stdout)[0]
        running = bool(info["State"]["Running"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return UNKNOWN, f"docker inspect {container} output unreadable: {exc}"
    if not running:
        status = info["State"].get("Status", "?")
        return NOT_LIVE, f"container {container} is {status}; no game in memory"

    # The mount the running container actually uses beats the .env file.
    evidence = None
    for mount in info.get("Mounts") or []:
        if isinstance(mount, dict) and mount.get("Destination") == EVIDENCE_MOUNT:
            evidence = mount.get("Source")
            break
    evidence = evidence or _env_evidence_dir(repo_root)
    if not evidence:
        return UNKNOWN, f"container {container} running; evidence dir unknown"
    evidence_dir = Path(evidence)
    if not evidence_dir.is_dir():
        return UNKNOWN, f"container {container} running; evidence dir {evidence_dir} missing"
    status_path = evidence_dir / "status.json"
    try:
        mtime = status_path.stat().st_mtime
    except FileNotFoundError:
        return NOT_LIVE, f"container {container} running; no session ever started ({status_path} absent)"
    except OSError as exc:
        return UNKNOWN, f"container {container} running; cannot stat {status_path}: {exc}"
    try:
        session = json.loads(status_path.read_text())
        state = session["state"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return UNKNOWN, f"container {container} running; {status_path} unreadable: {exc}"
    game = session.get("game_id", "?") if isinstance(session, dict) else "?"
    if state in LIVE_STATES:
        # status.json is written at start/ready/state changes, not per frame, so
        # age is not proof of staleness; it is reported so an operator can judge.
        age_h = max(0.0, time.time() - mtime) / 3600
        return LIVE, (f"session state={state} game={game} in running container {container}"
                      f" (status.json last written {age_h:.1f}h ago)")
    return NOT_LIVE, f"session state={state} game={game}; not live"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--container", default=CONTAINER)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    code, reason = check(args.container, args.repo_root, os.environ.get("ORION_DOCKER_BIN", "docker"))
    print(reason)
    return code


if __name__ == "__main__":
    sys.exit(main())
