"""Guards the fix for the ephemeral-/tmp cursor durability gap.

`services/orion-actions/docker-compose.yml` previously had zero `volumes:`
entries, so every container recreate wiped `ACTIONS_SCHEDULER_CURSOR_STORE_PATH`
and `ACTIONS_WORKFLOW_SCHEDULE_STORE_PATH` (both defaulted under `/tmp`).
`SchedulerCursorStore._load()` treats a missing file as "no job ran today"
with no error, so the built-in daily jobs (daily_journal, daily_pulse_v1,
world_pulse) re-fired on every restart within the same calendar day.

This test asserts the durability fix stays wired:
- `.env_example` no longer points either store path at `/tmp`.
- `docker-compose.yml` mounts a host volume onto the directory that contains
  both configured store paths, so restarts/recreates no longer lose state.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

KEY_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

ENV_EXAMPLE_PATH = Path(SERVICE_DIR) / ".env_example"
COMPOSE_PATH = Path(SERVICE_DIR) / "docker-compose.yml"

CURSOR_KEY = "ACTIONS_SCHEDULER_CURSOR_STORE_PATH"
SCHEDULE_KEY = "ACTIONS_WORKFLOW_SCHEDULE_STORE_PATH"


def _parse_env_example() -> dict:
    out: dict = {}
    for raw in ENV_EXAMPLE_PATH.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        m = KEY_LINE.match(s)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text())


def test_env_example_store_paths_not_under_tmp() -> None:
    env = _parse_env_example()
    assert CURSOR_KEY in env, f"{CURSOR_KEY} missing from .env_example"
    assert SCHEDULE_KEY in env, f"{SCHEDULE_KEY} missing from .env_example"

    cursor_path = env[CURSOR_KEY]
    schedule_path = env[SCHEDULE_KEY]

    assert not cursor_path.startswith("/tmp"), (
        f"{CURSOR_KEY}={cursor_path!r} still points at ephemeral /tmp; "
        "container recreate wipes it and the daily jobs re-fire same-day."
    )
    assert not schedule_path.startswith("/tmp"), (
        f"{SCHEDULE_KEY}={schedule_path!r} still points at ephemeral /tmp; "
        "container recreate wipes durable schedule state."
    )


def test_compose_mounts_volume_covering_state_paths() -> None:
    env = _parse_env_example()
    cursor_dir = str(Path(env[CURSOR_KEY]).parent)
    schedule_dir = str(Path(env[SCHEDULE_KEY]).parent)

    # Both env keys are expected to resolve under the same durable directory.
    assert cursor_dir == schedule_dir, (
        f"expected cursor store dir ({cursor_dir}) and workflow schedule "
        f"store dir ({schedule_dir}) to share one mounted directory"
    )

    compose = _load_compose()
    services = compose.get("services", {})
    assert "actions" in services, "expected an 'actions' service in docker-compose.yml"

    volumes = services["actions"].get("volumes") or []
    assert volumes, (
        "orion-actions service declares no volumes: container filesystem is "
        "fully ephemeral, so scheduler cursors / workflow schedules are wiped "
        "on every recreate"
    )

    mount_targets = []
    for entry in volumes:
        # Support both "host:container" short syntax and long-form mapping dicts.
        # rsplit is required because host paths commonly use compose
        # ${VAR:-default} substitution syntax, which itself contains a colon
        # (e.g. "${ORION_DATA_ROOT:-/mnt/graphdb}/x:/container/path").
        if isinstance(entry, str):
            parts = entry.rsplit(":", 1)
            if len(parts) == 2:
                mount_targets.append(parts[1])
        elif isinstance(entry, dict):
            target = entry.get("target")
            if target:
                mount_targets.append(target)

    assert cursor_dir in mount_targets, (
        f"no compose volume mounts the directory containing {CURSOR_KEY} "
        f"({cursor_dir}); found mount targets: {mount_targets}"
    )
