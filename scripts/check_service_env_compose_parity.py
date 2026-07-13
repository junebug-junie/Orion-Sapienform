#!/usr/bin/env python3
"""Verify a service's docker-compose.yml actually exposes every key its .env_example
declares.

Context: services/orion-recall/docker-compose.yml's `environment:` list was missing
RECALL_GRAPHITI_IN_CHAT / RECALL_GRAPHITI_ADAPTER_URL (and, it turns out, several other
keys) even though .env_example documented them as real, live config. That specific gap
was harmless today only because orion-recall's Dockerfile bakes .env directly into the
image and pydantic-settings reads it via env_file=".env", bypassing docker-compose's
environment: list entirely -- a working accident, not something to rely on. If a
service's Dockerfile pattern ever changes (e.g. to a bind-mount-only build), any key
missing from docker-compose.yml's environment: list silently stops being read. This
script makes that gap visible and gate-able instead of a working accident.

A service is exempt from key-by-key parity if its docker-compose.yml declares an
`env_file:` directive pointing at .env -- in that case every key in .env reaches the
container regardless of the environment: list, and this script reports N/A rather than
scanning for individual keys.

Usage:
    python scripts/check_service_env_compose_parity.py orion-recall
    python scripts/check_service_env_compose_parity.py orion-recall --json
    python scripts/check_service_env_compose_parity.py orion-recall --report-only

Exit codes: 0 = every .env_example key is either exposed via environment: or covered by
                 an env_file: directive.
            1 = at least one key is missing (FAIL) -- unless --report-only.
            2 = could not run the check (missing files, bad service name).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Same convention as scripts/sync_local_env_from_example.py's KEY_LINE.
_ENV_EXAMPLE_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
# Matches docker-compose `environment:` list items: "      - KEY=${KEY}" / "- KEY=value"
_COMPOSE_ENV_KEY_RE = re.compile(r"^\s*-\s*([A-Za-z_][A-Za-z0-9_]*)=")
_ENV_FILE_RE = re.compile(r"^\s*env_file:\s*(.*)$")
_ENV_FILE_LIST_ITEM_RE = re.compile(r"^\s*-\s*(\S+)\s*$")


def _read_env_example_keys(path: Path) -> list[str]:
    keys: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ENV_EXAMPLE_KEY_RE.match(stripped)
        if match:
            keys.append(match.group(1))
    return keys


def _read_compose_env_keys(path: Path) -> tuple[set[str], bool]:
    """Returns (keys exposed via `environment:` list items, has env_file directive)."""
    keys: set[str] = set()
    has_env_file = False
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        env_file_match = _ENV_FILE_RE.match(line)
        if env_file_match:
            rest = env_file_match.group(1).strip()
            if rest and not rest.startswith("#"):
                has_env_file = True
            else:
                # env_file: on its own line, values follow as a YAML list.
                j = i + 1
                while j < len(lines) and _ENV_FILE_LIST_ITEM_RE.match(lines[j]):
                    has_env_file = True
                    j += 1
        compose_match = _COMPOSE_ENV_KEY_RE.match(line)
        if compose_match:
            keys.add(compose_match.group(1))
        i += 1
    return keys, has_env_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("service", help="service directory name under services/, e.g. orion-recall")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="always exit 0, even if keys are missing (report but don't gate).",
    )
    args = parser.parse_args(argv)

    service_dir = _REPO_ROOT / "services" / args.service
    env_example_path = service_dir / ".env_example"
    compose_path = service_dir / "docker-compose.yml"

    if not env_example_path.is_file():
        print(f"check_service_env_compose_parity: missing {env_example_path}", file=sys.stderr)
        return 2
    if not compose_path.is_file():
        print(f"check_service_env_compose_parity: missing {compose_path}", file=sys.stderr)
        return 2

    example_keys = _read_env_example_keys(env_example_path)
    compose_keys, has_env_file = _read_compose_env_keys(compose_path)

    if has_env_file:
        missing: list[str] = []
    else:
        missing = [k for k in example_keys if k not in compose_keys]

    if args.json:
        print(json.dumps({
            "service": args.service,
            "env_example_key_count": len(example_keys),
            "compose_key_count": len(compose_keys),
            "has_env_file_directive": has_env_file,
            "missing_from_compose": missing,
        }))
    else:
        if has_env_file:
            print(
                f"check_service_env_compose_parity: {args.service} declares env_file: -- "
                f"all {len(example_keys)} .env_example keys reach the container regardless "
                f"of environment: list. N/A."
            )
        elif missing:
            print(
                f"check_service_env_compose_parity: {args.service} is missing {len(missing)} "
                f"of {len(example_keys)} .env_example keys from docker-compose.yml's "
                f"environment: list:"
            )
            for key in missing:
                print(f"    {key}")
        else:
            print(
                f"check_service_env_compose_parity: {args.service} OK -- all "
                f"{len(example_keys)} .env_example keys are exposed via environment:."
            )

    if missing and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
