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
`env_file:` directive (string, list-of-strings, or the extended `- path: .env` mapping
form all count) -- in that case every key in .env reaches the container regardless of
the environment: list, and this script reports N/A rather than scanning for individual
keys.

Uses real YAML parsing (PyYAML, already a repo dependency -- see
scripts/check_inner_state_registry.py / scripts/check_single_consumer_channels.py for
the same import convention) rather than regex line-scanning, specifically so multi-
service compose files and every env_file: syntax form are handled correctly instead of
silently mis-parsed.

Usage:
    python scripts/check_service_env_compose_parity.py orion-recall
    python scripts/check_service_env_compose_parity.py orion-recall --json
    python scripts/check_service_env_compose_parity.py orion-recall --report-only

Exit codes: 0 = every .env_example key is either exposed via environment: or covered by
                 an env_file: directive.
            1 = at least one key is missing (FAIL) -- unless --report-only.
            2 = could not run the check (missing files, bad service name, ambiguous
                multi-service compose file -- see ServiceSelectionError).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/check_service_env_compose_parity.py` puts scripts/ on
# sys.path[0], which shadows stdlib modules (same issue documented in
# scripts/check_inner_state_registry.py / scripts/check_activation_saturation.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)
# Re-added (not at position 0) so the sibling-script import below resolves, matching
# tests/test_concept_relation_digest.py's flat-import-off-scripts/ convention.
if _SCRIPT_DIR not in sys.path:
    sys.path.append(_SCRIPT_DIR)

_REPO_ROOT = Path(__file__).resolve().parents[1]

from sync_local_env_from_example import KEY_LINE as _ENV_EXAMPLE_KEY_RE  # noqa: E402


class ServiceSelectionError(ValueError):
    """Raised when the target service can't be unambiguously identified inside a
    multi-service compose file."""


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


def _select_service_block(compose: dict[str, Any], service_dirname: str) -> tuple[str, dict[str, Any]]:
    """Pick the compose `services:` entry that corresponds to `service_dirname`
    (e.g. "orion-recall"). Single-service files (the norm in this repo) are
    unambiguous. Multi-service files are resolved by exact name or by the
    "orion-"-stripped suffix (services/orion-recall's compose key is "recall")."""
    services = compose.get("services") or {}
    if not services:
        raise ServiceSelectionError(f"no 'services:' block found for {service_dirname}")
    if len(services) == 1:
        name = next(iter(services))
        return name, services[name] or {}

    suffix = re.sub(r"^orion-", "", service_dirname)
    candidates = [name for name in services if name in (service_dirname, suffix)]
    if len(candidates) == 1:
        return candidates[0], services[candidates[0]] or {}

    raise ServiceSelectionError(
        f"compose file declares {len(services)} services ({', '.join(sorted(services))}) and "
        f"none unambiguously matches '{service_dirname}' (tried '{service_dirname}' and "
        f"'{suffix}'). This checker only supports single-service compose files or ones where "
        f"a service key matches the directory name."
    )


def _read_compose_env_keys(path: Path, service_dirname: str) -> tuple[set[str], bool]:
    """Returns (keys exposed via `environment:`, has env_file directive) for the
    service block matching `service_dirname` inside `path`."""
    import yaml

    compose = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _, block = _select_service_block(compose, service_dirname)

    keys: set[str] = set()
    env = block.get("environment")
    if isinstance(env, dict):
        keys.update(str(k) for k in env.keys())
    elif isinstance(env, list):
        for item in env:
            if isinstance(item, str) and "=" in item:
                keys.add(item.split("=", 1)[0].strip())

    # env_file: accepts a bare string, a list of strings, or a list of mappings
    # ({"path": ..., "required": ...}) -- any non-empty form means every key in
    # that file reaches the container regardless of the environment: list above.
    has_env_file = bool(block.get("env_file"))

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
    try:
        compose_keys, has_env_file = _read_compose_env_keys(compose_path, args.service)
    except ServiceSelectionError as exc:
        print(f"check_service_env_compose_parity: {exc}", file=sys.stderr)
        return 2

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
