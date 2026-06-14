#!/usr/bin/env python3
"""Sync selected keys from services/*/.env_example into local .env (never commits .env).

Default: only keys we routinely change during feature work (memory graph, autonomy, goal archive).
Use --all-keys to mirror every example key (skips host-specific overrides below).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# Host .env may legitimately differ from docker-oriented .env_example.
NEVER_SYNC_KEYS = frozenset(
    {
        "ORION_KNOWLEDGE_ROOT",
        "PUBLISH_CORTEX_EXEC_GRAMMAR",
    }
)

# Prefixes / exact keys synced after .env_example edits (default mode).
SYNC_PREFIXES = (
    "CRYSTALLIZER_",
    "GRAPHITI_",
    "FALKORDB_",
    "CHROMA_",
    "GRAPHITI_ADAPTER_",
    "MEMORY_GRAPH_SUGGEST_",
    "LLM_MEMORY_GRAPH_SUGGEST_",
    "AUTONOMY_CHAT_STANCE_",
    "AUTONOMY_DRIVES_SUBQUERY_",
    "AUTONOMY_GOAL_ARCHIVE_",
    "AUTONOMY_GOAL_RETENTION_",
    "AUTONOMY_GOAL_MAX_ACTIVE_",
    "ACTIONS_DAILY_GOAL_ARCHIVE_",
    # Transport substrate stack (PR #648 / M3–M7)
    "ENABLE_TRANSPORT_",
    "TRANSPORT_PROPOSAL_",
    "TRANSPORT_SUBSTRATE_",
    "CONTEXT_EXEC_",
    "CHANNEL_CONTEXT_EXEC_",
    "CORTEX_METACOG_",
    "CORTEX_DAILY_METACOG_",
)

SYNC_EXACT = frozenset(
    {
        "PROJECT",
        "AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES",
        "AUTONOMY_GRAPH_QUERY_URL",
        "AUTONOMY_GRAPH_UPDATE_URL",
        "AUTONOMY_GRAPH_TIMEOUT_SEC",
    }
)

DEFAULT_SERVICES = (
    "orion-memory-crystallizer",
    "orion-graphiti-adapter",
    "orion-hub",
    "orion-cortex-exec",
    "orion-context-exec",
    "orion-actions",
    "orion-spark-concept-induction",
    # Transport substrate stack (M3–M7)
    "orion-substrate-runtime",
    "orion-field-digester",
    "orion-attention-runtime",
    "orion-self-state-runtime",
    "orion-proposal-runtime",
)


def should_sync_key(key: str, *, all_keys: bool) -> bool:
    if key in NEVER_SYNC_KEYS:
        return False
    if all_keys:
        return True
    if key in SYNC_EXACT:
        return True
    return any(key.startswith(p) for p in SYNC_PREFIXES)


def parse_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = KEY_LINE.match(s)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def sync_file(
    env_path: Path,
    example_path: Path,
    *,
    dry_run: bool,
    all_keys: bool,
) -> list[str]:
    if not example_path.is_file():
        return [f"skip {env_path.parent.name}: no .env_example"]
    if not env_path.is_file():
        return [f"skip {env_path.parent.name}: no .env"]

    desired = {
        k: v
        for k, v in parse_kv(example_path).items()
        if should_sync_key(k, all_keys=all_keys)
    }
    if not desired:
        return []

    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    seen: set[str] = set()
    changed: list[str] = []
    out_lines: list[str] = []

    for raw in lines:
        stripped = raw.rstrip("\n")
        m = KEY_LINE.match(stripped)
        if not m:
            out_lines.append(raw if raw.endswith("\n") else raw + "\n")
            continue
        key, old = m.group(1), m.group(2)
        seen.add(key)
        if key in desired and desired[key] != old:
            changed.append(f"{env_path.parent.name}: {key} {old!r} -> {desired[key]!r}")
            out_lines.append(f"{key}={desired[key]}\n")
        else:
            out_lines.append(raw if raw.endswith("\n") else raw + "\n")

    missing = [k for k in desired if k not in seen]
    if missing:
        out_lines.append("\n# synced from .env_example\n")
        for key in missing:
            changed.append(f"{env_path.parent.name}: +{key}={desired[key]!r}")
            out_lines.append(f"{key}={desired[key]}\n")

    if changed and not dry_run:
        env_path.write_text("".join(out_lines), encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("services", nargs="*", help=f"Default: {', '.join(DEFAULT_SERVICES)}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--all-keys",
        action="store_true",
        help="Sync every .env_example key (still skips NEVER_SYNC_KEYS)",
    )
    args = parser.parse_args()

    services_dir = ROOT / "services"
    names = args.services or list(DEFAULT_SERVICES)
    all_changes: list[str] = []
    for name in names:
        svc = services_dir / name
        if not svc.is_dir():
            print(f"unknown service: {name}", file=sys.stderr)
            return 1
        all_changes.extend(
            sync_file(svc / ".env", svc / ".env_example", dry_run=args.dry_run, all_keys=args.all_keys)
        )

    if not all_changes:
        print("No changes needed (feature keys already match .env_example).")
        return 0
    prefix = "Would update" if args.dry_run else "Updated"
    print(prefix + ":")
    for line in all_changes:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
