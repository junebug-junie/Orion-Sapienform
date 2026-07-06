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
        # Host-specific Tailscale / mesh address — never overwrite from docker-oriented templates.
        "ORION_BUS_URL",
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
    "ENABLE_PRE_TURN_",
    "ENABLE_POST_TURN_",
    "PRE_TURN_APPRAISAL_",
    "CHANNEL_PRE_TURN_",
    "REPAIR_PRESSURE_WEIGHTS_",
    "REPAIR_PRESSURE_PROBE_",
    "TRANSPORT_PROPOSAL_",
    "TRANSPORT_SUBSTRATE_",
    "HUB_PROPOSAL_REVIEW_",
    "HUB_LLM_GATEWAY_",
    "HUB_AGENT_CONTEXT_EXEC_",
    "HUB_AGENT_CLAUDE_",
    "HUB_AITOWN_",
    "HUB_FCC_",
    "LLM_GATEWAY_OPENAI_",
    "ORION_VECTOR_HOST_",
    "VECTOR_HOST_",
    "HUB_CONTEXT_EXEC_",
    "CONTEXT_EXEC_",
    "CHANNEL_CONTEXT_EXEC_",
    "SELF_EXPERIMENTS_",
    "ACTIONS_SELF_EXPERIMENTS_",
    "CHANNEL_MEMORY_",
    "CORTEX_METACOG_",
    "CORTEX_DAILY_METACOG_",
    "MEMORY_CONSOLIDATION_",
    "MEMORY_BOUNDARY_",
    "MEMORY_CLASSIFY_",
    "MEMORY_SUGGEST_",
    "MEMORY_FAILED_",
    "MEMORY_WINDOW_",
    "SPARK_INTROSPECTION_",
    "TURN_CHANGE_",
    "CHANNEL_SIGNALS_",
    "CHANNEL_VISION_",
    "CHANNEL_EDGE_",
    "CHANNEL_THOUGHT_",
    "CHANNEL_REVERIE_",
    "CHANNEL_HARNESS_",
    "CHANNEL_FINALIZE_",
    "CHANNEL_POST_TURN_",
    "ORION_UNIFIED_",
    "ORION_HARNESS_",
    "ORION_ATTENTION_",
    "ORION_REVERIE_",
    "SUBSTRATE_FELT_STATE_",
    "STANCE_REACT_",
    "HARNESS_FCC_",
    "HARNESS_AITOWN_",
    "HARNESS_GOVERNOR_",
    "FINALIZE_REFLECT_",
    "VOICE_FINALIZE_",
    "SUBSTRATE_FINALIZE_",
    "FINALIZE_QUICK_",
    "THOUGHT_",
    "EDGE_ACTIVITY_",
    "EDGE_PUBLISH_",
    "VISION_VLM_",
    "COUNCIL_",
    "HOST_PORT",
    "SQL_WRITER_EMIT_MEMORY_",
)

SYNC_EXACT = frozenset(
    {
        "PROJECT",
        "AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES",
        "AUTONOMY_GRAPH_QUERY_URL",
        "AUTONOMY_GRAPH_UPDATE_URL",
        "AUTONOMY_GRAPH_TIMEOUT_SEC",
        "CHANNEL_MEMORY_TURN_PERSISTED",
        "CHANNEL_CHAT_HISTORY_SPARK_META_PATCH",
        "SQL_WRITER_EMIT_MEMORY_TURN_PERSISTED",
        "MEMORY_CONSOLIDATION_ENABLED",
        "LLM_LOGPROB_SUMMARY_ENABLED",
        "LLM_GATEWAY_OPENAI_PASSTHROUGH_ENABLED",
        "HOST_PORT",
        # Reverie/dream weave — Phase F (orion-dream REM compaction, default-off).
        "ORION_DREAM_REM_ENABLED",
        "CHANNEL_DREAM_COMPACTION_DELTA",
        "DREAM_REM_MAX_REQUESTS",
        # Phase G (compaction applier — the hot gate, hard-off, mutates memory).
        "ORION_DREAM_COMPACTION_APPLY_ENABLED",
        "ORION_DREAM_COMPACTION_DOWNSCALE_ONLY",
        "DREAM_COMPACTION_SNAPSHOT_DIR",
    }
)

DEFAULT_SERVICES = (
    "orion-memory-crystallizer",
    "orion-memory-consolidation",
    "orion-sql-writer",
    "orion-graphiti-adapter",
    "orion-hub",
    "orion-cortex-exec",
    "orion-context-exec",
    "orion-self-experiments",
    "orion-actions",
    "orion-spark-concept-induction",
    "orion-spark-introspector",
    # Transport substrate stack (M3–M7)
    "orion-substrate-runtime",
    "orion-field-digester",
    "orion-attention-runtime",
    "orion-self-state-runtime",
    "orion-proposal-runtime",
    "orion-thought",
    "orion-harness-governor",
    "orion-dream",
    "orion-llm-gateway",
    "orion-vector-host",
)


def should_sync_key(key: str, *, all_keys: bool) -> bool:
    if key in NEVER_SYNC_KEYS:
        return False
    if all_keys:
        return True
    if key in SYNC_EXACT:
        return True
    return any(key.startswith(p) for p in SYNC_PREFIXES)


def example_value_is_host_placeholder(key: str, value: str) -> bool:
    """Skip syncing template placeholders that must stay host-specific in local .env."""
    v = value.strip().strip("'\"")
    if key == "ORION_BUS_URL":
        lowered = v.lower()
        if "x.x.x" in lowered or "x.x.x.x" in lowered:
            return True
        if lowered in {
            "redis://bus-core:6379/0",
            "redis://localhost:6379/0",
            "redis://redis:6379/0",
            "${orion_bus_url}",
        }:
            return True
    return False


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
        if should_sync_key(k, all_keys=all_keys) and not example_value_is_host_placeholder(k, v)
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
