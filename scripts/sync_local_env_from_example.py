#!/usr/bin/env python3
"""Sync selected keys from services/*/.env_example into local .env (never commits .env).

Default: only keys we routinely change during feature work (memory graph, autonomy, goal archive).
Use --all-keys to mirror every example key (skips host-specific overrides below).

Default overwrite behavior:
- A key missing from local .env but present in .env_example is always auto-added.
  There is no existing value to clobber, so this is safe.
- A key that already exists in local .env with a value that differs from
  .env_example is treated as a possible *intentional* deployment-specific
  override (e.g. a live backend flipped on for real, a host-specific tuning
  value) and is NOT silently overwritten. It is reported as "diverged" so an
  operator can review it by hand.
- Pass --force to restore the old behavior and overwrite diverged keys too
  (use this when a key's meaning genuinely changed upstream and every host's
  local value should follow .env_example).

NEVER_SYNC_KEYS keys are excluded from sync entirely (even with --force) and
are a separate, stricter mechanism from the diverged/--force behavior above.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
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
    "ACTIVATION_",
    "MEMORY_FORMATION_",
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
    "SIGNALS_",
    "SIGNAL_GATEWAY_",
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
    # Drive attribution + substrate act (PR drive-attribution-substrate-act)
    "ORION_SUBSTRATE_",
    "ORION_AUTONOMY_",
    "ORION_METABOLISM_",
    "ORION_CAPABILITY_POLICY_",
    "ORION_EPISODE_",
    "STANCE_REACT_",
    # Mind stance enrichment (unified turn; orion-thought → orion-mind, default-off)
    "ORION_THOUGHT_MIND_",
    "ORION_MIND_",
    "HARNESS_FCC_",
    "ORION_FCC_",
    "HARNESS_AITOWN_",
    "HARNESS_GOVERNOR_",
    "FCC_",
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
    # Embodiment (mind-to-sprite bridge, default-off)
    "EMBODIMENT_",
    # Durable world-pulse run-result stream (producer dual-write + consumer group)
    "WP_RUN_RESULT_",
    # Plan 2 phi encoder / inner-state features (seed-v2)
    "INNER_FEATURES_",
    "PHI_DEGENERATE_",
    "PHI_ENCODER_",
    "ORION_PHI_ENCODER_",
    "SUBSTRATE_READ_",
    "EXEC_TRAJECTORY_",
    # Mood-arc corpus collector (roadmap item 1, 2026-07-13)
    "MOOD_ARC_",
    # Corpus sink rotation/retention (shared by inner-state + mood-arc sinks)
    "CORPUS_SINK_",
    # Reasoning telemetry adapter (cortex-exec -> orion-thought -> phi, default-off)
    "REASONING_ACTIVITY_",
    # Corpus hygiene: execution_trajectory projection cap (orion-substrate-runtime)
    "EXECUTION_TRAJECTORY_",
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
        # Mind run artifact channel (reused by unified-turn enrichment, mode=orion)
        "CHANNEL_MIND_ARTIFACT",
        "SQL_WRITER_EMIT_MEMORY_TURN_PERSISTED",
        "MEMORY_CONSOLIDATION_ENABLED",
        "LLM_LOGPROB_SUMMARY_ENABLED",
        "LLM_GATEWAY_OPENAI_PASSTHROUGH_ENABLED",
        "HOST_PORT",
        "HARNESS_REPO_MOUNT",
        "HARNESS_AITOWN_CONVEX_URL",
        # Embodiment master switch (prefix is EMBODIMENT_, so the ORION_ master flag needs an exact entry).
        "ORION_EMBODIMENT_ENABLED",
        # Reverie/dream weave — Phase F (orion-dream REM compaction, default-off).
        "ORION_DREAM_REM_ENABLED",
        "CHANNEL_DREAM_COMPACTION_DELTA",
        "DREAM_REM_MAX_REQUESTS",
        # Phase G (compaction applier — the hot gate, hard-off, mutates memory).
        "ORION_DREAM_COMPACTION_APPLY_ENABLED",
        "ORION_DREAM_COMPACTION_DOWNSCALE_ONLY",
        "DREAM_COMPACTION_SNAPSHOT_DIR",
        "GOAL_DRIVE_ORIGIN_SOURCE",
        # Keep town/embodiment journals out of Juniper's inbox (email gate, keeps journaling).
        "ACTIONS_JOURNAL_POST_PERSIST_EMAIL_EXCLUDE_SOURCE_KINDS",
        # Worker log level (stdlib->loguru bridge) — must stay in sync so traces are visible.
        "LOG_LEVEL",
        # Plan 2 phi encoder / inner-state features
        "SUBSTRATE_RUNTIME_URL",
        "CHANNEL_INNER_FEATURES",
        "CHANNEL_PHI_REWARD",
        # Reasoning telemetry adapter (cortex-exec -> orion-thought -> phi, default-off)
        "PUBLISH_REASONING_TELEMETRY",
        "CHANNEL_REASONING_CALL",
        # Seed-v4 feature set: spark-introspector reads orion-thought's reasoning_activity projection
        "ORION_THOUGHT_BASE_URL",
        # Voluntary attention goal-context listener (orion-substrate-runtime)
        "CHANNEL_GOAL_PROPOSAL",
        # Unified-turn harness governor RPC wait budget (hub <-> orion-harness-governor)
        "HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC",
        "HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC",
        "HUB_HARNESS_GOVERNOR_LIVENESS_WINDOW_SEC",
        "HUB_HARNESS_STEP_RELAY_LIVENESS_TTL_SEC",
        "HUB_HARNESS_STEP_RELAY_LIVENESS_MAX_ENTRIES",
    }
)

DEFAULT_SERVICES = (
    "orion-memory-crystallizer",
    "orion-memory-consolidation",
    "orion-recall",
    "orion-sql-writer",
    "orion-graphiti-adapter",
    "orion-hub",
    "orion-cortex-exec",
    "orion-context-exec",
    "orion-self-experiments",
    "orion-actions",
    "orion-spark-concept-induction",
    "orion-spark-introspector",
    "orion-world-pulse",
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
    "orion-fcc",
    "orion-vector-host",
    "orion-embodiment",
    "orion-signals",
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


@dataclass
class SyncResult:
    """Result of syncing one service's .env from its .env_example.

    updated: keys actually written (missing keys auto-added, plus diverged
        keys overwritten only when force=True).
    diverged: keys that exist locally with a value different from
        .env_example and were left untouched (force=False). Informational —
        not an error.
    """

    updated: list[str] = field(default_factory=list)
    diverged: list[str] = field(default_factory=list)


def sync_file(
    env_path: Path,
    example_path: Path,
    *,
    dry_run: bool,
    all_keys: bool,
    force: bool = False,
) -> SyncResult:
    name = env_path.parent.name
    if not example_path.is_file():
        return SyncResult(updated=[f"skip {name}: no .env_example"])
    if not env_path.is_file():
        return SyncResult(updated=[f"skip {name}: no .env"])

    desired = {
        k: v
        for k, v in parse_kv(example_path).items()
        if should_sync_key(k, all_keys=all_keys) and not example_value_is_host_placeholder(k, v)
    }
    if not desired:
        return SyncResult()

    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    seen: set[str] = set()
    updated: list[str] = []
    diverged: list[str] = []
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
            if force:
                updated.append(f"{name}: {key} {old!r} -> {desired[key]!r}")
                out_lines.append(f"{key}={desired[key]}\n")
            else:
                diverged.append(f"{name}: {key} local={old!r} example={desired[key]!r}")
                out_lines.append(raw if raw.endswith("\n") else raw + "\n")
        else:
            out_lines.append(raw if raw.endswith("\n") else raw + "\n")

    missing = [k for k in desired if k not in seen]
    if missing:
        out_lines.append("\n# synced from .env_example\n")
        for key in missing:
            updated.append(f"{name}: +{key}={desired[key]!r}")
            out_lines.append(f"{key}={desired[key]}\n")

    if updated and not dry_run:
        env_path.write_text("".join(out_lines), encoding="utf-8")
    return SyncResult(updated=updated, diverged=diverged)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("services", nargs="*", help=f"Default: {', '.join(DEFAULT_SERVICES)}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--all-keys",
        action="store_true",
        help="Sync every .env_example key (still skips NEVER_SYNC_KEYS)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite local .env values that diverge from .env_example instead of "
            "just reporting them. Use when a key's meaning genuinely changed upstream. "
            "NEVER_SYNC_KEYS are still excluded even with --force."
        ),
    )
    args = parser.parse_args()

    services_dir = ROOT / "services"
    names = args.services or list(DEFAULT_SERVICES)
    all_updated: list[str] = []
    all_diverged: list[str] = []
    for name in names:
        svc = services_dir / name
        if not svc.is_dir():
            print(f"unknown service: {name}", file=sys.stderr)
            return 1
        result = sync_file(
            svc / ".env",
            svc / ".env_example",
            dry_run=args.dry_run,
            all_keys=args.all_keys,
            force=args.force,
        )
        all_updated.extend(result.updated)
        all_diverged.extend(result.diverged)

    if not all_updated and not all_diverged:
        print("No changes needed (feature keys already match .env_example).")
        return 0

    if all_updated:
        prefix = "Would update" if args.dry_run else "Updated"
        print(prefix + ":")
        for line in all_updated:
            print(f"  {line}")
    if all_diverged:
        print(
            "Diverged (not changed — local value differs from .env_example, "
            "review manually or pass --force):"
        )
        for line in all_diverged:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
