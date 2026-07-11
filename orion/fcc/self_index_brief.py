"""FCC tool brief for the semantic self-indexing MCPs (GitNexus, Context Mode).

Kept out of github_repo_context.py so the GitHub coordinate logic stays narrow.
Lines are appended to the harness motor prefix only when the corresponding
feature flag is on, mirroring append_github_mcp_harness_brief.
"""

from __future__ import annotations

import os


def _env_truthy(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}


def gitnexus_enabled() -> bool:
    return _env_truthy("HARNESS_FCC_GITNEXUS_ENABLED")


def context_mode_enabled() -> bool:
    return _env_truthy("HARNESS_FCC_CONTEXT_MODE_ENABLED")


def gitnexus_brief_lines() -> list[str]:
    return [
        (
            "GitNexus code-graph MCP is available for repository introspection. "
            "Use query to locate a concept or process, context for the 360-degree "
            "view of one symbol, impact for upstream/downstream blast radius, and "
            "trace only between two known endpoints. Request full symbol content "
            "only after narrowing."
        ),
        (
            "The graph is derived cache, never authority. Before graph-grounded "
            "claims, read the GitNexus repo status/context resource; if the index "
            "is stale, disclose that and fall back to source search — never present "
            "stale structure as current truth. Verify authority claims in source "
            "and tests before asserting them."
        ),
    ]


def context_mode_brief_lines() -> list[str]:
    return [
        (
            "Context Mode MCP is available. Route bulk file reads, command output, "
            "and multi-file analysis through ctx_batch_execute / ctx_execute so raw "
            "output stays out of the live context; recover exact omitted evidence "
            "later with ctx_search instead of re-running bulk queries."
        ),
    ]


def append_self_index_harness_brief(parts: list[str]) -> None:
    """Append GitNexus/Context Mode usage lines when their harness flags are on.

    Gated on the master MCP flag first: without HARNESS_FCC_MCP_ENABLED no MCP
    config is rendered at all (fcc_motor._maybe_render_mcp_config returns None),
    so advertising these tools would point the motor at servers that don't exist.
    """
    from orion.fcc.github_repo_context import harness_mcp_enabled

    if not harness_mcp_enabled():
        return
    if gitnexus_enabled():
        parts.extend(gitnexus_brief_lines())
    if context_mode_enabled():
        parts.extend(context_mode_brief_lines())
