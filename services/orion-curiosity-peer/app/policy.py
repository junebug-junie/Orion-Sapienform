"""Read-only Cursor tool policy for contractor peer jobs.

Allowlist is the real gate. Deny names are optional defense-in-depth and must
only use SDK-documented public tool names (unknown names raise at create).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

READ_ONLY_CURSOR_TOOLS: tuple[str, ...] = ("read", "grep", "glob", "ls")

# Documented mutating / expansive capabilities only. Prefer allowlist alone if
# a name drifts and Agent.create starts rejecting it.
PINNED_DISALLOWED_CURSOR_TOOLS: tuple[str, ...] = ("shell", "edit", "task")


def assert_read_only_tools(tools: Sequence[str]) -> None:
    """Raise ValueError if any tool is outside the read-only allowlist."""
    allowed = set(READ_ONLY_CURSOR_TOOLS)
    bad = [t for t in tools if t not in allowed]
    if bad:
        raise ValueError(
            "read-only Cursor tools allowlist violated: "
            + ", ".join(bad)
            + f" (allowed={list(READ_ONLY_CURSOR_TOOLS)})"
        )


def _options_get(options: Any, key: str) -> Any:
    if options is None:
        return None
    if isinstance(options, Mapping):
        return options.get(key)
    return getattr(options, key, None)


def _is_nonempty_collection(value: Any) -> bool:
    """True when value is a present, non-empty mapping/sequence (None/[]/{} ok)."""
    if value is None:
        return False
    if isinstance(value, Mapping):
        return len(value) > 0
    if isinstance(value, (str, bytes)):
        return bool(value)
    if isinstance(value, Sequence):
        return len(value) > 0
    # Fallback for SDK objects that expose __len__ / truthiness.
    try:
        return len(value) > 0  # type: ignore[arg-type]
    except TypeError:
        return bool(value)


def assert_read_only_agent_options(options: Any) -> None:
    """Raise ValueError unless AgentOptions.tools is exactly the allowlist.

    Also rejects cloud agents (tool restrictions are local-only) and any
    disallowed_tools entry that would re-enable nothing but must not include
    allowlisted names.

    Expansive surfaces must stay closed: mcp_servers, agents,
    local.custom_tools, and local.setting_sources must be None or empty.
    """
    if options is None:
        raise ValueError("AgentOptions required for read-only Cursor jobs")

    tools = _options_get(options, "tools")
    if tools is None:
        raise ValueError(
            "tools allowlist required; omitting tools offers the full toolset"
        )
    tool_list = list(tools)
    assert_read_only_tools(tool_list)
    if tuple(tool_list) != READ_ONLY_CURSOR_TOOLS:
        raise ValueError(
            f"tools must be exactly {list(READ_ONLY_CURSOR_TOOLS)}, got {tool_list}"
        )

    cloud = _options_get(options, "cloud")
    if cloud is not None:
        raise ValueError("read-only peer jobs must use local agents, not cloud")

    local = _options_get(options, "local")
    if local is None:
        raise ValueError("local=LocalAgentOptions(...) required for read-only jobs")

    mcp_servers = _options_get(options, "mcp_servers")
    if _is_nonempty_collection(mcp_servers):
        raise ValueError("mcp_servers must be empty/None for read-only peer jobs")

    agents = _options_get(options, "agents")
    if _is_nonempty_collection(agents):
        raise ValueError("agents must be empty/None for read-only peer jobs")

    custom_tools = _options_get(local, "custom_tools")
    if _is_nonempty_collection(custom_tools):
        raise ValueError("local.custom_tools must be empty/None for read-only peer jobs")

    setting_sources = _options_get(local, "setting_sources")
    if _is_nonempty_collection(setting_sources):
        raise ValueError(
            "local.setting_sources must be empty/None for read-only peer jobs"
        )

    disallowed = _options_get(options, "disallowed_tools") or ()
    for name in disallowed:
        if name in READ_ONLY_CURSOR_TOOLS:
            raise ValueError(f"disallowed_tools must not include allowlisted tool {name!r}")
