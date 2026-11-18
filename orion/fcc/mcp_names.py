"""MCP server keys and the exact tool names Claude Code derives from them.

Claude Code exposes a tool as ``mcp__<server>__<tool>`` and ``ToolSearch
select:`` only matches that exact string. Turn briefs that named tools bare
("curiosity") left the model guessing a prefix; a wrong guess returned "No
matching deferred tools" and the turn concluded the server was down
(2026-10-10). Briefs build names here; render_mcp_config uses the same keys.
"""
from __future__ import annotations

GITHUB = "github"
FIRECRAWL = "firecrawl"
ORION_READING = "orion-reading"
ORION_INTROSPECT = "orion-introspect"
ORION_AITOWN = "orion-aitown"
GITNEXUS = "gitnexus"
CONTEXT_MODE = "context-mode"
# Hook mode: the context-mode Claude Code plugin registers its own server,
# which Claude Code names plugin_<plugin>_<server>.
CONTEXT_MODE_PLUGIN = "plugin_context-mode_context-mode"


def mcp_tool(server: str, tool: str) -> str:
    return f"mcp__{server}__{tool}"
