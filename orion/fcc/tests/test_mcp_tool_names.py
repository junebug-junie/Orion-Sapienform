"""Turn briefs name MCP tools exactly as Claude Code exposes them.

ToolSearch ``select:`` matches only ``mcp__<server>__<tool>``. A brief that
named "curiosity" bare led a live turn (2026-10-10) to guess
``orion-introspect__curiosity``, get "No matching deferred tools", and
conclude a working server was down.
"""
from __future__ import annotations

import json
import re

import pytest

from orion.fcc import github_repo_context, mcp_names
from orion.fcc.mcp_config import _TEMPLATE_PATH
from orion.fcc.self_index_brief import context_mode_brief_lines, gitnexus_brief_lines
from orion.introspect.brief import introspect_brief_lines
from orion.introspect.tools import IntrospectTools
from orion.schemas.introspect import IntrospectToolBindingV1
from orion.world_pulse_read.tools import reading_brief_lines

_FULL_NAME = re.compile(r"mcp__([A-Za-z0-9_\-]+?)__([A-Za-z0-9_]+)")

# Third-party servers: names confirmed in live harness transcripts 2026-10-10.
_GITHUB_TOOLS = {"list_pull_requests", "get_pull_request", "search_pull_requests"}
_GITNEXUS_TOOLS = {"query", "context", "impact", "trace"}
_CTX_TOOLS = {"ctx_execute", "ctx_batch_execute", "ctx_search"}
_READING_TOOLS = {"recommend_reading", "reading_status"}


def _binding() -> IntrospectToolBindingV1:
    return IntrospectToolBindingV1(
        invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=True,
    )


def _known_tools() -> dict[str, set[str]]:
    introspect = {spec.name for spec in IntrospectTools(None, _binding()).tool_specs()}
    return {
        mcp_names.GITHUB: _GITHUB_TOOLS,
        mcp_names.GITNEXUS: _GITNEXUS_TOOLS,
        mcp_names.CONTEXT_MODE: _CTX_TOOLS,
        mcp_names.CONTEXT_MODE_PLUGIN: _CTX_TOOLS,
        mcp_names.ORION_READING: _READING_TOOLS,
        mcp_names.ORION_INTROSPECT: introspect,
    }


def _all_brief_text(monkeypatch, *, hooks: bool) -> str:
    monkeypatch.setattr(
        github_repo_context, "resolve_github_repo_coordinate", lambda workspace=None: ("o", "r"),
    )
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "true" if hooks else "")
    lines = (
        github_repo_context.github_mcp_brief_lines()
        + gitnexus_brief_lines()
        + context_mode_brief_lines()
        + reading_brief_lines()
        + introspect_brief_lines(_binding())
    )
    return "\n".join(lines)


@pytest.mark.parametrize("hooks", [False, True])
def test_every_full_name_in_briefs_resolves_to_a_real_tool(monkeypatch, hooks):
    text = _all_brief_text(monkeypatch, hooks=hooks)
    known = _known_tools()
    found = _FULL_NAME.findall(text)
    assert found
    for server, tool in found:
        assert server in known, f"unknown server in brief: mcp__{server}__{tool}"
        assert tool in known[server], f"unknown tool in brief: mcp__{server}__{tool}"


@pytest.mark.parametrize("hooks", [False, True])
def test_no_tool_is_named_bare(monkeypatch, hooks):
    text = _all_brief_text(monkeypatch, hooks=hooks)
    bare_candidates = set().union(_GITHUB_TOOLS, _CTX_TOOLS, _READING_TOOLS, {"reading_results"})
    for tool in bare_candidates:
        bare = re.findall(rf"(?<!__)\b{tool}\b", text)
        assert not bare, f"{tool} named without its mcp__<server>__ prefix"


def test_introspect_brief_lists_every_tool_by_exact_name():
    text = "\n".join(introspect_brief_lines(_binding()))
    for spec in IntrospectTools(None, _binding()).tool_specs():
        assert mcp_names.mcp_tool(mcp_names.ORION_INTROSPECT, spec.name) in text
    assert "orion-introspect__curiosity" not in text.replace("mcp__orion-introspect__curiosity", "")


def test_context_mode_brief_follows_the_active_server(monkeypatch):
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "true")
    assert "mcp__plugin_context-mode_context-mode__ctx_execute" in context_mode_brief_lines()[0]
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "")
    assert "mcp__context-mode__ctx_execute" in context_mode_brief_lines()[0]


def test_template_server_keys_match_names_module():
    template = json.loads(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert set(template["mcpServers"]) == {mcp_names.GITHUB, mcp_names.FIRECRAWL}
