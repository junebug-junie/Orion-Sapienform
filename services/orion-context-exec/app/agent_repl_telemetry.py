"""Deterministic semantic tool detection for agent_repl code_action telemetry."""

from __future__ import annotations

import ast
import re

AGENT_REPL_SEMANTIC_TOOLS = frozenset({
    "repo_grep",
    "repo_read",
    "repo_read_range",
    "repo_find_files",
    "repo_tree",
    "repo_outline",
    "repo_list",
    "patch_validate",
    "workspace_write",
    "workspace_read",
    "workspace_list",
    "workspace_write_patch",
    "workspace_write_report",
    "recall_query",
})

# Word-boundary tool name followed by '(' — conservative regex fallback.
_REGEX_TOOL = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in sorted(AGENT_REPL_SEMANTIC_TOOLS, key=len, reverse=True)) + r")\s*\("
)


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _detect_via_ast(code: str) -> list[str]:
    tree = ast.parse(code)
    hits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name and name in AGENT_REPL_SEMANTIC_TOOLS:
            hits.append((node.lineno, node.col_offset, name))
    hits.sort()
    seen: set[str] = set()
    ordered: list[str] = []
    for _, _, name in hits:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _detect_via_regex(code: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _REGEX_TOOL.finditer(code):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def detect_semantic_tools_from_code(code: str) -> list[str]:
    """Return registered workbench tool names found in *code*, first appearance order."""
    if not code or not code.strip():
        return []
    try:
        return _detect_via_ast(code)
    except SyntaxError:
        return _detect_via_regex(code)
    except Exception:
        return []
