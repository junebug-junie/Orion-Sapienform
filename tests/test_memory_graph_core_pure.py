"""Memory-graph core stays deterministic: no LLM vendor SDKs or bus client imports."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "orion" / "memory_graph"
BANNED_TOP_LEVEL = frozenset({"openai", "anthropic", "litellm"})


def _import_top_levels(source: str) -> set[str]:
    roots: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_memory_graph_core_modules_have_no_llm_vendor_or_bus_imports() -> None:
    for name in ("json_to_rdf.py", "validate.py", "project.py", "graphdb.py", "approve.py"):
        path = CORE_DIR / name
        text = path.read_text(encoding="utf-8")
        assert "orion.core.bus" not in text, f"{name} must not import bus stack"
        overlap = _import_top_levels(text) & BANNED_TOP_LEVEL
        assert not overlap, f"{name}: banned imports {sorted(overlap)}"


def test_json_extract_has_no_nonstdlib_imports() -> None:
    path = CORE_DIR / "json_extract.py"
    levels = _import_top_levels(path.read_text(encoding="utf-8"))
    assert levels.issubset({"__future__", "typing"}), levels
