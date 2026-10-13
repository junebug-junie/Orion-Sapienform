"""The proposal runtime image must install every third-party module its import chain needs.

Regression for 2026-10-13: orion.proposals.builder._is_internal_target imports
orion.attention.world_first, which pulls in orion.substrate.graphdb_store -> `requests`.
The image lacked `requests`, so every tick with a node target raised ModuleNotFoundError.
"""

from __future__ import annotations

import ast
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]


def _requirement_names() -> set[str]:
    names: set[str] = set()
    for raw in (SERVICE / "requirements.txt").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for sep in ("==", ">=", "<=", "~=", "[", "<", ">"):
            line = line.split(sep, 1)[0]
        names.add(line.strip().lower())
    return names


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_requests_is_installed_for_the_world_first_import_chain() -> None:
    graphdb_store = REPO / "orion" / "substrate" / "graphdb_store.py"
    assert "requests" in _top_level_imports(graphdb_store), "chain changed; revisit this test"
    assert "requests" in _requirement_names()
