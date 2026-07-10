# orion/autonomy/tests/test_autonomy_isolation.py
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # orion/autonomy/tests → repo root

# Modules that must not import AutonomyStateV2 / reduce_autonomy_state.
_BANNED_ROOTS = [
    REPO / "orion" / "self_state",
    REPO / "services" / "orion-spark-introspector",
]


def _python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*.py") if p.is_file()]


def _imports_autonomy_v2(path: Path) -> list[str]:
    hits: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return hits
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = {a.name for a in node.names}
            if "AutonomyStateV2" in names or "reduce_autonomy_state" in names:
                hits.append(f"{path}: from {mod} import {sorted(names)}")
            if mod in {"orion.autonomy.models", "orion.autonomy.reducer", "orion.autonomy"}:
                if names & {"AutonomyStateV2", "reduce_autonomy_state", "AutonomyEvidenceRefV1"}:
                    hits.append(f"{path}: from {mod} import {sorted(names)}")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in {"orion.autonomy.reducer", "orion.autonomy.models"}:
                    hits.append(f"{path}: import {a.name}")
    return hits


def test_autonomy_state_v2_not_wired_into_phi_or_self_state() -> None:
    hits: list[str] = []
    for root in _BANNED_ROOTS:
        for path in _python_files(root):
            hits.extend(_imports_autonomy_v2(path))
    assert hits == [], "AutonomyStateV2 isolation violated:\n" + "\n".join(hits)
