from __future__ import annotations

import ast
from pathlib import Path

RETINA_APP = Path(__file__).resolve().parents[1] / "services" / "orion-vision-retina" / "app"

BANNED_IMPORT_ROOTS = {
    "ultralytics",
    "torch",
    "torchvision",
}
BANNED_SYMBOLS = {
    "VisionEdgeArtifact",
}


def _walk_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_retina_has_no_detector_imports() -> None:
    found: set[str] = set()
    for py in RETINA_APP.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        found |= _walk_imports(tree)
    assert not (found & BANNED_IMPORT_ROOTS)
    assert "detector_worker" not in found


def test_retina_source_no_banned_symbols() -> None:
    text = "\n".join(p.read_text(encoding="utf-8") for p in RETINA_APP.glob("*.py"))
    for sym in BANNED_SYMBOLS:
        assert sym not in text
