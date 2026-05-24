from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN = ("cv2", "ultralytics", "yolo", "detector_worker")

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def test_router_does_not_import_forbidden_modules() -> None:
    hits: list[str] = []
    for path in APP_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in FORBIDDEN:
                        hits.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in FORBIDDEN:
                    hits.append(f"{path.name}: from {node.module}")
    assert hits == []
