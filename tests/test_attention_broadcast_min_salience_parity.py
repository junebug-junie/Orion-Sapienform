"""ORION_ATTENTION_BROADCAST_MIN_SALIENCE must have ONE default everywhere.

The floor a substrate node's salience must clear to enter the attention
competition was lowered 0.2 -> 0.05 on 2026-09-05 (commit 263b762d6) in
.env_example only. The module constant, settings.py and the compose fallback
stayed at 0.2 for five weeks, so any run without the .env value (tests, a
fresh host, a dropped key) silently admitted about a third as many
competitors. Static parse only: no service imports, so this runs in the
lightweight static-gates job.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KEY = "ORION_ATTENTION_BROADCAST_MIN_SALIENCE"
SERVICE = REPO / "services" / "orion-substrate-runtime"


def _module_default() -> float:
    tree = ast.parse((REPO / "orion/substrate/attention_broadcast.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DEFAULT_MIN_SALIENCE" for t in node.targets
        ):
            return float(ast.literal_eval(node.value))
    raise AssertionError("DEFAULT_MIN_SALIENCE not found in attention_broadcast.py")


def _settings_default() -> float:
    tree = ast.parse((SERVICE / "app/settings.py").read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Field"):
            continue
        if any(
            kw.arg == "alias" and isinstance(kw.value, ast.Constant) and kw.value.value == KEY
            for kw in node.keywords
        ):
            return float(ast.literal_eval(node.args[0]))
    raise AssertionError(f"Field(alias={KEY!r}) not found in settings.py")


def _compose_fallback() -> float:
    text = (SERVICE / "docker-compose.yml").read_text()
    hits = re.findall(r"\$\{" + KEY + r":-([^}]*)\}", text)
    assert len(hits) == 1, f"expected one {KEY} fallback in docker-compose.yml, got {hits}"
    return float(hits[0])


def _env_example_value() -> float:
    hits = re.findall(rf"^{KEY}=(.*)$", (SERVICE / ".env_example").read_text(), re.M)
    assert len(hits) == 1, f"expected one {KEY} line in .env_example, got {hits}"
    return float(hits[0].strip())


def test_min_salience_default_is_identical_everywhere() -> None:
    values = {
        "orion/substrate/attention_broadcast.py DEFAULT_MIN_SALIENCE": _module_default(),
        "services/orion-substrate-runtime/app/settings.py": _settings_default(),
        "services/orion-substrate-runtime/docker-compose.yml fallback": _compose_fallback(),
        "services/orion-substrate-runtime/.env_example": _env_example_value(),
    }
    assert len(set(values.values())) == 1, f"{KEY} default drifted: {values}"
