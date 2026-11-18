"""Gate: app.js's fail-closed @claude pattern matches claude-address.js.

app.js carries a copy of the address pattern so an "@claude ..." message never
reaches Orion when claude-address.js fails to load. Two copies of one rule can
drift; this pins them equal.
"""
from __future__ import annotations

import re
from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "static" / "js"


def _pattern(path: Path, name: str) -> str:
    match = re.search(rf"const {name} = (/.+?/[a-z]*);", path.read_text(encoding="utf-8"))
    assert match, f"{name} not found in {path.name}"
    return match.group(1)


def test_fallback_pattern_equals_module_pattern() -> None:
    module = _pattern(JS / "claude-address.js", "ADDRESS")
    fallback = _pattern(JS / "app.js", "CLAUDE_ADDRESS_FALLBACK")
    assert module == fallback
    assert module.endswith("/i"), "the address is case-insensitive"
