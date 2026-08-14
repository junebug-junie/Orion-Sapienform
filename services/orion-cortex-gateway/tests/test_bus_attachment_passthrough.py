"""Every Hub-facing entry point must carry attachments into the context.

This exists because of a live bug on 2026-08-14: the HTTP handler in
``app/main.py`` was wired for attachments but ``app/bus_client.py`` -- the path
the Hub actually uses -- was not, so images were dropped one line after arriving.

The failure shape is why this is a test rather than a comment. Because the refs
never reached the LLM gateway, its "refuse loudly" guard never fired, the turn
completed normally, and Orion answered "I don't see an image attached in this
chat". A silent drop that looks like a successful turn is the single failure mode
this feature was built to prevent.

The test is source-level on purpose. Both handlers construct
``CortexClientContext`` inline inside long bus/HTTP plumbing that needs a live
Redis and a running orchestrator to exercise, so asserting on the AST is the
cheap check that actually runs in CI -- and it catches the real defect, which was
a missing keyword argument, not a logic error.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"

# (module, enclosing-callable hint) for every Hub-facing entry point.
ENTRY_POINTS = [
    ("bus_client.py", "the bus path the Hub actually uses"),
    ("main.py", "the HTTP /v1/cortex/chat path"),
]


def _context_calls(source: str) -> list[ast.Call]:
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CortexClientContext"
    ]


@pytest.mark.parametrize("filename,description", ENTRY_POINTS)
def test_entry_point_passes_attachments_into_context(filename, description):
    path = APP / filename
    assert path.exists(), f"{filename} missing"

    calls = _context_calls(path.read_text(encoding="utf-8"))
    assert calls, f"no CortexClientContext(...) found in {filename}"

    for call in calls:
        kwargs = {kw.arg for kw in call.keywords if kw.arg}
        assert "attachments" in kwargs, (
            f"{filename}:{call.lineno} builds CortexClientContext without "
            f"`attachments` ({description}). Images sent through this path are "
            f"silently dropped: the LLM gateway never sees them, so its refusal "
            f"guard never fires and the turn returns a normal-looking answer."
        )


def test_both_entry_points_are_actually_covered():
    """Guard the guard: if a handler is renamed or moved, fail loudly."""
    for filename, _ in ENTRY_POINTS:
        assert (APP / filename).exists(), (
            f"{filename} no longer exists -- update ENTRY_POINTS so this check "
            f"keeps covering every Hub-facing context builder."
        )
