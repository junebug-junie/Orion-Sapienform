"""WebSocket agent-claude mode must branch to FCC bridge, not context-exec."""
from __future__ import annotations

from pathlib import Path


HUB_ROOT = Path(__file__).resolve().parents[1]
WS_PATH = HUB_ROOT / "scripts" / "websocket_handler.py"


def test_websocket_handler_imports_agent_claude_bridge() -> None:
    source = WS_PATH.read_text(encoding="utf-8")
    assert "run_turn_from_settings" in source
    assert "prepare_agent_claude_input" in source
    assert 'mode == "agent-claude"' in source or "agent-claude" in source


def test_websocket_handler_emits_claude_step_kind() -> None:
    source = WS_PATH.read_text(encoding="utf-8")
    assert '"claude_step"' in source or "'claude_step'" in source


def test_turn_orchestrator_import_is_guarded_and_reports_a_client_facing_error() -> None:
    """Regression guard for the 2026-08-23 production outage: `from
    orion.hub.turn_orchestrator import run_unified_turn` used to be a bare
    module-level import inside the per-message loop -- when it raised
    (a Python-3.11-only `datetime.UTC` import in a transitive dependency,
    on this container's actual Python 3.10 runtime), the exception
    propagated past this whole handler with no client-facing error frame,
    silently killing the WebSocket connection. Every orion-mode chat turn
    was down; from the browser it looked exactly like a hang, not a crash.

    No live-execution test exists for this file (it has no FastAPI
    TestClient WebSocket harness -- see test_websocket_handler_imports_
    agent_claude_bridge above for the same static-source-check convention
    already established here), so this checks the actual guard shape:
    guarded, sent via the disconnect-safe helper, and logged server-side."""
    source = WS_PATH.read_text(encoding="utf-8")
    import_stmt = "from orion.hub.turn_orchestrator import run_unified_turn"
    assert import_stmt in source

    idx = source.index(import_stmt)
    # The import must sit inside a try block, not be a bare statement --
    # look at what immediately precedes it.
    preceding = source[max(0, idx - 200) : idx]
    assert "try:" in preceding

    # And the except branch right after it must (a) log server-side, (b)
    # send a client-facing turn_error frame via the disconnect-safe
    # helper, and (c) not swallow the just-appended-but-never-answered
    # user turn into `history`.
    following = source[idx : idx + 800]
    assert "except ImportError" in following
    assert "logger.error" in following
    assert '"phase": "import"' in following
    assert "_safe_ws_send_json" in following
    assert "history.pop()" in following
