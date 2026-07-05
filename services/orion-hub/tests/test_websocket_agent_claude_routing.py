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
