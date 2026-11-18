"""Per-turn wiring for Orion's tools in a room turn (Patch 1b).

Design: docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md sections 3-4.

For one room turn this module:

1. writes the turn's MCP config (one server, ``orion_room``, started as
   ``<python> -P -m orion.claude_tools.mcp_server``) with the turn's binding in the server's env, file
   mode 0600 under ROOM_COMPANION_TOOLS_STATE_DIR. The binding is server-authored: room, Claude session
   and the invite text come from the request, never from Claude;
2. after the turn, reads the server's call log and renders the footer Juniper sees under Claude's reply
   ("Looked at: body economy (Orion: yes, "...")."), then deletes both files.

Nothing here grants anything. The broker in orion-harness-governor decides every call.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from orion.claude_tools.mcp_server import BINDING_ENV, SERVER_NAME
from orion.claude_tools.render import render_footer
from orion.schemas.claude_tools import TOPIC_MAX, ClaudeToolBindingV1
from orion.schemas.room_claude import RoomClaudeRequestV1

logger = logging.getLogger("orion-room-companion.tools")

MCP_MODULE = "orion.claude_tools.mcp_server"


@dataclass(frozen=True)
class TurnTools:
    mcp_config_path: Path
    call_log_path: Path

    def cleanup(self) -> None:
        for path in (self.mcp_config_path, self.call_log_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("room_tools_cleanup_failed path=%s", path)


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:80] or "turn"


def build_binding(request: RoomClaudeRequestV1, *, claude_session_id: str, call_log_path: Path) -> ClaudeToolBindingV1:
    topic = (request.prompt or "").strip()
    return ClaudeToolBindingV1(
        turn_request_id=request.request_id,
        room_id=request.room_id,
        platform=request.platform,
        claude_session_id=claude_session_id,
        hub_session_id=request.session_id,
        trigger=request.trigger,
        topic=(topic[: TOPIC_MAX - 1] + "…") if len(topic) > TOPIC_MAX else (topic or None),
        topic_from=(request.invited_by or None) and str(request.invited_by)[:80],
        call_log_path=str(call_log_path),
    )


def mcp_config(binding: ClaudeToolBindingV1, *, python: str, pythonpath: str, bus_url: str) -> dict[str, Any]:
    return {
        "mcpServers": {
            SERVER_NAME: {
                "type": "stdio",
                "command": python,
                "args": ["-P", "-m", MCP_MODULE],
                "env": {
                    "ORION_BUS_URL": bus_url,
                    "PYTHONPATH": pythonpath,
                    BINDING_ENV: binding.model_dump_json(),
                },
            }
        }
    }


STALE_SEC = 24 * 3600


def sweep_stale(state: Path, *, now: Optional[float] = None) -> int:
    """Delete turn files a crashed turn left behind (they hold the invite text)."""
    import time

    cutoff = (now if now is not None else time.time()) - STALE_SEC
    removed = 0
    for path in state.glob("*.json*"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def prepare_turn_tools(settings: Any, request: RoomClaudeRequestV1, *, claude_session_id: str) -> TurnTools:
    state = Path(settings.ROOM_COMPANION_TOOLS_STATE_DIR)
    state.mkdir(parents=True, exist_ok=True)
    sweep_stale(state)
    name = _safe_name(request.request_id)
    call_log = state / f"{name}.calls.jsonl"
    config_path = state / f"{name}.mcp.json"
    call_log.unlink(missing_ok=True)
    binding = build_binding(request, claude_session_id=claude_session_id, call_log_path=call_log)
    body = json.dumps(mcp_config(binding, python=settings.ROOM_COMPANION_TOOLS_PYTHON,
                                 pythonpath=settings.ROOM_COMPANION_TOOLS_PYTHONPATH,
                                 bus_url=settings.ORION_BUS_URL))
    fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(body)
    return TurnTools(mcp_config_path=config_path, call_log_path=call_log)


def read_call_log(path: Path) -> List[dict]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    out: List[dict] = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def footer_for(path: Path) -> Optional[str]:
    return render_footer(read_call_log(path))


def with_footer(text: str, footer: Optional[str]) -> str:
    return f"{text}\n\n{footer}" if footer and text else text


__all__ = [
    "MCP_MODULE",
    "TurnTools",
    "build_binding",
    "footer_for",
    "mcp_config",
    "prepare_turn_tools",
    "read_call_log",
    "with_footer",
]
