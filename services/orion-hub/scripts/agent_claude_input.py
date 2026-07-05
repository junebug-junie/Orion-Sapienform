"""Prepare Hub agent-claude turn input. v2 adds slash-command dispatch."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurnRequest:
    prompt: str


def prepare_agent_claude_input(text: str) -> TurnRequest:
    return TurnRequest(prompt=str(text or "").strip())
