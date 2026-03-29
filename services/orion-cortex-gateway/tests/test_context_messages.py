from __future__ import annotations

import sys
from pathlib import Path

from orion.schemas.cortex.gateway import CortexChatRequest


REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.bus_client import _context_messages_from_chat_request  # type: ignore  # noqa: E402


def test_context_messages_prefers_structured_messages() -> None:
    req = CortexChatRequest(
        prompt="well, lanes would you build?",
        messages=[
            {"role": "user", "content": "Let's discuss metacognition lanes."},
            {"role": "assistant", "content": "We can define three lanes."},
            {"role": "user", "content": "well, lanes would you build?"},
        ],
        mode="agent",
    )

    out = _context_messages_from_chat_request(req)
    assert [m.role for m in out] == ["user", "assistant", "user"]
    assert out[0].content == "Let's discuss metacognition lanes."


def test_context_messages_falls_back_to_prompt_when_messages_missing() -> None:
    req = CortexChatRequest(prompt="hello", mode="brain")
    out = _context_messages_from_chat_request(req)
    assert len(out) == 1
    assert out[0].role == "user"
    assert out[0].content == "hello"
