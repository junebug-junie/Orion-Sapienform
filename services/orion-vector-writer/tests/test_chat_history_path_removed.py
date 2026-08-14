"""Regression coverage for the 2026-08-14 chat-history vector-write kill.

Before this patch, `app/chat_history.py` normalized `chat.history.message.v1`
envelopes into a `VectorWriteRequest` targeting the `orion_chat` Chroma
collection. That module (and its only caller in `app/main.py`) is deleted --
this test proves `normalize_to_request` no longer has any chat-history-
specific branch and that a chat-history-shaped envelope produces no write
request via the generic fallback path either.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

_HERE = Path(__file__).resolve()
SERVICE_ROOT = _HERE.parents[1]
REPO_ROOT = _HERE.parents[3]

if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

os.environ.setdefault("ORION_BUS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ORION_BUS_ENABLED", "false")

from app import main as vector_writer_main  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.chat_history import (  # noqa: E402
    CHAT_HISTORY_MESSAGE_KIND,
    CHAT_HISTORY_TURN_KIND,
)


def test_app_chat_history_module_is_gone():
    assert not (SERVICE_ROOT / "app" / "chat_history.py").exists()


def test_main_has_no_chat_history_import_or_call():
    """The comment explaining the kill is allowed to name the removed
    symbols; the point is there is no live import or call left."""
    assert not hasattr(vector_writer_main, "chat_history_envelope_to_request")
    assert not hasattr(vector_writer_main, "CHAT_HISTORY_MESSAGE_KIND")
    assert "app.chat_history" not in sys.modules


def test_normalize_to_request_returns_none_for_chat_history_envelope():
    env = BaseEnvelope(
        kind=CHAT_HISTORY_MESSAGE_KIND,
        source=ServiceRef(name="orion-hub", node="athena", version="0.1.0"),
        correlation_id=uuid4(),
        payload={
            "message_id": "msg-1",
            "session_id": "sess-1",
            "role": "user",
            "content": "this used to become an orion_chat vector write",
        },
    )
    assert vector_writer_main.normalize_to_request(env) is None


def test_normalize_to_request_has_no_resurrectable_chat_branch():
    """Guards the landmine code review flagged 2026-08-14: the generic
    fallback used to have `elif kind in ("chat.message", "chat.history"):`
    hard-coding `collection="orion_chat"` -- the exact collection this kill
    removed. `CHAT_HISTORY_TURN_KIND == "chat.history"` (a literal match),
    so re-subscribing orion-vector-writer to orion:chat:history:turn (or any
    future producer emitting these kinds on an already-subscribed channel)
    would otherwise silently resurrect the killed write with zero code
    changes. Both literal kind strings must now fall through to the inert
    "generic text/event" fallback (payload has no `text`/`summary` key here,
    so content is empty and normalize_to_request returns None)."""
    for kind in (CHAT_HISTORY_TURN_KIND, "chat.message", "chat.history"):
        env = BaseEnvelope(
            kind=kind,
            source=ServiceRef(name="orion-hub", node="athena", version="0.1.0"),
            correlation_id=uuid4(),
            payload={
                "role": "user",
                "session_id": "sess-1",
                "content": "would have hard-coded collection=orion_chat before this kill",
            },
        )
        assert vector_writer_main.normalize_to_request(env) is None, kind
