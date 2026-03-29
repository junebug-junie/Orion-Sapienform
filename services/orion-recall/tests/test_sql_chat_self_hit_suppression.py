from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app import sql_chat


class _Conn:
    def __init__(self, rows):
        self._rows = rows

    async def fetch(self, query):
        return self._rows

    async def close(self):
        return None


def test_fetch_chat_history_pairs_suppresses_active_prompt(monkeypatch):
    rows = [
        {"prompt": "Teddy loves Addy", "response": "ok", "created_at": 1},
        {"prompt": "older prompt", "response": "old", "created_at": 0},
    ]

    async def _connect(_dsn):
        return _Conn(rows)

    monkeypatch.setattr(sql_chat, "asyncpg", type("_AsyncPg", (), {"connect": _connect}))

    out = asyncio.run(sql_chat.fetch_chat_history_pairs(
        limit=5,
        since_minutes=60,
        exclude_text="Teddy loves Addy",
    ))

    assert len(out) == 1
    assert "older prompt" in out[0].text


def test_fetch_chat_messages_suppresses_user_echo(monkeypatch):
    rows = [
        {"role": "user", "text": "Teddy loves Addy", "created_at": 1},
        {"role": "assistant", "text": "Teddy loves Addy", "created_at": 1},
    ]

    async def _connect(_dsn):
        return _Conn(rows)

    monkeypatch.setattr(sql_chat, "asyncpg", type("_AsyncPg", (), {"connect": _connect}))
    monkeypatch.setattr(sql_chat.settings, "RECALL_SQL_MESSAGE_TABLE", "chat_messages")

    out = asyncio.run(sql_chat.fetch_chat_messages(
        limit=5,
        since_minutes=60,
        exclude_text="Teddy loves Addy",
    ))

    assert len(out) == 1
    assert out[0].text.startswith("assistant:")
