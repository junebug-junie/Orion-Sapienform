from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_chatgpt_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def _source() -> ServiceRef:
    return ServiceRef(name="chatgpt-import", version="0.1.0", node="test")


def test_route_map_covers_chatgpt_message_kind() -> None:
    assert worker.settings.route_map["chat.gpt.message.v1"] == "ChatGptMessageSQL"


def test_chatgpt_message_export_metadata_is_preserved(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeSession:
        def merge(self, obj):
            captured["table"] = obj.__tablename__
            captured["row"] = obj

        def commit(self):
            return None

        def rollback(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(worker, "get_session", lambda: _FakeSession())
    monkeypatch.setattr(worker, "remove_session", lambda: None)

    payload = {
        "message_id": "env-msg-1",
        "session_id": "chatgpt:conv-1",
        "role": "assistant",
        "speaker": "ChatGPT",
        "content": "hello from export",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "content_type": "text",
        "content_blocks": [{"index": 0, "type": "text", "text": "hello from export"}],
        "attachments": [{"name": "file.txt"}],
        "parent_message_id": "parent-node",
        "child_message_ids": ["child-node"],
        "shared_conversation_id": "conv-1",
        "metadata": {"node_id": "assistant-node", "source": "chatgpt_export"},
    }

    assert worker._write_row(worker.ChatGptMessageSQL, payload) is True
    assert captured["table"] == "chat_gpt_message"
    row = captured["row"]
    assert row.id == payload["message_id"]
    assert row.session_id == payload["session_id"]
    assert row.content == payload["content"]
    assert row.meta["parent_message_id"] == payload["parent_message_id"]
    assert row.meta["child_message_ids"] == payload["child_message_ids"]
    assert row.meta["content_blocks"] == payload["content_blocks"]
    assert row.meta["attachments"] == payload["attachments"]
    assert row.meta["shared_conversation_id"] == payload["shared_conversation_id"]
    assert row.meta["message_metadata"] == payload["metadata"]


def test_chatgpt_message_legacy_payload_still_routes(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_write_row(sql_model_cls, data):
        captured["table"] = sql_model_cls.__tablename__
        captured["data"] = data
        return True

    monkeypatch.setattr(worker, "_write_row", _fake_write_row)

    payload = {
        "id": "legacy-msg-1",
        "conversation_id": "chatgpt:legacy-conv",
        "role": "user",
        "content": "legacy payload",
    }
    env = BaseEnvelope(kind="chat.gpt.message.v1", source=_source(), payload=payload)

    asyncio.run(worker.handle_envelope(env, bus=None))

    assert captured["table"] == "chat_gpt_message"
    written = captured["data"]
    assert written["message_id"] == payload["id"]
    assert written["session_id"] == payload["conversation_id"]
    assert written["content"] == payload["content"]
