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
from orion.importers.chatgpt_export import build_example_envelope, iter_messages, pair_turns
from app.models.chat_gpt_conversation import ChatGptConversationSQL

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_chatgpt_layers_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def _source() -> ServiceRef:
    return ServiceRef(name="chatgpt-import", version="0.1.0", node="test")


def test_route_map_covers_layered_chatgpt_kinds() -> None:
    route_map = worker.settings.route_map
    assert route_map["chat.gpt.import.run.v1"] == "ChatGptImportRunSQL"
    assert route_map["chat.gpt.conversation.v1"] == "ChatGptConversationSQL"
    assert route_map["chat.gpt.example.v1"] == "ChatGptDerivedExampleSQL"


def test_import_run_payload_routes_to_import_run_table(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_write_row(sql_model_cls, data):
        captured["table"] = sql_model_cls.__tablename__
        captured["data"] = data
        return True

    monkeypatch.setattr(worker, "_write_row", _fake_write_row)
    env = BaseEnvelope(
        kind="chat.gpt.import.run.v1",
        source=_source(),
        payload={
            "import_run_id": "run-1",
            "importer_name": "chatgpt-import",
            "importer_version": "0.1.0",
            "metadata": {"sha": "abc"},
        },
    )

    asyncio.run(worker.handle_envelope(env, bus=None))

    assert captured["table"] == "chat_gpt_import_run"
    assert captured["data"]["import_run_id"] == "run-1"
    assert captured["data"]["metadata"] == {"sha": "abc"}


def test_conversation_and_example_payloads_route(monkeypatch) -> None:
    writes: list[tuple[str, dict]] = []

    def _fake_write_row(sql_model_cls, data):
        writes.append((sql_model_cls.__tablename__, data))
        return True

    monkeypatch.setattr(worker, "_write_row", _fake_write_row)

    convo_env = BaseEnvelope(
        kind="chat.gpt.conversation.v1",
        source=_source(),
        payload={
            "conversation_id": "conv-1",
            "import_run_id": "run-1",
            "session_id": "chatgpt:conv-1",
            "message_count": 3,
            "turn_count": 1,
            "branch_count": 2,
            "metadata": {"current_node_id": "n3"},
        },
    )
    ex_env = BaseEnvelope(
        kind="chat.gpt.example.v1",
        source=_source(),
        payload={
            "example_id": "ex-1",
            "import_run_id": "run-1",
            "conversation_id": "conv-1",
            "prompt": "hello",
            "response": "hi",
            "metadata": {"assistant_node_id": "a1"},
        },
    )

    asyncio.run(worker.handle_envelope(convo_env, bus=None))
    asyncio.run(worker.handle_envelope(ex_env, bus=None))

    assert writes[0][0] == "chat_gpt_conversation"
    assert writes[0][1]["conversation_id"] == "conv-1"
    assert writes[1][0] == "chat_gpt_derived_example"
    assert writes[1][1]["example_id"] == "ex-1"


def test_chatgpt_conversation_model_keys_include_import_run_id() -> None:
    pk_cols = {col.name for col in ChatGptConversationSQL.__table__.primary_key.columns}
    assert pk_cols == {"conversation_id", "import_run_id"}


def test_example_id_is_deterministic_per_import_run() -> None:
    conv = {
        "id": "conv-a",
        "title": "Demo",
        "current_node": "a1",
        "mapping": {
            "u1": {
                "id": "u1",
                "parent": None,
                "children": ["a1"],
                "message": {
                    "id": "m-u1",
                    "author": {"role": "user"},
                    "create_time": 1,
                    "content": {"content_type": "text", "parts": ["hello"]},
                    "metadata": {},
                },
            },
            "a1": {
                "id": "a1",
                "parent": "u1",
                "children": [],
                "message": {
                    "id": "m-a1",
                    "author": {"role": "assistant"},
                    "create_time": 2,
                    "content": {"content_type": "text", "parts": ["hi"]},
                    "metadata": {"model_slug": "gpt"},
                },
            },
        },
    }
    turns = pair_turns(iter_messages(conv, include_branches=False))
    env_a = build_example_envelope(
        import_run_id="run-1",
        conversation_id="conv-a",
        conversation_title="Demo",
        turn=turns[0],
    )
    env_b = build_example_envelope(
        import_run_id="run-1",
        conversation_id="conv-a",
        conversation_title="Demo",
        turn=turns[0],
    )
    env_c = build_example_envelope(
        import_run_id="run-2",
        conversation_id="conv-a",
        conversation_title="Demo",
        turn=turns[0],
    )
    assert env_a.payload["example_id"] == env_b.payload["example_id"]
    assert env_a.payload["example_id"] != env_c.payload["example_id"]


def test_iter_messages_keeps_non_text_block_payload() -> None:
    conv = {
        "id": "conv-media",
        "current_node": "img1",
        "mapping": {
            "img1": {
                "id": "img1",
                "parent": None,
                "children": [],
                "message": {
                    "id": "msg-img1",
                    "author": {"role": "assistant"},
                    "create_time": 12,
                    "content": {
                        "content_type": "multimodal",
                        "parts": [{"type": "image", "asset_id": "asset-1"}],
                    },
                    "metadata": {"provider": "openai"},
                },
            }
        },
    }
    messages = list(iter_messages(conv, include_branches=True))
    assert len(messages) == 1
    assert messages[0].content == "[non_text_content]"
    assert messages[0].content_blocks[0]["asset_id"] == "asset-1"
