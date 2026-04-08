from __future__ import annotations

import asyncio
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError


REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.journaler import JournalEntryWriteV1

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_journal_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def _base_source() -> ServiceRef:
    return ServiceRef(name="test-producer", version="0.0.1", node="local")


def _journal_payload() -> dict[str, str]:
    return {
        "entry_id": str(uuid4()),
        "author": "orion",
        "mode": "manual",
        "body": "Persist this journal entry.",
    }


def test_journal_entry_write_uses_nested_payload_for_validation_and_write(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_write_row(sql_model_cls, data):
        captured["table"] = sql_model_cls.__tablename__
        captured["data"] = data
        return True

    monkeypatch.setattr(worker, "_write_row", _fake_write_row)

    journal_payload = _journal_payload()
    wrapped_payload = {
        "schema_id": "orion.envelope",
        "schema_version": "2.0.0",
        "id": str(uuid4()),
        "kind": "journal.entry.write.v1",
        "causality_chain": [],
        "source": {"name": "orion-actions", "version": "1.0.0", "node": "athena"},
        "ttl_ms": None,
        "trace": {},
        "reply_to": None,
        "payload": deepcopy(journal_payload),
    }
    env = BaseEnvelope(kind="journal.entry.write.v1", source=_base_source(), payload=wrapped_payload)

    asyncio.run(worker.handle_envelope(env, bus=None))

    assert captured["table"] == "journal_entries"
    written = captured["data"]
    assert written["author"] == journal_payload["author"]
    assert written["mode"] == journal_payload["mode"]
    assert written["body"] == journal_payload["body"]
    assert "schema_id" not in written
    assert "payload" not in written


def test_regression_full_envelope_shape_fails_journal_schema_but_nested_payload_passes() -> None:
    journal_payload = _journal_payload()
    wrapped_payload = {
        "schema_id": "orion.envelope",
        "schema_version": "2.0.0",
        "id": str(uuid4()),
        "kind": "journal.entry.write.v1",
        "causality_chain": [],
        "source": {"name": "orion-actions", "version": "1.0.0", "node": "athena"},
        "ttl_ms": None,
        "trace": {},
        "reply_to": None,
        "payload": deepcopy(journal_payload),
    }

    with pytest.raises(ValidationError):
        JournalEntryWriteV1.model_validate(wrapped_payload)

    extracted, boundary = worker._payload_for_schema_validation(
        wrapped_payload, JournalEntryWriteV1, kind="journal.entry.write.v1"
    )
    parsed = JournalEntryWriteV1.model_validate(extracted)
    assert boundary == "envelope_payload"
    assert parsed.author == journal_payload["author"]


def test_non_regression_other_route_keeps_payload_shape(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_write_row(sql_model_cls, data):
        captured["table"] = sql_model_cls.__tablename__
        captured["data"] = data
        return True

    monkeypatch.setattr(worker, "_write_row", _fake_write_row)

    payload = {
        "message_id": "m-1",
        "session_id": "s-1",
        "role": "assistant",
        "content": "hello",
    }
    env = BaseEnvelope(kind="chat.gpt.message.v1", source=_base_source(), payload=payload)

    asyncio.run(worker.handle_envelope(env, bus=None))

    assert captured["table"] == "chat_gpt_message"
    written = captured["data"]
    assert written["message_id"] == payload["message_id"]
    assert written["content"] == payload["content"]
