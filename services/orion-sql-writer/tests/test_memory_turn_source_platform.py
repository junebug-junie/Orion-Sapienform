"""sql-writer stamps source_platform on memory.turn.persisted.v1.

The consolidation service's ai-town gate reads this field and nothing else. If
sql-writer emits it as None for a real ai-town turn, every downstream unit test
still passes (they build turn dicts by hand) and the gate is silently inert --
so this exercises the real emit path off a real chat.history envelope shape.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_source_platform_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

# The exact client_meta shape orion-embodiment writes for an ai-town room, as
# read back off live chat_history_log rows on 2026-08-14.
AITOWN_CLIENT_META = {
    "external_room": {"platform": "aitown", "room_id": "town-square"},
    "external_participant": {"participant_name": "Lucky"},
}


# --------------------------------------------------------------------------
# _chat_source_platform parsing
# --------------------------------------------------------------------------


def test_platform_from_dict():
    assert worker._chat_source_platform(AITOWN_CLIENT_META) == "aitown"


def test_platform_from_json_string():
    """asyncpg / raw bus payloads can hand back jsonb as an encoded string."""
    assert worker._chat_source_platform(json.dumps(AITOWN_CLIENT_META)) == "aitown"


@pytest.mark.parametrize(
    "client_meta",
    [
        None,
        {},
        {"external_room": None},
        {"external_room": {}},
        {"external_room": {"platform": ""}},
        {"external_room": "not-a-dict"},
        "not json at all",
        123,
    ],
)
def test_non_external_shapes_read_as_none(client_meta):
    assert worker._chat_source_platform(client_meta) is None


# --------------------------------------------------------------------------
# the real emit path
# --------------------------------------------------------------------------


def _source() -> ServiceRef:
    return ServiceRef(name="test-hub", version="0.0.1", node="local")


async def _emit_and_capture(monkeypatch, *, client_meta):
    published: list[tuple[str, BaseEnvelope]] = []

    def _fake_write_row(sql_model_cls, data: dict) -> bool:
        return True

    async def _capture_publish(channel: str, env: BaseEnvelope) -> None:
        published.append((channel, env))

    bus = AsyncMock()
    bus.publish = _capture_publish
    monkeypatch.setattr(worker, "_write_row", _fake_write_row)
    monkeypatch.setattr(worker.settings, "sql_writer_emit_memory_turn_persisted", True)
    monkeypatch.setattr(
        worker.settings, "channel_memory_turn_persisted", "orion:memory:turn:persisted"
    )

    payload = {
        "prompt": "You're not the flame, Orion. You're the breath that keeps it alive.",
        "response": "I hear you.",
        "session_id": "sess-1",
        "spark_meta": {},
    }
    if client_meta is not None:
        payload["client_meta"] = client_meta

    await worker.handle_envelope(
        BaseEnvelope(
            kind="chat.history",
            correlation_id=str(uuid4()),
            source=_source(),
            payload=payload,
        ),
        bus=bus,
    )
    assert len(published) == 1
    return published[0][1]


@pytest.mark.asyncio
async def test_aitown_turn_emits_platform(monkeypatch):
    out_env = await _emit_and_capture(monkeypatch, client_meta=AITOWN_CLIENT_META)
    assert out_env.payload["source_platform"] == "aitown"


@pytest.mark.asyncio
async def test_direct_turn_emits_null_platform(monkeypatch):
    out_env = await _emit_and_capture(monkeypatch, client_meta=None)
    assert out_env.payload["source_platform"] is None
