"""orion:memory:turn:persisted is published ONCE per turn, from the turn envelope when there is one.

Before (2026-10-02): a Hub turn produced two publishes -- one from the ``chat.history`` turn
envelope (carrying the turn's spark_meta, including conversation_phase) and one from the assistant
``chat.history.message.v1`` (read back from the row, without that spark_meta). Live logs show both
arrival orders. The consumer then judged the turn twice (the self-comparison bug fixed on the
consumer side in #2479) and, whichever copy arrived first won -- sometimes the copy without the
conversation_phase stamp.

Now: the turn envelope claims the correlation id and publishes at once. An assistant message only
schedules a row-based publish after a short delay, which is dropped if the turn envelope claimed
the id meanwhile. A message-only flow (Collapse Mirror reply: a user message, then an assistant
reply, no turn envelope) still publishes exactly once, from the row.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, SQL_WRITER_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_turn_once_tests", WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

CHANNEL = "orion:memory:turn:persisted"


@pytest.fixture()
def harness(monkeypatch):
    published: list[BaseEnvelope] = []
    bus = AsyncMock()

    async def _publish(channel, env):
        if channel == CHANNEL:
            published.append(env)

    bus.publish = _publish
    monkeypatch.setattr(worker, "_write_row", lambda model, data: True)
    monkeypatch.setattr(worker.settings, "sql_writer_emit_memory_turn_persisted", True)
    monkeypatch.setattr(worker.settings, "channel_memory_turn_persisted", CHANNEL)
    if hasattr(worker.settings, "sql_writer_memory_turn_row_emit_delay_sec"):
        monkeypatch.setattr(worker.settings, "sql_writer_memory_turn_row_emit_delay_sec", 0.05)
    monkeypatch.setattr(worker, "_MEMORY_TURN_EMIT_CLAIMS", {}, raising=False)

    def _row(corr):
        return {"correlation_id": corr, "prompt": "row prompt", "response": "row reply", "spark_meta": {},
                "session_id": "s", "source_platform": None}

    monkeypatch.setattr(worker, "_fetch_chat_turn_for_memory_emit", _row)
    return bus, published


def _src():
    return ServiceRef(name="test-hub", version="0", node="local")


def _turn(corr):
    return BaseEnvelope(kind="chat.history", correlation_id=corr, source=_src(), payload={
        "prompt": "hello", "response": "hi there", "session_id": "s",
        "spark_meta": {"conversation_phase": {"phase_change": "short_pause"}}})


def _message(corr, role):
    return BaseEnvelope(kind="chat.history.message.v1", correlation_id=corr, source=_src(), payload={
        "message_id": f"{corr}:{role}", "session_id": "s", "role": role, "speaker": role,
        "content": "x", "correlation_id": corr})


async def _settle():
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_turn_then_message_publishes_once_from_the_turn(harness):
    bus, published = harness
    corr = str(uuid4())
    await worker.handle_envelope(_turn(corr), bus=bus)
    await worker.handle_envelope(_message(corr, "assistant"), bus=bus)
    await _settle()
    assert len(published) == 1
    assert published[0].payload["spark_meta"]["conversation_phase"]["phase_change"] == "short_pause"


@pytest.mark.asyncio
async def test_message_then_turn_publishes_once_from_the_turn(harness):
    bus, published = harness
    corr = str(uuid4())
    await worker.handle_envelope(_message(corr, "user"), bus=bus)
    await worker.handle_envelope(_message(corr, "assistant"), bus=bus)
    await worker.handle_envelope(_turn(corr), bus=bus)
    await _settle()
    assert len(published) == 1
    assert published[0].payload["prompt"] == "hello"          # the turn envelope's copy, with its spark_meta
    assert "conversation_phase" in published[0].payload["spark_meta"]


@pytest.mark.asyncio
async def test_message_only_flow_still_publishes_once_from_the_row(harness):
    """Collapse Mirror reply: no turn envelope ever arrives."""
    bus, published = harness
    corr = str(uuid4())
    await worker.handle_envelope(_message(corr, "user"), bus=bus)
    await worker.handle_envelope(_message(corr, "assistant"), bus=bus)
    await _settle()
    assert len(published) == 1 and published[0].payload["prompt"] == "row prompt"


@pytest.mark.asyncio
async def test_redelivered_turn_envelope_publishes_once(harness):
    bus, published = harness
    corr = str(uuid4())
    await worker.handle_envelope(_turn(corr), bus=bus)
    await worker.handle_envelope(_turn(corr), bus=bus)
    await _settle()
    assert len(published) == 1



def _bare_turn(corr, client_meta):
    return BaseEnvelope(kind="chat.history", correlation_id=corr, source=_src(), payload={
        "prompt": "", "response": "Morning! I kept thinking about the porch camera.", "session_id": "s",
        "client_meta": client_meta})


@pytest.mark.asyncio
async def test_outreach_turn_envelope_publishes_as_orion(harness):
    """2026-10-09: Orion's own outreach (empty prompt, client_meta.unsolicited) now reaches memory."""
    bus, published = harness
    corr = str(uuid4())
    await worker.handle_envelope(_bare_turn(corr, {"unsolicited": True}), bus=bus)
    await _settle()
    assert [e.payload["initiated_by"] for e in published] == ["orion"]


@pytest.mark.asyncio
async def test_other_promptless_turn_envelope_is_not_memory(harness, monkeypatch):
    """Claude speaking in the room (client_meta.room_claude) also has no prompt; not Orion. The
    row read-back fallback finds nothing for such a row (test_fetch_chat_turn_for_memory_emit)."""
    bus, published = harness
    monkeypatch.setattr(worker, "_fetch_chat_turn_for_memory_emit", lambda corr: None)
    await worker.handle_envelope(_bare_turn(str(uuid4()), {"room_claude": True}), bus=bus)
    await _settle()
    assert published == []
