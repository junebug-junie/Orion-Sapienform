from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.collectors.active_packet import fetch_active_packet_fragments


def _query(**overrides):
    fields = {
        "session_id": "s1",
        "node_id": "n1",
        "retrieval_intent": "semantic",
        "fragment": "what did we decide about the move",
        "task_hints": {"task_mode": "chat_general"},
    }
    fields.update(overrides)
    return type("Q", (), fields)()


def _fake_packet(*, crystallization_refs, card_refs=None, retrieval_trace=None):
    return type(
        "Pkt",
        (),
        {
            "items": [
                {
                    "crystallization_id": cid,
                    "summary": f"belief {cid}",
                    "salience": 0.8,
                    "kind": "semantic",
                    "bucket": "semantic",
                }
                for cid in crystallization_refs
            ],
            "crystallization_refs": list(crystallization_refs),
            "card_refs": list(card_refs or []),
            "retrieval_trace": dict(retrieval_trace or {"rails": ["postgres_active"]}),
        },
    )()


@pytest.mark.asyncio
async def test_logs_retrieval_event_when_refs_present():
    packet = _fake_packet(crystallization_refs=["crys_1", "crys_2"], card_refs=["card_9"])
    insert_mock = AsyncMock(return_value="evt-1")

    with patch(
        "app.collectors.active_packet.list_crystallizations", new=AsyncMock(return_value=[])
    ), patch(
        "app.collectors.active_packet.retrieve_active_packet", new=AsyncMock(return_value=packet)
    ), patch(
        "app.collectors.active_packet.insert_retrieval_event", new=insert_mock
    ):
        pool = object()
        frags = await fetch_active_packet_fragments(
            _query(),
            pool=pool,
            settings=type("S", (), {"RECALL_ACTIVE_PACKET_ENABLED": True})(),
        )

    assert len(frags) == 2
    insert_mock.assert_awaited_once()
    args, kwargs = insert_mock.call_args
    assert args[0] is pool
    assert kwargs["query"] == "what did we decide about the move"
    assert kwargs["task_type"] == "chat_general"
    assert kwargs["project_id"] is None
    assert kwargs["session_id"] == "s1"
    assert kwargs["crystallization_ids"] == ["crys_1", "crys_2"]
    assert kwargs["card_refs"] == ["card_9"]
    assert kwargs["trace"] == {"rails": ["postgres_active"]}


@pytest.mark.asyncio
async def test_does_not_log_when_no_crystallization_refs():
    packet = _fake_packet(crystallization_refs=[])
    insert_mock = AsyncMock(return_value="evt-1")

    with patch(
        "app.collectors.active_packet.list_crystallizations", new=AsyncMock(return_value=[])
    ), patch(
        "app.collectors.active_packet.retrieve_active_packet", new=AsyncMock(return_value=packet)
    ), patch(
        "app.collectors.active_packet.insert_retrieval_event", new=insert_mock
    ):
        await fetch_active_packet_fragments(
            _query(),
            pool=object(),
            settings=type("S", (), {"RECALL_ACTIVE_PACKET_ENABLED": True})(),
        )

    insert_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieval_event_log_failure_does_not_break_fragments():
    packet = _fake_packet(crystallization_refs=["crys_1"])
    insert_mock = AsyncMock(side_effect=RuntimeError("db down"))

    with patch(
        "app.collectors.active_packet.list_crystallizations", new=AsyncMock(return_value=[])
    ), patch(
        "app.collectors.active_packet.retrieve_active_packet", new=AsyncMock(return_value=packet)
    ), patch(
        "app.collectors.active_packet.insert_retrieval_event", new=insert_mock
    ):
        frags = await fetch_active_packet_fragments(
            _query(),
            pool=object(),
            settings=type("S", (), {"RECALL_ACTIVE_PACKET_ENABLED": True})(),
        )

    insert_mock.assert_awaited_once()
    assert len(frags) == 1
    assert frags[0]["source"] == "active_packet"
