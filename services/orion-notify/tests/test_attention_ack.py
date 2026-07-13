import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import main
from orion.schemas.notify import ChatAttentionAck, ChatAttentionState

ATTN_ID = "8c0c1a20-9a9f-45b1-9aa0-1af6d1f7a6f3"


def _persisted_state(**overrides) -> dict:
    base = dict(
        attention_id=ATTN_ID,
        notification_id=None,
        created_at=datetime(2026, 7, 1, 12, 0, 0).isoformat(),
        source_service="orion-cortex",
        reason="I want to talk",
        severity="info",
        message="Can you check the latest summary?",
        context={},
        require_ack=True,
        acked_at=datetime.now(timezone.utc).isoformat(),
        ack_type="dismissed",
        ack_actor="juniper",
        ack_note="handled",
        status="acked",
    )
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_attention_ack_calls_sql_writer_proxy_not_bus():
    """The fix routes acks through sql-writer's direct write endpoint
    (proxy_post) instead of publishing a NotificationReceiptEvent to the bus
    -- that event was keyed on message_id and never persisted attention ack
    state at all."""
    proxy_post = AsyncMock(return_value=_persisted_state())

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(proxy_post=proxy_post))
    )
    payload = ChatAttentionAck(
        attention_id=ATTN_ID,
        ack_type="dismissed",
        actor="juniper",
        note="handled",
    )

    result = await main.attention_ack(ATTN_ID, payload, request, None)

    assert isinstance(result, ChatAttentionState)
    assert result.status == "acked"
    assert result.ack_type == "dismissed"
    assert result.ack_actor == "juniper"
    assert result.ack_note == "handled"

    proxy_post.assert_awaited_once()
    call_args = proxy_post.await_args.args
    assert call_args[0] == f"/attention/{ATTN_ID}/ack"
    sent_payload = call_args[1]
    assert sent_payload["attention_id"] == ATTN_ID
    assert sent_payload["ack_type"] == "dismissed"
    assert sent_payload["actor"] == "juniper"
    assert sent_payload["note"] == "handled"


@pytest.mark.asyncio
async def test_attention_ack_rejects_mismatched_attention_id():
    proxy_post = AsyncMock()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(proxy_post=proxy_post))
    )
    payload = ChatAttentionAck(attention_id=ATTN_ID, ack_type="seen")

    with pytest.raises(HTTPException) as exc_info:
        await main.attention_ack("00000000-0000-0000-0000-000000000000", payload, request, None)

    assert exc_info.value.status_code == 400
    proxy_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_attention_ack_propagates_not_found_from_sql_writer():
    proxy_post = AsyncMock(
        side_effect=HTTPException(status_code=404, detail="Resource not found via sql-writer")
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(proxy_post=proxy_post))
    )
    payload = ChatAttentionAck(attention_id=ATTN_ID, ack_type="seen")

    with pytest.raises(HTTPException) as exc_info:
        await main.attention_ack(ATTN_ID, payload, request, None)

    assert exc_info.value.status_code == 404
