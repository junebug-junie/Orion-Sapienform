from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from orion.core.contracts.memory_cards import MemoryCardV1
from orion.memory.crystallization.bus_emit_memory_card import (
    card_qualifies_for_crystallizer,
    emit_memory_card_active_for_crystallizer,
    MEMORY_CARD_ACTIVE_CHANNEL,
    MEMORY_CARD_ACTIVE_KIND,
)


def _card(*, status: str = "active", priority: str = "high_recall") -> MemoryCardV1:
    now = datetime.now(timezone.utc)
    return MemoryCardV1(
        card_id=uuid4(),
        slug="test-card",
        types=["fact"],
        status=status,  # type: ignore[arg-type]
        priority=priority,  # type: ignore[arg-type]
        provenance="operator_highlight",
        title="Wild card",
        summary="Operator approved this in Memory tab.",
        visibility_scope=["project:orion"],
        created_at=now,
        updated_at=now,
    )


def test_card_qualifies_only_active_high_salience() -> None:
    assert card_qualifies_for_crystallizer(_card(status="active", priority="high_recall"))
    assert card_qualifies_for_crystallizer(_card(status="active", priority="always_inject"))
    assert not card_qualifies_for_crystallizer(_card(status="pending_review", priority="high_recall"))
    assert not card_qualifies_for_crystallizer(_card(status="active", priority="episodic_detail"))


@pytest.mark.asyncio
async def test_emit_publishes_qualifying_card() -> None:
    published: list[tuple[str, object]] = []

    class _Bus:
        enabled = True

        async def publish(self, channel: str, env: object) -> None:
            published.append((channel, env))

    card = _card()
    ok = await emit_memory_card_active_for_crystallizer(_Bus(), card, service_name="orion-hub")
    assert ok is True
    assert len(published) == 1
    channel, env = published[0]
    assert channel == MEMORY_CARD_ACTIVE_CHANNEL
    assert env.kind == MEMORY_CARD_ACTIVE_KIND  # type: ignore[attr-defined]
    assert env.payload["card_id"] == str(card.card_id)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_emit_skips_non_qualifying_card() -> None:
    class _Bus:
        enabled = True

        async def publish(self, channel: str, env: object) -> None:
            raise AssertionError("should not publish")

    ok = await emit_memory_card_active_for_crystallizer(_Bus(), _card(priority="archival"))
    assert ok is False
