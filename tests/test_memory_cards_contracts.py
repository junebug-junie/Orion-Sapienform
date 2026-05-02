from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.core.contracts.memory_cards import (
    MemoryCardV1,
    TimeHorizonV1,
    visibility_allows_card,
)


def test_time_horizon_era_requires_start() -> None:
    with pytest.raises(ValidationError):
        TimeHorizonV1(kind="era_bound", start=None)


def test_anchor_requires_anchor_class() -> None:
    with pytest.raises(ValidationError):
        MemoryCardV1(
            slug="x",
            types=["anchor"],
            provenance="operator_highlight",
            title="t",
            summary="s",
            anchor_class=None,
        )


def test_visibility_social_excludes_intimate() -> None:
    assert visibility_allows_card(["chat"], "chat")
    assert not visibility_allows_card(["intimate"], "social")
    assert visibility_allows_card(["all"], "social")


def test_memory_card_minimal() -> None:
    c = MemoryCardV1(
        slug="test-card",
        types=["fact"],
        provenance="operator_highlight",
        title="Title",
        summary="Summary text",
    )
    assert c.slug == "test-card"
