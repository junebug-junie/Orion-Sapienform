"""Cards recall rail must only admit active operator-curated rows."""

from __future__ import annotations

from pathlib import Path


def test_cards_adapter_sql_filters_active_status_only() -> None:
    repo = Path(__file__).resolve().parents[3]
    text = (repo / "services" / "orion-recall" / "app" / "cards_adapter.py").read_text(encoding="utf-8")
    assert "WHERE status = 'active'" in text
    assert "pending_review" not in text.split("fetch_card_fragments", 1)[1].split("async def", 1)[0]
