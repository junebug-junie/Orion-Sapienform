"""Cards recall rail must only admit active operator-curated rows."""

from __future__ import annotations

from pathlib import Path


def test_cards_adapter_sql_filters_active_status_only() -> None:
    repo = Path(__file__).resolve().parents[3]
    text = (repo / "services" / "orion-recall" / "app" / "cards_adapter.py").read_text(encoding="utf-8")
    assert text.count("status = 'active'") >= 2
    fragment = text.split("async def fetch_card_fragments", 1)[1].split("async def fetch_card_fragments_guarded", 1)[0]
    assert "pending_review" not in fragment
    assert "_EXCLUDED_STATUS" not in fragment.split("n_rows = await conn.fetch", 1)[1].split("n_added = 0", 1)[0]
