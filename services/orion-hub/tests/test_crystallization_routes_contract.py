from __future__ import annotations

from pathlib import Path

ROUTES = Path(__file__).resolve().parents[1] / "scripts" / "crystallization_routes.py"


def test_crystallization_api_surface_present() -> None:
    text = ROUTES.read_text(encoding="utf-8")
    for path in (
        "/api/memory/crystallizations/propose",
        "/api/memory/crystallizations/proposals/{crystallization_id}/approve",
        "/api/memory/crystallizations/{crystallization_id}/links",
        "/api/memory/crystallizations/{crystallization_id}/neighborhood",
        "/api/memory/crystallizations/projection/rebuild",
        "/api/memory/graphiti/sync/{crystallization_id}",
        "/api/memory/active-packet",
    ):
        assert path in text, f"missing route {path}"


def test_active_packet_route_filters_by_recall_eligibility() -> None:
    text = ROUTES.read_text(encoding="utf-8")
    start = text.index("async def memory_active_packet")
    block = text[start : start + 2500]
    assert "eligible_for_recall" in block
