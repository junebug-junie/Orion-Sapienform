"""Anchor rail must pass the profile sql_since_minutes into fetch_exact_fragments."""

from __future__ import annotations

import asyncio

from app import worker


def test_fetch_anchor_candidates_forwards_sql_since_minutes(monkeypatch) -> None:
    captured: dict = {}

    async def _fake_exact(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(worker, "fetch_exact_fragments", _fake_exact)
    monkeypatch.setattr(worker, "_anchor_tokens", lambda _text, **_: ["connection123"])

    asyncio.run(
        worker._fetch_anchor_candidates(
            query_text="anything",
            profile={"profile": "journal.daily.grounded.v1", "sql_since_minutes": 1440, "sql_top_k": 50},
            session_id=None,
            node_id=None,
            diagnostic=False,
            exclusion={},
        )
    )

    assert captured.get("since_minutes") == 1440
