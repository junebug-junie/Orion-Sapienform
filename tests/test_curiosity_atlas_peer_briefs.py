"""Curiosity Atlas surfaces PeerBriefs (including refused_budget)."""

from __future__ import annotations

from orion.curiosity.atlas import (
    ATLAS_PEER_BRIEFS_CYPHER,
    AtlasPeerBrief,
    AtlasView,
    read_atlas,
    to_payload,
)
from orion.curiosity.worldview import WorldviewReader, WorldviewUnavailable


def test_to_payload_includes_peer_briefs_with_refused_budget() -> None:
    view = AtlasView(
        peer_briefs=[
            AtlasPeerBrief(
                brief_id="brief-1",
                help_id="help-1",
                run_id="abcd1234abcd",
                peer="cursor_auto",
                status="refused_budget",
                summary="",
                refusal_reason="budget_limited",
            ),
            AtlasPeerBrief(
                brief_id="brief-2",
                help_id="help-2",
                run_id="abcd1234abcd",
                peer="claude_room",
                status="ok",
                summary="See worldview.py RO_QUERY.",
                refusal_reason=None,
            ),
        ]
    )
    payload = to_payload(view)
    assert "peer_briefs" in payload
    assert payload["peer_briefs"][0]["status"] == "refused_budget"
    assert payload["peer_briefs"][1]["summary"].startswith("See worldview")


class _Reader(WorldviewReader):
    def __init__(self, *, answers=None, peer_raises=False) -> None:
        super().__init__(host="x", port=1, graph_name="g", client=object())
        self.answers = answers or {}
        self.peer_raises = peer_raises
        self.queries: list[str] = []

    def query(self, cypher: str):
        self.queries.append(cypher)
        if self.peer_raises and "PeerBrief" in cypher:
            raise WorldviewUnavailable("Unknown label :PeerBrief")
        hits = [rows for needle, rows in self.answers.items() if needle in cypher]
        assert len(hits) <= 1, f"needle collision on: {cypher[:80]}"
        return hits[0] if hits else []


def test_missing_peer_brief_label_does_not_mark_atlas_unavailable() -> None:
    """Best-effort: PeerBrief query failure must not blank the whole atlas."""
    view = read_atlas(_Reader(peer_raises=True))
    assert not view.is_unavailable
    assert view.peer_briefs == []
    payload = to_payload(view)
    assert payload["available"] is True
    assert payload["peer_briefs"] == []


def test_read_atlas_loads_peer_briefs() -> None:
    reader = _Reader(
        answers={
            "PeerBrief": [
                {
                    "brief_id": "b1",
                    "help_id": "h1",
                    "run_id": "abcd1234abcd",
                    "peer": "cursor_auto",
                    "status": "refused_budget",
                    "summary": "",
                    "refusal_reason": "budget_limited",
                }
            ]
        }
    )
    view = read_atlas(reader)
    assert any("PeerBrief" in q for q in reader.queries)
    assert ATLAS_PEER_BRIEFS_CYPHER in reader.queries
    assert len(view.peer_briefs) == 1
    assert view.peer_briefs[0].status == "refused_budget"
    assert view.peer_briefs[0].refusal_reason == "budget_limited"
