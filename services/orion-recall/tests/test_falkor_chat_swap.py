"""RECALL_FALKOR_IN_CHAT swaps the chatturn backend, it doesn't merge with RDF
(unlike RECALL_GRAPHITI_IN_CHAT's additive pattern) -- see settings.py's
RECALL_FALKOR_IN_CHAT comment for why."""

from __future__ import annotations

import asyncio
from unittest.mock import patch


def _profile(**overrides):
    base = {
        "enable_rdf": True,
        "rdf_top_k": 4,
    }
    base.update(overrides)
    return base


def test_falkor_chat_used_instead_of_rdf_chat_when_flag_enabled():
    with patch("app.worker.fetch_falkor_chatturn_fragments", return_value=[
             {"id": "t1", "source": "falkor_chat", "text": "falkor fragment"}
         ]) as mock_falkor, \
         patch("app.worker.fetch_rdf_chatturn_fragments", return_value=[
             {"id": "t2", "source": "rdf_chat", "text": "rdf fragment"}
         ]) as mock_rdf, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = "http://fuseki/query"
        mock_settings.RECALL_FALKOR_IN_CHAT = True
        # bare MagicMock: unset attrs are truthy, which would silently
        # enable the neighborhood swap too and route through the real
        # (unmocked) Falkor adapter instead of the mocked rdf/falkor_chat
        # fetches this test is actually exercising.
        mock_settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT = False

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "hi",
                _profile(),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_falkor.assert_called_once()
        mock_rdf.assert_not_called()
        assert counts.get("falkor_chat") == 1
        assert "rdf_chat" not in counts
        assert [c["source"] for c in candidates if c["source"] in ("falkor_chat", "rdf_chat")] == ["falkor_chat"]


def test_rdf_chat_used_when_falkor_flag_disabled():
    with patch("app.worker.fetch_falkor_chatturn_fragments", return_value=[
             {"id": "t1", "source": "falkor_chat", "text": "falkor fragment"}
         ]) as mock_falkor, \
         patch("app.worker.fetch_rdf_chatturn_fragments", return_value=[
             {"id": "t2", "source": "rdf_chat", "text": "rdf fragment"}
         ]) as mock_rdf, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = "http://fuseki/query"
        mock_settings.RECALL_FALKOR_IN_CHAT = False
        mock_settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT = False

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "hi",
                _profile(),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_falkor.assert_not_called()
        mock_rdf.assert_called_once()
        assert counts.get("rdf_chat") == 1
        assert "falkor_chat" not in counts


def test_falkor_chat_runs_independent_of_rdf_enabled():
    """RECALL_FALKOR_IN_CHAT works even when the profile doesn't opt into
    RDF at all -- Falkor chatturn fetch doesn't need "RDF" enabled as a
    concept (see settings.py's comment)."""
    with patch("app.worker.fetch_falkor_chatturn_fragments", return_value=[
             {"id": "t1", "source": "falkor_chat", "text": "falkor fragment"}
         ]) as mock_falkor, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = None  # RDF fully off
        mock_settings.RECALL_FALKOR_IN_CHAT = True
        # bare MagicMock: unset attrs are truthy, which would silently
        # enable the neighborhood swap too and route through the real
        # (unmocked) Falkor adapter instead of the mocked rdf/falkor_chat
        # fetches this test is actually exercising.
        mock_settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT = False

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "hi",
                _profile(enable_rdf=False, rdf_top_k=0),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_falkor.assert_called_once()
        assert counts.get("falkor_chat") == 1


def test_falkor_chat_respects_pcr_intent_suppression():
    """Regression for the real bug found in review: apply_collector_plan
    sets enable_falkor_chat=False for PCR intents (procedural,
    contradiction) that deliberately exclude chat-turn content. Without
    _query_backends checking profile.get("enable_falkor_chat"), turning on
    RECALL_FALKOR_IN_CHAT would leak that content back in regardless of the
    intent's suppression."""
    with patch("app.worker.fetch_falkor_chatturn_fragments", return_value=[
             {"id": "t1", "source": "falkor_chat", "text": "should not appear"}
         ]) as mock_falkor, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = None
        mock_settings.RECALL_FALKOR_IN_CHAT = True
        # bare MagicMock: unset attrs are truthy, which would silently
        # enable the neighborhood swap too and route through the real
        # (unmocked) Falkor adapter instead of the mocked rdf/falkor_chat
        # fetches this test is actually exercising.
        mock_settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT = False

        from app.pcr_collectors import apply_collector_plan, collectors_for_intent
        from app.worker import _query_backends

        plan = collectors_for_intent("procedural")
        profile = apply_collector_plan(_profile(enable_rdf=False, rdf_top_k=0), plan)

        candidates, counts = asyncio.run(
            _query_backends(
                "hi",
                profile,
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_falkor.assert_not_called()
        assert "falkor_chat" not in counts
        assert candidates == []


def test_falkor_chat_failure_degrades_to_empty_not_raise():
    async def _raise(*args, **kwargs):
        raise RuntimeError("falkor down")

    with patch("app.worker.fetch_falkor_chatturn_fragments", side_effect=_raise), \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = None
        mock_settings.RECALL_FALKOR_IN_CHAT = True
        # bare MagicMock: unset attrs are truthy, which would silently
        # enable the neighborhood swap too and route through the real
        # (unmocked) Falkor adapter instead of the mocked rdf/falkor_chat
        # fetches this test is actually exercising.
        mock_settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT = False

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "hi",
                _profile(enable_rdf=False, rdf_top_k=0),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )
        assert counts.get("falkor_chat") == 0
        assert candidates == []
