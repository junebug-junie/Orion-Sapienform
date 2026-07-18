"""RECALL_FALKOR_GRAPHTRI_IN_CHAT swaps the graphtri backend within
_query_backends's rdf_enabled block -- unlike RECALL_FALKOR_IN_CHAT, this
flag stays nested inside rdf_enabled/rdf_top_k rather than running
independent of it, since graphtri's availability is already correctly
governed by pcr_collectors.py's plan.get("rdf") suppression."""

from __future__ import annotations

import asyncio
from unittest.mock import patch


def _graphtri_profile(**overrides):
    base = {
        "profile": "graphtri.v1",
        "enable_rdf": True,
        "rdf_top_k": 8,
        "rdf_graphtri_mode": True,
    }
    base.update(overrides)
    return base


def test_falkor_graphtri_used_instead_of_rdf_graphtri_when_flag_enabled():
    with patch("app.worker.fetch_falkor_graphtri_fragments", return_value=[
             {"id": "t1:circe", "source": "falkor_graphtri", "text": "Claim: mentions circe"}
         ]) as mock_falkor, \
         patch("app.worker.fetch_rdf_graphtri_fragments") as mock_rdf, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = "http://fuseki/query"
        mock_settings.RECALL_FALKOR_IN_CHAT = False
        mock_settings.RECALL_FALKOR_GRAPHTRI_IN_CHAT = True

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "tell me about circe",
                _graphtri_profile(),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_falkor.assert_called_once()
        mock_rdf.assert_not_called()
        assert counts.get("falkor_graphtri") == 1
        assert counts.get("rdf") == 1
        assert any(c["source"] == "falkor_graphtri" for c in candidates)


def test_rdf_graphtri_used_when_falkor_graphtri_flag_disabled():
    with patch("app.worker.fetch_falkor_graphtri_fragments") as mock_falkor, \
         patch("app.worker.fetch_rdf_graphtri_fragments", return_value=[
             {"id": "claim-1", "source": "rdf", "text": "Claim: mentionsEntity circe"}
         ]) as mock_rdf, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = "http://fuseki/query"
        mock_settings.RECALL_FALKOR_IN_CHAT = False
        mock_settings.RECALL_FALKOR_GRAPHTRI_IN_CHAT = False

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "tell me about circe",
                _graphtri_profile(),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_rdf.assert_called_once()
        mock_falkor.assert_not_called()
        assert "falkor_graphtri" not in counts
        assert counts.get("rdf") == 1


def test_falkor_graphtri_stays_off_when_rdf_disabled_by_profile():
    """Unlike falkor_chat_enabled, this flag does NOT run independent of
    rdf_enabled -- it's nested inside that block on purpose, so it
    correctly inherits pcr_collectors.py's plan.get("rdf") suppression
    (rdf_top_k=0) for free."""
    with patch("app.worker.fetch_falkor_graphtri_fragments") as mock_falkor, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = "http://fuseki/query"
        mock_settings.RECALL_FALKOR_IN_CHAT = False
        mock_settings.RECALL_FALKOR_GRAPHTRI_IN_CHAT = True

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "tell me about circe",
                _graphtri_profile(enable_rdf=False, rdf_top_k=0),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_falkor.assert_not_called()
        assert "falkor_graphtri" not in counts


def test_falkor_graphtri_falls_back_to_generic_rdf_fragments_on_empty():
    with patch("app.worker.fetch_falkor_graphtri_fragments", return_value=[]) as mock_falkor, \
         patch("app.worker.fetch_rdf_fragments", return_value=[
             {"id": "legacy-1", "source": "rdf", "text": "legacy fallback fragment"}
         ]) as mock_fallback, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = "http://fuseki/query"
        mock_settings.RECALL_FALKOR_IN_CHAT = False
        mock_settings.RECALL_FALKOR_GRAPHTRI_IN_CHAT = True

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "tell me about circe",
                _graphtri_profile(),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_falkor.assert_called_once()
        mock_fallback.assert_called_once()
        assert counts.get("falkor_graphtri") == 0
        assert counts.get("rdf") == 1


def test_falkor_graphtri_works_when_rdf_endpoint_unset():
    """Regression for a real bug found in code review: the Falkor path must
    not depend on RECALL_RDF_ENDPOINT_URL at all -- that's the whole point
    of this migration's target end-state (Fuseki retired). The original fix
    only added the falkor_graphtri_enabled OR-clause to process_recall's
    graphtri branch, not to _query_backends's rdf_enabled -- meaning this
    exact scenario (RDF endpoint gone, Falkor flag on) silently went dark
    for deep.graph.v1/brain.recall.v1 profiles until fixed."""
    with patch("app.worker.fetch_falkor_graphtri_fragments", return_value=[
             {"id": "t1:circe", "source": "falkor_graphtri", "text": "Claim: mentions circe"}
         ]) as mock_falkor, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = None  # Fuseki fully gone
        mock_settings.RECALL_FALKOR_IN_CHAT = False
        mock_settings.RECALL_FALKOR_GRAPHTRI_IN_CHAT = True

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "tell me about circe",
                _graphtri_profile(),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_falkor.assert_called_once()
        assert counts.get("falkor_graphtri") == 1
        assert any(c["source"] == "falkor_graphtri" for c in candidates)


def test_falkor_graphtri_failure_degrades_to_empty_not_raise():
    async def _raise(*args, **kwargs):
        raise RuntimeError("falkor down")

    with patch("app.worker.fetch_falkor_graphtri_fragments", side_effect=_raise), \
         patch("app.worker.fetch_rdf_fragments", return_value=[]), \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = "http://fuseki/query"
        mock_settings.RECALL_FALKOR_IN_CHAT = False
        mock_settings.RECALL_FALKOR_GRAPHTRI_IN_CHAT = True

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "tell me about circe",
                _graphtri_profile(),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )
        assert counts.get("falkor_graphtri") == 0
