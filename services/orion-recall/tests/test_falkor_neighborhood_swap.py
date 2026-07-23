"""RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT swaps the generic Fuseki neighborhood
fetch, it doesn't merge with it -- same swap-not-additive convention as
RECALL_FALKOR_IN_CHAT (see test_falkor_chat_swap.py / settings.py's
RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT comment)."""

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


def test_falkor_neighborhood_used_instead_of_rdf_when_flag_enabled():
    with patch("app.worker.fetch_falkor_neighborhood_fragments", return_value=[
             {"id": "t1", "source": "falkor_neighborhood", "text": "falkor fragment"}
         ]) as mock_falkor, \
         patch("app.worker.fetch_rdf_fragments", return_value=[
             {"id": "t2", "source": "rdf", "text": "rdf fragment"}
         ]) as mock_rdf, \
         patch("app.worker.fetch_falkor_chatturn_fragments", return_value=[]), \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = "http://fuseki/query"
        mock_settings.RECALL_FALKOR_IN_CHAT = False
        mock_settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT = True

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
        assert counts.get("falkor_neighborhood") == 1
        assert "rdf" not in counts
        assert [c["source"] for c in candidates if c["source"] in ("falkor_neighborhood", "rdf")] == ["falkor_neighborhood"]


def test_rdf_used_when_falkor_neighborhood_flag_disabled():
    with patch("app.worker.fetch_falkor_neighborhood_fragments", return_value=[
             {"id": "t1", "source": "falkor_neighborhood", "text": "falkor fragment"}
         ]) as mock_falkor, \
         patch("app.worker.fetch_rdf_fragments", return_value=[
             {"id": "t2", "source": "rdf", "text": "rdf fragment"}
         ]) as mock_rdf, \
         patch("app.worker.fetch_falkor_chatturn_fragments", return_value=[]), \
         patch("app.worker.fetch_rdf_chatturn_fragments", return_value=[]), \
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
        assert counts.get("rdf") == 1
        assert "falkor_neighborhood" not in counts


def test_falkor_neighborhood_respects_profile_suppression():
    """Same suppression contract as falkor_chat_enabled: a profile can opt
    out of the neighborhood swap specifically via enable_falkor_neighborhood,
    independent of the global flag."""
    with patch("app.worker.fetch_falkor_neighborhood_fragments", return_value=[
             {"id": "t1", "source": "falkor_neighborhood", "text": "should not appear"}
         ]) as mock_falkor, \
         patch("app.worker.fetch_rdf_fragments", return_value=[
             {"id": "t2", "source": "rdf", "text": "rdf fragment"}
         ]) as mock_rdf, \
         patch("app.worker.fetch_falkor_chatturn_fragments", return_value=[]), \
         patch("app.worker.fetch_rdf_chatturn_fragments", return_value=[]), \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = "http://fuseki/query"
        mock_settings.RECALL_FALKOR_IN_CHAT = False
        mock_settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT = True

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "hi",
                _profile(enable_falkor_neighborhood=False),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_falkor.assert_not_called()
        mock_rdf.assert_called_once()
        assert counts.get("rdf") == 1


def test_falkor_neighborhood_runs_independent_of_rdf_enabled():
    """Regression for a Critical finding caught in review: the first version
    of this swap was nested inside `if rdf_enabled:`, which is gated on
    `bool(settings.RECALL_RDF_ENDPOINT_URL)` -- meaning it would go
    permanently inert the moment Fuseki's endpoint URL is removed from
    config, exactly the end-state this whole migration is working toward.
    Must behave like falkor_chat_enabled: reachable even when RDF is fully
    off, as long as rdf_top_k > 0 (the profile still wants some amount of
    graph-neighborhood candidates)."""
    with patch("app.worker.fetch_falkor_neighborhood_fragments", return_value=[
             {"id": "t1", "source": "falkor_neighborhood", "text": "falkor fragment"}
         ]) as mock_falkor, \
         patch("app.worker.fetch_rdf_fragments") as mock_rdf, \
         patch("app.worker.fetch_falkor_chatturn_fragments", return_value=[]), \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = None  # RDF fully off
        mock_settings.RECALL_FALKOR_IN_CHAT = False
        mock_settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT = True

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "hi",
                _profile(enable_rdf=False, rdf_top_k=4),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_falkor.assert_called_once()
        mock_rdf.assert_not_called()
        assert counts.get("falkor_neighborhood") == 1


def test_falkor_neighborhood_failure_degrades_to_empty_not_raise():
    async def _raise(*args, **kwargs):
        raise RuntimeError("falkor down")

    with patch("app.worker.fetch_falkor_neighborhood_fragments", side_effect=_raise), \
         patch("app.worker.fetch_falkor_chatturn_fragments", return_value=[]), \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = "http://fuseki/query"
        mock_settings.RECALL_FALKOR_IN_CHAT = False
        mock_settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT = True

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
        assert counts.get("falkor_neighborhood") == 0
        assert [c for c in candidates if c.get("source") == "falkor_neighborhood"] == []


def test_falkor_neighborhood_skipped_when_profile_rdf_top_k_is_zero():
    """rdf_top_k > 0 is the per-profile 'wants graph-neighborhood candidates
    at all' signal this swap still respects (unlike falkor_chat, which is a
    standalone concept from rdf) -- a profile with rdf_top_k=0 must not
    start getting candidates just because the global flag is on."""
    with patch("app.worker.fetch_falkor_neighborhood_fragments", return_value=[
             {"id": "t1", "source": "falkor_neighborhood", "text": "should not appear"}
         ]) as mock_falkor, \
         patch("app.worker.fetch_falkor_chatturn_fragments", return_value=[]), \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_RDF_ENDPOINT_URL = None
        mock_settings.RECALL_FALKOR_IN_CHAT = False
        mock_settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT = True

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

        mock_falkor.assert_not_called()
        assert "falkor_neighborhood" not in counts


def test_falkor_neighborhood_backend_weight_matches_rdf():
    """Regression for a High finding caught in review: without an explicit
    fusion.py weight entry, 'falkor_neighborhood' candidates would silently
    score with the generic 0.5 fallback instead of rdf's 0.3 -- a 67%
    relative composite-score bump nobody intended."""
    from app.fusion import DEFAULT_BACKEND_WEIGHTS

    assert DEFAULT_BACKEND_WEIGHTS.get("falkor_neighborhood") == DEFAULT_BACKEND_WEIGHTS.get("rdf")


def test_falkor_neighborhood_belief_source_rank_matches_rdf():
    """Regression for a Medium finding caught in review: _belief_source_rank
    checks src.startswith('rdf') first, so any source not literally prefixed
    'rdf' needs its own _BELIEF_SOURCE_ORDER entry or it silently falls to
    the 99 (last) default in PCR belief-digest ordering."""
    from app.fusion import _belief_source_rank

    assert _belief_source_rank("falkor_neighborhood") == _belief_source_rank("rdf")
    assert _belief_source_rank("falkor_neighborhood") != 99
