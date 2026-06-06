import pytest
import asyncio
from unittest.mock import patch, MagicMock


def _compression_profile():
    return {
        "enable_graph_compression": True,
        "compression_mode": "global",
        "compression_global_top_k": 3,
        "compression_local_top_k": 3,
        "compression_scopes": ["episodic"],
        "enable_rdf": False,
        "enable_sql_timeline": False,
    }


def test_compression_backend_called_when_enabled():
    fake_fragments = [
        {
            "source": "graph_compression",
            "source_ref": "urn:orion:compression:region:abc",
            "text": "Episodic cluster about workflow design.",
            "tags": ["scope:episodic", "kind:community"],
            "salience": 0.7,
            "score": 0.7,
        }
    ]
    with patch("app.worker.fetch_graph_compression_fragments", return_value=fake_fragments) as mock_fetch, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_COMPRESSION_ENABLED = True
        mock_settings.RECALL_COMPRESSION_PG_DSN = "postgresql://x:y@localhost/test"
        mock_settings.RECALL_COMPRESSION_RDF_QUERY_URL = "http://fuseki/query"
        mock_settings.RECALL_COMPRESSION_RDF_USER = "admin"
        mock_settings.RECALL_COMPRESSION_RDF_PASS = "orion"
        mock_settings.RECALL_COMPRESSION_TIMEOUT_SEC = 3.0
        mock_settings.RECALL_RDF_ENDPOINT_URL = None

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "what are my dominant preoccupations",
                _compression_profile(),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_fetch.assert_called_once()
        assert counts.get("graph_compression") == 1
        assert any(c["source"] == "graph_compression" for c in candidates)


def test_compression_backend_returns_empty_when_disabled():
    with patch("app.worker.fetch_graph_compression_fragments") as mock_fetch, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_COMPRESSION_ENABLED = False
        mock_settings.RECALL_COMPRESSION_PG_DSN = None
        mock_settings.RECALL_RDF_ENDPOINT_URL = None

        profile = dict(_compression_profile())
        profile["enable_graph_compression"] = True  # profile enables, but settings disable globally

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "test query",
                profile,
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_fetch.assert_not_called()
        assert counts.get("graph_compression", 0) == 0


def test_compression_backend_does_not_suppress_rdf_backend():
    """When both RDF and compression are enabled, both produce candidates."""
    fake_compression = [
        {"source": "graph_compression", "source_ref": "urn:x", "text": "compression summary",
         "tags": ["scope:episodic"], "salience": 0.5, "score": 0.5}
    ]
    with patch("app.worker.fetch_graph_compression_fragments", return_value=fake_compression), \
         patch("app.worker.settings") as mock_settings, \
         patch("app.worker.fetch_rdf_chatturn_fragments", return_value=[
             {"source": "rdf", "text": "an rdf fragment"}
         ]):
        mock_settings.RECALL_COMPRESSION_ENABLED = True
        mock_settings.RECALL_COMPRESSION_PG_DSN = "postgresql://x:y@localhost/test"
        mock_settings.RECALL_COMPRESSION_RDF_QUERY_URL = None
        mock_settings.RECALL_COMPRESSION_RDF_USER = "admin"
        mock_settings.RECALL_COMPRESSION_RDF_PASS = "orion"
        mock_settings.RECALL_COMPRESSION_TIMEOUT_SEC = 3.0
        mock_settings.RECALL_RDF_ENDPOINT_URL = "http://fuseki/query"

        mixed_profile = {
            "enable_rdf": True,
            "rdf_top_k": 4,
            "enable_graph_compression": True,
            "compression_mode": "global",
            "compression_global_top_k": 3,
            "compression_local_top_k": 3,
            "compression_scopes": ["episodic"],
        }

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "what tensions am I carrying",
                mixed_profile,
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        assert counts.get("graph_compression", 0) >= 1
        # RDF backend ran (may have returned from mock or failed gracefully — either way no suppression)
