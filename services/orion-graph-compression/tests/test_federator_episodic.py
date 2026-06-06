from unittest.mock import patch, MagicMock
import pytest


def test_episodic_federator_builds_sparql_for_all_graphs():
    """SPARQL query must reference all 9 episodic named graphs."""
    with patch("httpx.Client") as mock_client_cls:
        from app.federators.episodic import EpisodicFederator, EPISODIC_GRAPHS

        f = EpisodicFederator(
            query_url="http://fuseki/query",
            user="admin",
            password="orion",
            timeout_sec=5.0,
        )
        query = f._build_sparql()
        for graph_uri in EPISODIC_GRAPHS:
            assert graph_uri in query, f"Missing graph: {graph_uri}"
        assert "SELECT" in query


def test_episodic_federator_returns_empty_on_http_error():
    """Fuseki failure → empty list, no exception."""
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("connection refused")

        from app.federators.episodic import EpisodicFederator
        f = EpisodicFederator(
            query_url="http://fuseki/query",
            user="admin",
            password="orion",
            timeout_sec=5.0,
        )
        triples = f.fetch(max_nodes=100)
        assert triples == []


def test_episodic_federator_parses_sparql_json():
    """Parse SPARQL JSON bindings into (subject, predicate, object) tuples."""
    sparql_response = {
        "results": {
            "bindings": [
                {
                    "s": {"type": "uri", "value": "http://example.org/A"},
                    "p": {"type": "uri", "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"},
                    "o": {"type": "uri", "value": "http://example.org/ChatTurn"},
                }
            ]
        }
    }
    with patch("httpx.Client") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = sparql_response
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        from app.federators.episodic import EpisodicFederator
        f = EpisodicFederator(
            query_url="http://fuseki/query",
            user="admin",
            password="orion",
            timeout_sec=5.0,
        )
        triples = f.fetch(max_nodes=100)
        assert len(triples) == 1
        assert triples[0] == (
            "http://example.org/A",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://example.org/ChatTurn",
        )
