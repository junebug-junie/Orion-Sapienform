import pytest
from unittest.mock import MagicMock, patch


def _mock_pg_rows(rows):
    """Helper: mock SQLAlchemy execute().mappings().fetchall() returning rows."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.mappings.return_value.fetchall.return_value = rows
    return mock_conn


def test_fetch_returns_empty_when_no_artifacts():
    """Empty Postgres table → empty list, no exception."""
    with patch("app.storage.graph_compression_adapter.create_engine") as mock_eng:
        engine = MagicMock()
        mock_eng.return_value = engine
        conn = _mock_pg_rows([])
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        from app.storage.graph_compression_adapter import fetch_graph_compression_fragments

        frags = fetch_graph_compression_fragments(
            query_text="what are my goals",
            mode="global",
            max_global=5,
            max_local=5,
            scopes=["episodic"],
            pg_dsn="postgresql://x:y@localhost/test",
            rdf_query_url=None,
            rdf_user="admin",
            rdf_pass="orion",
            timeout_sec=3.0,
        )
        assert frags == []


def test_fetch_returns_correct_fragment_shape():
    """Returned fragments have source='graph_compression' and required fields."""
    from datetime import datetime, timezone

    fake_row = {
        "region_id": "urn:orion:compression:region:abc",
        "scope": "episodic",
        "kind": "community",
        "summary_kind": "structural",
        "salience": 0.7,
        "trust_tier": "unverified",
        "compression_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc),
        "stale": False,
    }

    with patch("app.storage.graph_compression_adapter.create_engine") as mock_eng, \
         patch("app.storage.graph_compression_adapter._fetch_summary_from_fuseki") as mock_rdf:
        engine = MagicMock()
        mock_eng.return_value = engine
        conn = _mock_pg_rows([fake_row])
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_rdf.return_value = "A structural summary about episodic memories."

        from app.storage.graph_compression_adapter import fetch_graph_compression_fragments

        frags = fetch_graph_compression_fragments(
            query_text="what are my goals",
            mode="global",
            max_global=5,
            max_local=5,
            scopes=["episodic"],
            pg_dsn="postgresql://x:y@localhost/test",
            rdf_query_url="http://fuseki/query",
            rdf_user="admin",
            rdf_pass="orion",
            timeout_sec=3.0,
        )
        assert len(frags) == 1
        f = frags[0]
        assert f["source"] == "graph_compression"
        assert f["source_ref"] == "urn:orion:compression:region:abc"
        assert "scope:episodic" in f["tags"]
        assert "kind:community" in f["tags"]
        assert f["text"] == "A structural summary about episodic memories."


def test_fetch_does_not_raise_on_fuseki_error():
    """If Fuseki summary fetch fails, the fragment is still returned with a fallback summary."""
    from datetime import datetime, timezone

    fake_row = {
        "region_id": "urn:orion:compression:region:abc",
        "scope": "episodic",
        "kind": "community",
        "summary_kind": "structural",
        "salience": 0.7,
        "trust_tier": "unverified",
        "compression_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc),
        "stale": False,
    }

    with patch("app.storage.graph_compression_adapter.create_engine") as mock_eng, \
         patch("app.storage.graph_compression_adapter._fetch_summary_from_fuseki") as mock_rdf:
        engine = MagicMock()
        mock_eng.return_value = engine
        conn = _mock_pg_rows([fake_row])
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_rdf.side_effect = Exception("fuseki unreachable")

        from app.storage.graph_compression_adapter import fetch_graph_compression_fragments

        frags = fetch_graph_compression_fragments(
            query_text="what are my goals",
            mode="global",
            max_global=5,
            max_local=5,
            scopes=["episodic"],
            pg_dsn="postgresql://x:y@localhost/test",
            rdf_query_url="http://fuseki/query",
            rdf_user="admin",
            rdf_pass="orion",
            timeout_sec=3.0,
        )
        # Fragment still returned, just with fallback text
        assert len(frags) == 1
        assert frags[0]["source"] == "graph_compression"


def test_fetch_filters_stale_artifacts():
    """Stale artifacts are excluded from results."""
    from datetime import datetime, timezone

    stale_row = {
        "region_id": "urn:orion:compression:region:stale",
        "scope": "episodic",
        "kind": "community",
        "summary_kind": "structural",
        "salience": 0.7,
        "trust_tier": "unverified",
        "compression_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc),
        "stale": True,
    }

    with patch("app.storage.graph_compression_adapter.create_engine") as mock_eng:
        engine = MagicMock()
        mock_eng.return_value = engine
        conn = _mock_pg_rows([stale_row])
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        from app.storage.graph_compression_adapter import fetch_graph_compression_fragments

        frags = fetch_graph_compression_fragments(
            query_text="test",
            mode="global",
            max_global=5,
            max_local=5,
            scopes=["episodic"],
            pg_dsn="postgresql://x:y@localhost/test",
            rdf_query_url=None,
            rdf_user="admin",
            rdf_pass="orion",
            timeout_sec=3.0,
        )
        assert frags == []
