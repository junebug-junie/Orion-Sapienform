import os
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone


def _make_store():
    """Create a CompressionStore with a mock engine."""
    with patch("app.store.create_engine") as mock_eng:
        mock_eng.return_value = MagicMock()
        from app.store import CompressionStore
        store = CompressionStore("postgresql://x:y@localhost/test")
        store._engine = MagicMock()
        return store


def test_enqueue_stale_inserts_row():
    from app.store import CompressionStore
    store = _make_store()
    conn = MagicMock()
    store._engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    store._engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    store.enqueue_stale(scope="episodic", reason="rdf_write")
    conn.execute.assert_called_once()
    sql = str(conn.execute.call_args[0][0])
    assert "stale_queue" in sql


def test_drain_stale_queue_returns_up_to_batch():
    from app.store import CompressionStore
    store = _make_store()
    conn = MagicMock()
    store._engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    store._engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    fake_rows = [
        {"id": 1, "region_id": None, "scope": "episodic", "reason": "rdf_write", "priority": 0},
        {"id": 2, "region_id": None, "scope": "substrate", "reason": "rdf_write", "priority": 0},
    ]
    conn.execute.return_value.mappings.return_value.fetchall.return_value = fake_rows

    items = store.drain_stale_queue(batch_size=5)
    assert len(items) == 2
    assert items[0]["scope"] == "episodic"


def test_upsert_artifact_idempotent():
    from app.store import CompressionStore
    from datetime import datetime, timezone
    store = _make_store()
    conn = MagicMock()
    store._engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    store._engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    store.upsert_artifact(
        region_id="urn:orion:compression:region:abc",
        scope="episodic",
        kind="community",
        summary_kind="structural",
        salience=0.5,
        trust_tier="unverified",
        compression_version="1.0.0",
        generated_at=datetime.now(timezone.utc),
    )
    conn.execute.assert_called_once()
    sql = str(conn.execute.call_args[0][0])
    assert "compression_artifacts" in sql
    assert "ON CONFLICT" in sql
