"""Unit tests for the thin, fail-open vision_events reader.

Mocks sqlalchemy at the engine boundary -- no real DB required. Staleness and
the empty-narrative filter are enforced in the SQL WHERE clause (before LIMIT),
so this asserts the query is built with the right predicate/params rather than
re-filtering already-mocked rows in Python -- mirrors
`test_reverie_thin_import_boundary.py`'s discipline: this module must degrade
to an empty list on any failure, never raise out of a reverie tick.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


def _row(**kwargs):
    base = {
        "narrative": "a person walks into frame near the door",
        "created_at": datetime.now(timezone.utc),
    }
    base.update(kwargs)
    return base


def _mock_engine(rows):
    engine = MagicMock()
    conn = MagicMock()
    result = MagicMock()
    result.mappings.return_value.all.return_value = rows
    conn.execute.return_value = result
    engine.connect.return_value.__enter__.return_value = conn
    return engine, conn


def test_returns_fresh_narrated_rows():
    from app import vision_reader

    vision_reader._engine = None
    engine, _ = _mock_engine([_row()])
    with patch.object(vision_reader, "_get_engine", return_value=engine):
        out = vision_reader.read_recent_vision_events(max_age_sec=180, limit=3, stream_ids=["cam0"])
    assert len(out) == 1
    assert out[0]["narrative"] == "a person walks into frame near the door"
    assert out[0]["age_sec"] is not None and out[0]["age_sec"] < 5
    assert set(out[0].keys()) == {"narrative", "age_sec"}  # privacy contract: narrative-only


def test_query_filters_empty_narrative_and_staleness_before_limit():
    """The empty-narrative and staleness predicates must live in SQL, not
    Python post-processing -- otherwise LIMIT can crowd out real rows just
    past the cursor (the bug this reader's sibling perception_reader.py
    already avoids)."""
    from app import vision_reader

    vision_reader._engine = None
    engine, conn = _mock_engine([])
    with patch.object(vision_reader, "_get_engine", return_value=engine):
        vision_reader.read_recent_vision_events(max_age_sec=180, limit=3, stream_ids=["cam0"])
    query_text = str(conn.execute.call_args[0][0])
    params = conn.execute.call_args[0][1]
    assert "narrative IS NOT NULL" in query_text
    assert "narrative <> ''" in query_text
    assert "created_at >=" in query_text
    assert "LIMIT" in query_text
    # WHERE must appear before LIMIT in the compiled text
    assert query_text.index("narrative IS NOT NULL") < query_text.index("LIMIT")
    assert params["limit"] == 3
    assert params["stream_ids"] == ["cam0"]
    age = (datetime.now(timezone.utc) - params["cutoff"]).total_seconds()
    assert 179 <= age <= 185


def test_statement_timeout_configured_on_engine_construction():
    from app import vision_reader

    vision_reader._engine = None
    vision_reader._engine_url = None
    created = {}

    def _fake_create_engine(url, **kwargs):
        created.update(kwargs)
        return MagicMock()

    with patch("sqlalchemy.create_engine", side_effect=_fake_create_engine):
        vision_reader._get_engine()
    assert created.get("connect_args") == {
        "options": f"-c statement_timeout={vision_reader._QUERY_STATEMENT_TIMEOUT_MS}"
    }
    vision_reader._engine = None
    vision_reader._engine_url = None


def test_fails_open_on_db_error():
    from app import vision_reader

    vision_reader._engine = None
    with patch.object(vision_reader, "_get_engine", side_effect=RuntimeError("db down")):
        out = vision_reader.read_recent_vision_events(max_age_sec=180, limit=3, stream_ids=["cam0"])
    assert out == []


def test_limit_zero_short_circuits_without_querying():
    from app import vision_reader

    vision_reader._engine = None
    get_engine = MagicMock()
    with patch.object(vision_reader, "_get_engine", get_engine):
        out = vision_reader.read_recent_vision_events(max_age_sec=180, limit=0, stream_ids=["cam0"])
    assert out == []
    get_engine.assert_not_called()


def _sqlite_engine_with_rows(rows):
    from sqlalchemy import create_engine, text

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE vision_events (event_id TEXT PRIMARY KEY, narrative TEXT, "
            "stream_id TEXT, created_at TIMESTAMP)"))
        for i, (narrative, stream_id, created_at) in enumerate(rows):
            conn.execute(text(
                "INSERT INTO vision_events VALUES (:i, :n, :s, :c)"),
                {"i": str(i), "n": narrative, "s": stream_id, "c": created_at})
    return engine


def test_walkway_row_newer_than_room_row_is_not_a_room_percept():
    """Real SQL against SQLite: the walkway camera narrates the street and the
    patio into the same table. A newer walkway row must never be read as the
    room; legacy rows with no stream_id still are."""
    from datetime import timedelta

    from app import vision_reader

    now = datetime.now(timezone.utc)
    engine = _sqlite_engine_with_rows([
        ("a mug on the desk", "cam0", now - timedelta(seconds=60)),
        ("two people sit on the patio", "walkway", now - timedelta(seconds=5)),
        ("the usual van arrived as expected", "walkway", now - timedelta(seconds=2)),
        ("legacy room narrative", None, now - timedelta(seconds=90)),
    ])
    with patch.object(vision_reader, "_get_engine", return_value=engine):
        out = vision_reader.read_recent_vision_events(max_age_sec=180, limit=5, stream_ids=["carbon", "cam0"])
    assert [r["narrative"] for r in out] == ["a mug on the desk", "legacy room narrative"]


def test_empty_stream_list_reads_legacy_rows_only():
    from datetime import timedelta

    from app import vision_reader

    now = datetime.now(timezone.utc)
    engine = _sqlite_engine_with_rows([
        ("patio", "walkway", now - timedelta(seconds=5)),
        ("legacy", None, now - timedelta(seconds=9)),
    ])
    with patch.object(vision_reader, "_get_engine", return_value=engine):
        out = vision_reader.read_recent_vision_events(max_age_sec=180, limit=5, stream_ids=[])
    assert [r["narrative"] for r in out] == ["legacy"]
