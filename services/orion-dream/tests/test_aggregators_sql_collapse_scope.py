"""Regression test: fetch_recent_sql_fragments must only read the strict lane
(Juniper's manually-authored entries) from collapse_mirror. Orion's
machine-generated metacog entries now live in orion_metacog, not
collapse_mirror -- collapse_mirror is strict-lane-only going forward."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_collapse_mirror_query_is_scoped_to_juniper_observer():
    from app import aggregators_sql

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = []
    mock_conn.cursor.return_value = mock_cur

    with patch.object(aggregators_sql.psycopg2, "connect", return_value=mock_conn):
        fragments = aggregators_sql.fetch_recent_sql_fragments(hours=24, chat_sample_n=10)

        assert fragments == []
        assert mock_cur.execute.call_count >= 1
        first_call_sql = mock_cur.execute.call_args_list[0].args[0]
        assert "FROM collapse_mirror" in first_call_sql
        assert "lower(trim(observer)) = 'juniper'" in first_call_sql
