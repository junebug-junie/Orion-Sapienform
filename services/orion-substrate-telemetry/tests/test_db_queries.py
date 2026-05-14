from __future__ import annotations

import re

from app.db import CREATE_TABLE_SQL, GLOBAL_RETENTION_SQL, LATEST_SQL, PRUNE_CORRELATION_SQL


def test_sql_contains_expected_fragments() -> None:
    assert "substrate_tier_outcomes_events" in CREATE_TABLE_SQL
    assert "correlation_id" in CREATE_TABLE_SQL
    assert "generated_at" in CREATE_TABLE_SQL.lower()
    assert "ORDER BY" in PRUNE_CORRELATION_SQL
    assert re.search(r"OFFSET\s+(\$2::int|100)", PRUNE_CORRELATION_SQL, re.I)
    assert "received_at_utc" in GLOBAL_RETENTION_SQL
    assert "where correlation_id" in LATEST_SQL.lower()
