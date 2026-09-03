"""Boot + out-of-band SQL for orion_biometrics_induction(node, timestamp DESC).

Backs latest_biometrics_induction_by_node's LATERAL top-1-per-node read
(`... CROSS JOIN LATERAL (SELECT ... ORDER BY b.timestamp DESC LIMIT 1)`).

Explicitly NOT the `DISTINCT ON (node)` this replaced: that shape cannot use
this index at all, because DISTINCT ON must consume its whole sorted input and
Postgres has no loose index scan. Measured live 2026-09-03: DISTINCT ON stayed
at 911ms and ~150MB of temp spill WITH the index present; the LATERAL form is
0.47ms and 8 buffer hits.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_PY = Path(__file__).resolve().parents[1] / "app" / "main.py"
SQL_SCRIPT = REPO_ROOT / "scripts" / "sql" / "2026-09-03_biometrics_induction_node_ts_idx.sql"

INDEX_NAME = "orion_biometrics_induction_node_ts_idx"
# DESC is load-bearing, not cosmetic: it is the direction the per-node
# LIMIT 1 scans, and an ASC index yields the OLDEST row per node -- a wrong
# answer that still looks like a working query.
INDEX_DDL = "ON orion_biometrics_induction (node, timestamp DESC)"


def test_writer_boot_creates_biometrics_induction_node_ts_index() -> None:
    src = MAIN_PY.read_text()
    assert INDEX_NAME in src
    assert "CREATE INDEX IF NOT EXISTS" in src
    assert INDEX_DDL in src


def test_out_of_band_sql_script_creates_same_index() -> None:
    assert SQL_SCRIPT.is_file(), f"missing out-of-band script: {SQL_SCRIPT}"
    src = SQL_SCRIPT.read_text()
    assert INDEX_NAME in src
    assert "CREATE INDEX IF NOT EXISTS" in src
    assert INDEX_DDL in src


def test_boot_ddl_stays_non_concurrent() -> None:
    """Boot DDL runs inside `engine.begin()`; CONCURRENTLY cannot.

    Pinned because the out-of-band script documents the CONCURRENTLY form for
    live application, and copying it into main.py would fail at boot with
    "CREATE INDEX CONCURRENTLY cannot run inside a transaction block" -- which
    this service's lifespan swallows as a warning, leaving the index silently
    absent and the pathological scan silently back.
    """
    src = MAIN_PY.read_text()
    # Case-insensitive: a lowercase copy-paste of the out-of-band form is
    # exactly as fatal and exactly as easy to make.
    assert "create index concurrently" not in src.lower()


def test_boot_verifies_the_index_actually_exists() -> None:
    """Creating it is not the same as having it.

    Losing this index raises nothing and times out nowhere -- measured live
    2026-09-03, the read still completes in 163ms (1 node) / 422ms (3 nodes)
    with every index path disabled, well inside the Hub's 2000ms
    statement_timeout. It just silently sequential-scans a 247MB table six
    times a minute. So boot must check pg_indexes and say so, rather than
    assume the CREATE landed.
    """
    src = MAIN_PY.read_text()
    assert "pg_indexes" in src
    assert "MISSING after CREATE INDEX" in src


def test_index_ddl_is_outside_the_swallowing_bootstrap_transaction() -> None:
    """The ~700-statement `engine.begin()` block has one swallowing handler.

    Anything inside it is rolled back by an unrelated earlier failure and
    reported only as a warning. This index must not be in there -- assert it
    appears after that block's handler, following the same precedent the
    chat_response_feedback ALTER already set.
    """
    src = MAIN_PY.read_text()
    handler = src.index('logger.warning("chat_message migration warning')
    assert src.index(INDEX_NAME + " ") > handler or src.index(
        "CREATE INDEX IF NOT EXISTS " + INDEX_NAME
    ) > handler
