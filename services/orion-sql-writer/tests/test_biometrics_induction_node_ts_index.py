"""Boot + out-of-band SQL for orion_biometrics_induction(node, timestamp DESC).

Backs latest_biometrics_induction_by_node's `DISTINCT ON (node) ... ORDER BY
node, timestamp DESC`. Without this index that read parallel-seq-scans a 247MB
table and external-merge-sorts it to disk to return one row per node --
measured live 2026-09-03 at 418ms mean over 36,393 calls and 11.5 TB of temp
spill (92% of all temp I/O on the instance) once the Hub's Biometrics card
began polling it every 10s per node.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_PY = Path(__file__).resolve().parents[1] / "app" / "main.py"
SQL_SCRIPT = REPO_ROOT / "scripts" / "sql" / "2026-09-03_biometrics_induction_node_ts_idx.sql"

INDEX_NAME = "orion_biometrics_induction_node_ts_idx"
# DESC is load-bearing, not cosmetic: it is the direction the DISTINCT ON
# reads, and an ASC index yields the OLDEST row per node -- a wrong answer
# that still looks like a working query.
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
    assert "CREATE INDEX CONCURRENTLY" not in src
