"""Boot + out-of-band SQL for orion_biometrics_summary(node, timestamp) index.

Hub ambient history queries filter by node and time range; without this index those
scans are sequential on a growing table.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_PY = Path(__file__).resolve().parents[1] / "app" / "main.py"
SQL_SCRIPT = REPO_ROOT / "scripts" / "sql" / "2026-08-26_biometrics_summary_node_ts_idx.sql"

INDEX_NAME = "orion_biometrics_summary_node_ts_idx"


def test_writer_boot_creates_biometrics_summary_node_ts_index() -> None:
    src = MAIN_PY.read_text()
    assert INDEX_NAME in src
    assert "CREATE INDEX IF NOT EXISTS" in src
    assert "ON orion_biometrics_summary (node, timestamp)" in src


def test_out_of_band_sql_script_creates_same_index() -> None:
    assert SQL_SCRIPT.is_file(), f"missing out-of-band script: {SQL_SCRIPT}"
    src = SQL_SCRIPT.read_text()
    assert INDEX_NAME in src
    assert "CREATE INDEX IF NOT EXISTS" in src
    assert "ON orion_biometrics_summary (node, timestamp)" in src
