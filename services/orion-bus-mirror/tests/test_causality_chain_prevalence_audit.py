from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from causality_chain_prevalence_audit import audit  # type: ignore


def _make_db(tmp_path, rows: list[dict]) -> str:
    db_path = str(tmp_path / "bus_mirror.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE bus_events (timestamp TEXT, channel TEXT, envelope_json TEXT)"
    )
    for row in rows:
        conn.execute(
            "INSERT INTO bus_events(timestamp, channel, envelope_json) VALUES (?, ?, ?)",
            ("2026-07-24T00:00:00Z", "orion:test", json.dumps(row)),
        )
    conn.commit()
    conn.close()
    return db_path


def test_audit_counts_causality_chain_and_correlation_id_presence(tmp_path) -> None:
    rows = [
        {"kind": "a.v1", "correlation_id": "c1", "causality_chain": [{"kind": "parent"}]},
        {"kind": "b.v1", "correlation_id": "c2", "causality_chain": []},
        {"kind": "b.v1", "correlation_id": "c3"},
    ]
    db_path = _make_db(tmp_path, rows)

    result = audit(db_path, checkpoints=1, rows_per_checkpoint=10)

    assert result["total_sampled"] == 3
    assert result["decode_fail"] == 0
    assert result["has_causality_chain"] == 1
    assert result["has_correlation_id"] == 3
    assert result["kinds_with_chain"] == [("a.v1", 1)]


def test_audit_skips_malformed_json_without_crashing(tmp_path) -> None:
    db_path = _make_db(tmp_path, [])
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO bus_events(timestamp, channel, envelope_json) VALUES (?, ?, ?)",
        ("2026-07-24T00:00:00Z", "orion:test", "{not valid json"),
    )
    conn.commit()
    conn.close()

    result = audit(db_path, checkpoints=1, rows_per_checkpoint=10)

    assert result["total_sampled"] == 1
    assert result["decode_fail"] == 1
    assert result["has_causality_chain"] == 0


def test_audit_samples_across_full_rowid_range_via_checkpoints(tmp_path) -> None:
    # 100 rows, only the first and last have a populated causality_chain --
    # a naive "just take the first N rows" sample would miss the last one.
    rows = [{"kind": "mid.v1", "correlation_id": str(i)} for i in range(100)]
    rows[0] = {"kind": "first.v1", "correlation_id": "0", "causality_chain": [{"kind": "p"}]}
    rows[-1] = {"kind": "last.v1", "correlation_id": "99", "causality_chain": [{"kind": "p"}]}
    db_path = _make_db(tmp_path, rows)

    result = audit(db_path, checkpoints=10, rows_per_checkpoint=1)

    assert result["has_causality_chain"] == 2
    kinds = {k for k, _ in result["kinds_with_chain"]}
    assert kinds == {"first.v1", "last.v1"}
