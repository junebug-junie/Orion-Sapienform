"""scripts/personality_rev3_retire_prose_nodes.py: dry run by default, snapshot first, only rev 2 nodes.

Pure parts always run. The end-to-end part needs a disposable database (ORION_ADMISSION_TEST_DSN);
it creates its own schema and never touches POSTGRES_URI.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("retire", REPO / "scripts/personality_rev3_retire_prose_nodes.py")
retire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retire)

T0 = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)


def row(did, origin="recurrence", status="active", **kw):
    return {"disposition_id": did, "origin": origin, "status": status, "weight": kw.get("weight", 1.0),
            "evidence_count": 3, "chosen_count": 0, "lane_counts": {"reverie": 3}, "first_seen_at": T0,
            "last_reinforced_at": T0, "retired_at": None, "name": kw.get("name"), "orion_verdict": None,
            "label": kw.get("label", f"label {did}")}


def test_target_sql_never_selects_choice_or_retired_nodes():
    assert "status <> 'retired'" in retire.TARGET_SQL and "origin <> 'choice'" in retire.TARGET_SQL
    assert "origin <> 'choice'" in retire.RETIRE_SQL and "status <> 'retired'" in retire.RETIRE_SQL
    assert "ON CONFLICT (disposition_id, at) DO NOTHING" in retire.HISTORY_SQL


def test_plan_anomalies_and_report(tmp_path):
    rows = [row("a"), row("b", origin="seed_self_inquiry", status="candidate", name="Keeping my eyes honest"),
            row("c", origin="mystery")]
    assert [p[2:4] for p in retire.plan(rows)] == [("active", "retired"), ("candidate", "retired"), ("active", "retired")]
    notes = retire.anomalies(rows)
    assert "unexpected_origins:mystery" in notes and "named_nodes:1" in notes
    text = retire.report(tmp_path, applied=False, rows=rows, retired=0, errors=0, notes=notes, files=[tmp_path / "x"])
    assert "DRY RUN (nothing written)" in text and "recurrence/active=1" in text
    assert (tmp_path / "report.md").read_text() == text


def test_progress_line_has_every_section_14_field(tmp_path):
    prog = retire.Progress(tmp_path / "progress.log", 4)
    prog.anomalies.append("x")
    line = prog.line("step", 2)
    for field in ("pct=50.0", "eta_s=", "rows=2/4", "rate=", "errors=0", "anomalies=x"):
        assert field in line
    assert (tmp_path / "progress.log").read_text().strip() == line


def test_main_requires_a_dsn(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    with pytest.raises(SystemExit):
        retire.main(["--out", "/tmp/never-written"])


DSN = os.getenv("ORION_ADMISSION_TEST_DSN")


@pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")
def test_dry_run_then_apply_on_postgres(tmp_path):
    import psycopg

    schema = "retire_test_" + uuid4().hex[:10]
    migrations = [(REPO / f"services/orion-sql-db/manual_migration_personality_{n}_v1.sql").read_text()
                  for n in ("graph", "naming", "choice")]
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    dsn = DSN + ("&" if "?" in DSN else "?") + f"options=-csearch_path%3D{schema}"
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            for migration in migrations:
                conn.execute(re.sub(r"^\s*(BEGIN|COMMIT);\s*$", "", migration, flags=re.M))
            for did, origin, status in (("p1", "recurrence", "active"), ("p2", "seed_self_inquiry", "candidate"),
                                        ("p3", "recurrence", "retired"), ("c1", "choice", "active")):
                conn.execute("INSERT INTO personality_disposition (disposition_id, origin, status, label, centroid, "
                             "weight, evidence_count, first_seen_at, last_reinforced_at, updated_at) VALUES "
                             "(%s, %s, %s, 'l', ARRAY[1]::real[], 2.0, 4, now(), now(), now())", (did, origin, status))
        out = tmp_path / "dry"
        assert retire.main(["--dsn", dsn, "--out", str(out)]) == 0
        with (out / "before.csv").open() as fh:
            assert sorted(r["disposition_id"] for r in csv.DictReader(fh)) == ["p1", "p2"]
        assert "DRY RUN" in (out / "report.md").read_text() and (out / "before_after.csv").exists()
        with psycopg.connect(dsn) as conn:
            assert conn.execute("SELECT count(*) FROM personality_disposition WHERE status = 'retired'").fetchone()[0] == 1

        out = tmp_path / "apply"
        assert retire.main(["--dsn", dsn, "--out", str(out), "--apply"]) == 0
        with psycopg.connect(dsn) as conn:
            st = dict(conn.execute("SELECT disposition_id, status FROM personality_disposition").fetchall())
            hist = conn.execute("SELECT disposition_id FROM personality_weight_history ORDER BY 1").fetchall()
        assert st == {"p1": "retired", "p2": "retired", "p3": "retired", "c1": "active"}
        assert [h[0] for h in hist] == ["p1", "p2"]
        assert "verdict: APPLIED" in (out / "report.md").read_text()
        # A second apply finds nothing left to do.
        assert retire.main(["--dsn", dsn, "--out", str(tmp_path / "again"), "--apply"]) == 0
        assert "target rows (non-retired, origin <> 'choice'): 0" in (tmp_path / "again" / "report.md").read_text()
    finally:
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
