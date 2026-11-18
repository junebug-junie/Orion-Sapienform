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


def test_each_run_gets_its_own_directory():
    t = datetime(2026, 11, 13, 12, 0, 1, 5, tzinfo=timezone.utc)
    assert retire.run_dir_for(Path("/x"), t, False).name == "20261113T120001000005Z-dry"
    assert retire.run_dir_for(Path("/x"), t, True).name.endswith("-apply")


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
        def only_run(out):
            (d,) = [p for p in out.iterdir() if p.is_dir()]
            return d

        out = tmp_path / "runs"
        assert retire.main(["--dsn", dsn, "--out", str(out)]) == 0
        dry = only_run(out)
        assert dry.name.endswith("-dry")
        with (dry / "before.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert sorted(r["disposition_id"] for r in rows) == ["p1", "p2"] and rows[0]["updated_at"]
        assert "DRY RUN" in (dry / "report.md").read_text() and (dry / "before_after.csv").exists()
        with psycopg.connect(dsn) as conn:
            assert conn.execute("SELECT count(*) FROM personality_disposition WHERE status = 'retired'").fetchone()[0] == 1

        # Before the rev 4 reducer has ticked, --apply refuses and writes nothing.
        refused = tmp_path / "refused"
        assert retire.main(["--dsn", dsn, "--out", str(refused), "--apply"]) == 3
        with psycopg.connect(dsn) as conn:
            assert conn.execute("SELECT count(*) FROM personality_disposition WHERE status = 'retired'").fetchone()[0] == 1
            conn.execute("INSERT INTO personality_cursor (source, cursor_at) VALUES ('curiosity_sync', now())")
            conn.commit()

        applied = tmp_path / "applied"
        assert retire.main(["--dsn", dsn, "--out", str(applied), "--apply"]) == 0
        with psycopg.connect(dsn) as conn:
            st = dict(conn.execute("SELECT disposition_id, status FROM personality_disposition").fetchall())
            hist = conn.execute("SELECT disposition_id FROM personality_weight_history ORDER BY 1").fetchall()
        assert st == {"p1": "retired", "p2": "retired", "p3": "retired", "c1": "active"}
        assert [h[0] for h in hist] == ["p1", "p2"]
        run = only_run(applied)
        assert "verdict: APPLIED" in (run / "report.md").read_text()
        # A later run never overwrites the snapshot the apply was made from.
        assert retire.main(["--dsn", dsn, "--out", str(applied), "--apply"]) == 0
        assert len([p for p in applied.iterdir() if p.is_dir()]) == 2
        with (run / "before.csv").open() as fh:
            assert len(list(csv.DictReader(fh))) == 2
    finally:
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
