"""scripts/personality_relabel_choice_nodes.py (#3444): dry run by default, section 14 files, apply goes
through orion-durable-runs with the reviewed pairs. No database: the plan is built in memory."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

from orion.personality.relabel import Relabel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "personality_relabel_choice_nodes.py"


def _mod():
    spec = importlib.util.spec_from_file_location("personality_relabel_choice_nodes", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


PLAN = [
    Relabel("d1", "candidate", "A read, \"The Guide, 1 of 1", "The Guide, 1 of 1", 0.5, 0.5, 1, "choice:a"),
    Relabel("d2", "active", "same, 3 of 3", "same, 3 of 3", 0.9, 0.9, 3, "choice:b"),
    Relabel("d3", "candidate", "x over y (+14 more), 1 of 1", "x, 1 of 1", 0.5, 0.5, 1, "choice:c",
            ("choice_rows_missing:1",)),
]


def test_dry_run_writes_snapshot_csv_progress_and_report(tmp_path, monkeypatch):
    m = _mod()

    async def fake_plan(dsn):
        return PLAN

    def no_post(*a, **k):
        raise AssertionError("a dry run never calls durable-runs")

    monkeypatch.setattr(m, "read_plan", fake_plan)
    monkeypatch.setattr(m, "post_apply", no_post)
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path)])
    assert m.main() == 0
    [run] = list(tmp_path.iterdir())
    assert run.name.endswith("-dry")
    snap = [json.loads(line) for line in (run / "snapshot.jsonl").read_text().splitlines()]
    assert [s["disposition_id"] for s in snap] == ["d1", "d2", "d3"]  # every node: the rollback source
    rows = list(csv.DictReader((run / "before_after.csv").open()))
    assert [r["disposition_id"] for r in rows] == ["d1", "d3"]
    log = (run / "progress.log").read_text()
    for field in ("%", "eta=", "rows=", "rate=", "errors=", "anomalies="):
        assert field in log
    rep = (run / "report.md").read_text()
    assert "DRY RUN" in rep and "Labels that change: 2" in rep and "before 2, after 0" in rep
    assert "choice_rows_missing:1" in rep


def test_apply_posts_the_reviewed_pairs_and_saves_the_response(tmp_path, monkeypatch):
    m = _mod()
    sent = {}

    async def fake_plan(dsn):
        return PLAN

    def fake_post(url, changes):
        sent["pairs"] = {r.disposition_id: [r.old_label, r.new_label] for r in changes}
        return {"applied": True, "written": 2, "written_ids": ["d1", "d3"], "refused_ids": []}

    monkeypatch.setattr(m, "read_plan", fake_plan)
    monkeypatch.setattr(m, "post_apply", fake_post)
    monkeypatch.setattr(sys, "argv", ["x", "--apply", "--out", str(tmp_path)])
    assert m.main() == 0
    assert sent["pairs"] == {"d1": [PLAN[0].old_label, PLAN[0].new_label], "d3": [PLAN[2].old_label, PLAN[2].new_label]}
    [run] = list(tmp_path.iterdir())
    assert json.loads((run / "apply_response.json").read_text())["written"] == 2
    assert (run / "snapshot.jsonl").exists()  # snapshot before the write
