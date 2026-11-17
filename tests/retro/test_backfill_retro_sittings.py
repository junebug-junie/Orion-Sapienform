"""scripts/backfill_retro_sittings.py: a dry run plans rows for retro_sitting only, the same way twice."""
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("backfill_retro_sittings", ROOT / "scripts/backfill_retro_sittings.py")
backfill = importlib.util.module_from_spec(spec); spec.loader.exec_module(backfill)
SID = "c" * 36


def _journal(path: Path, game: str, presses: list[int], controller: str | None = "human") -> None:
    rows = [{"kind": "start", "session_id": SID, "game_id": game, "utc": "2026-11-10T20:00:00+00:00"}]
    for i, sec in enumerate(presses):
        row = {"kind": "applied", "session_id": SID, "port": 0, "mask": 16 if i % 2 == 0 else 0,
               "utc": f"2026-11-10T20:{sec // 60:02d}:{sec % 60:02d}+00:00"}
        if controller: row["controller"] = controller
        rows.append(row)
        if i == 2:   # a journal is in time order: the save sits between two presses
            rows.append({"kind": "save", "session_id": SID, "utc": f"2026-11-10T20:{sec // 60:02d}:{sec % 60 + 5:02d}+00:00"})
    # written as the service writes it: compact separators
    path.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n")


def test_dry_run_plans_sittings_from_mask_changes_and_writes_no_database(tmp_path):
    evidence, out = tmp_path / "evidence", tmp_path / "out"
    evidence.mkdir()
    _journal(evidence / "one.jsonl", "a-link-to-the-past", [10, 70, 130, 190, 250, 900, 905])   # two sittings: a 650 s gap
    _journal(evidence / "old.jsonl", "crystalis", [10, 70, 130], controller=None)                 # unattributed: not claimed
    (tmp_path / "catalog.json").write_text(json.dumps([{"game_id": "a-link-to-the-past", "title": "A Link to the Past"}]))
    argv = ["--evidence-dir", str(evidence), "--catalog", str(tmp_path / "catalog.json"), "--out", str(out),
            "--now", "2026-11-17T00:00:00+00:00", "--psql", "false"]
    assert backfill.main(argv) == 0
    rows = list(csv.DictReader((out / "plan.csv").open()))
    assert [(r["game_title"], r["mask_changes"], r["saves"], r["backfilled"], r["end_reason"]) for r in rows] == [
        ("A Link to the Past", "5", "1", "True", "idle"), ("A Link to the Past", "2", "0", "True", "idle")]
    assert all(r["sitting_id"].startswith("backfill-") and r["player_stated"] == "" for r in rows)
    sql = (out / "inserts.sql").read_text()
    assert sql.startswith("INSERT INTO retro_sitting (") and "ON CONFLICT (sitting_id) DO NOTHING" in sql
    assert "retro_play_sessions" not in sql and "temporal_self" not in sql      # a baseline, not a memory
    report = (out / "report.md").read_text()
    assert "DRY RUN: nothing was written" in report and "| crystalis | 1 | 0 | 0 | - | 3 | 0 | 0 | 0 |" in report
    assert list(csv.reader((out / "before_after.csv").open()))[1] == ["retro_sitting", "", "", "2"]
    first_ids = [r["sitting_id"] for r in rows]
    assert backfill.main(argv) == 0                                             # a second run plans the same ids
    assert [r["sitting_id"] for r in csv.DictReader((out / "plan.csv").open())] == first_ids


def test_a_sitting_older_than_the_window_is_left_out(tmp_path):
    evidence = tmp_path / "evidence"; evidence.mkdir()
    _journal(evidence / "one.jsonl", "tetris", [10, 70])
    out = tmp_path / "out"
    assert backfill.main(["--evidence-dir", str(evidence), "--out", str(out), "--days", "1",
                          "--now", "2026-11-17T00:00:00+00:00", "--catalog", str(tmp_path / "none.json")]) == 0
    assert list(csv.DictReader((out / "plan.csv").open())) == []
    assert (out / "inserts.sql").read_text().startswith("-- no sittings")
