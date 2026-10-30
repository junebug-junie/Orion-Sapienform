"""`dig capacities`: read-only, capped, the full reading in a file, no ledger written."""
from __future__ import annotations

import json
from pathlib import Path

from orion.body_atlas.tests.test_capacities import _body
from orion.dig import __main__ as dig
from orion.dig.tests.test_dig import run, world  # noqa: F401  (fixture)


def test_capacities_prints_every_job_and_writes_the_full_reading_only(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, out, _ = run(capsys, "capacities")
    assert code == 0 and len(out.splitlines()) <= dig.MAX_LINES and len(out) <= dig.MAX_CHARS
    assert "S2 coordination" in out and "pool (6 requesters / 3 workers, replies 2)" in out
    assert "servers gateway" in out
    path = Path(out.strip().splitlines()[-1].removeprefix("full result: "))
    reading = json.loads(path.read_text())
    assert reading["schema_version"] == "capacity_reading.v1" and reading["live_parts"] == 13
    written = sorted(p.relative_to(world / "home").as_posix() for p in (world / "home").rglob("*") if p.is_file())
    assert written == ["dig/t/capacities-1.json"]  # no ledger, no self/ file


def test_capacities_gaps_lists_code_written_gap_lines(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, out, _ = run(capsys, "capacities", "gaps")
    assert code == 0 and "I cannot remove them" in out and "- S3: I cannot see my own control" in out


def test_capacities_with_an_unknown_word_is_malformed(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, _, err = run(capsys, "capacities", "claim")
    assert code == 2 and "unknown or malformed" in err
