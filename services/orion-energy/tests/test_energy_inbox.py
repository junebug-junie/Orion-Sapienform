from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.inbox import load_processed, scan_inbox

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "orion/energy/tests/fixtures/espi_two_flows.xml"
NOW = datetime(2026, 9, 11, 12, 0, 5, tzinfo=timezone.utc)


def test_scan_parses_and_moves_to_processed(tmp_path: Path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    inbox.mkdir()
    shutil.copy(FIXTURE, inbox / "sept.xml")
    rows = scan_inbox(inbox, processed, now=NOW)
    assert len(rows) == 3
    assert all(r.retrieved_at == NOW.replace(microsecond=0) for r in rows)
    assert not (inbox / "sept.xml").exists()
    assert [p.name for p in processed.iterdir()] == ["20260911T120005Z__sept.xml"]
    assert rows[0].source_file == "20260911T120005Z__sept.xml"


def test_unparseable_file_goes_to_failed_not_processed(tmp_path: Path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    inbox.mkdir()
    (inbox / "junk.xml").write_bytes(b"<nope")
    assert scan_inbox(inbox, processed, now=NOW) == []
    assert (inbox / "failed" / "junk.xml").exists()
    assert not processed.exists() or not any(processed.iterdir())


def test_non_xml_files_ignored(tmp_path: Path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    inbox.mkdir()
    (inbox / "notes.txt").write_text("hi")
    assert scan_inbox(inbox, processed, now=NOW) == []
    assert (inbox / "notes.txt").exists()


def test_replay_restores_retrieved_at_from_filename(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    shutil.copy(FIXTURE, processed / "20260911T120005Z__sept.xml")
    rows = load_processed(processed)
    assert len(rows) == 3
    assert rows[0].retrieved_at == datetime(2026, 9, 11, 12, 0, 5, tzinfo=timezone.utc)
