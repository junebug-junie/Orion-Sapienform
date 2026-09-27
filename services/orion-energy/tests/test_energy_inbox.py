from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.inbox import latest_processed_at, load_processed, scan_inbox

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
    assert (inbox / "failed" / "20260911T120005Z__junk.xml").exists()
    assert not processed.exists() or not any(processed.iterdir())


def test_repeated_failures_with_same_name_both_survive(tmp_path: Path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    inbox.mkdir()
    (inbox / "junk.xml").write_bytes(b"<first")
    scan_inbox(inbox, processed, now=NOW)
    (inbox / "junk.xml").write_bytes(b"<second")
    scan_inbox(inbox, processed, now=NOW.replace(minute=5))
    failed = sorted((inbox / "failed").iterdir())
    assert [p.name for p in failed] == ["20260911T120005Z__junk.xml", "20260911T120505Z__junk.xml"]
    assert [p.read_bytes() for p in failed] == [b"<first", b"<second"]


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


def test_read_oserror_leaves_file_and_still_returns_other_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    caplog.set_level(logging.WARNING, logger="orion-energy.inbox")
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    inbox.mkdir()
    shutil.copy(FIXTURE, inbox / "good.xml")
    bad = inbox / "bad.xml"
    bad.write_text("<feed/>")

    real_read = Path.read_bytes

    def _read_bytes(self: Path) -> bytes:
        if self.name == "bad.xml":
            raise OSError("read failed")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", _read_bytes)
    rows = scan_inbox(inbox, processed, now=NOW)
    assert len(rows) == 3
    assert (inbox / "bad.xml").exists()
    assert not (processed / "bad.xml").exists()
    assert any("energy_inbox_io_failed" in r.getMessage() for r in caplog.records)


def test_portal_named_file_is_labeled_rockymountain_power(tmp_path: Path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    inbox.mkdir()
    shutil.copy(FIXTURE, inbox / "rmp-portal-20260927T060000Z.xml")
    shutil.copy(FIXTURE, inbox / "manual.xml")
    now = datetime(2026, 9, 27, 6, tzinfo=timezone.utc)
    rows = scan_inbox(inbox, processed, now=now)
    sources = {r.source_file.split("__", 1)[1]: r.source for r in rows}
    assert sources["rmp-portal-20260927T060000Z.xml"] == "rockymountain_power"
    assert sources["manual.xml"] == "file_drop"
    replayed = {r.source for r in load_processed(processed)}
    assert replayed == {"rockymountain_power", "file_drop"}
    assert latest_processed_at(processed, ".xml") == now


def test_latest_processed_at_empty(tmp_path: Path) -> None:
    assert latest_processed_at(tmp_path / "missing", ".xml") is None
