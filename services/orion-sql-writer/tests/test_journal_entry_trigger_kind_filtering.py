"""Part A.4 verification: does `journal_entries` (the raw append-only SQL table,
`app/models/journal_entry.py::JournalEntrySQL`) need a `trigger_kind` column now that
`JournalEntryWriteV1` (orion/journaler/schemas.py) carries one?

Answer: no. `_write_row` (services/orion-sql-writer/app/worker.py) filters the
schema-dumped payload down to `{attr.key for attr in inspect(sql_model_cls).attrs}`
before constructing the SQLAlchemy model instance -- any pydantic field without a
matching column is silently dropped, not rejected. This file proves that claim
explicitly (mapper inspection + a real write through `_write_row` against an
in-memory sqlite `journal_entries` table) instead of assuming it's safe.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from orion.journaler.schemas import JournalEntryWriteV1  # noqa: E402

from app.models.journal_entry import JournalEntrySQL  # noqa: E402

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_trigger_kind_filter_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def _write_with_trigger_kind() -> JournalEntryWriteV1:
    return JournalEntryWriteV1(
        entry_id=str(uuid4()),
        author="orion",
        mode="daily",
        body="Daily summary body.",
        source_kind="scheduler",
        trigger_kind="daily_summary",
    )


def test_journal_entries_table_has_no_trigger_kind_column() -> None:
    """`JournalEntryWriteV1.model_dump()` now includes `trigger_kind`, but
    `JournalEntrySQL`'s mapper does not -- confirming the field is schema-only on
    this table, by design (append-only raw table; trigger_kind lives on
    journal_entry_index instead)."""
    mapper = inspect(JournalEntrySQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    dumped = _write_with_trigger_kind().model_dump()
    assert "trigger_kind" in dumped
    assert "trigger_kind" not in valid_keys


def test_write_row_silently_drops_trigger_kind_for_journal_entries_table(monkeypatch) -> None:
    """Real end-to-end proof: `_write_row` against an in-memory sqlite
    `journal_entries` table, given a payload that includes `trigger_kind`, succeeds
    (no unexpected-kwarg error) and the persisted row carries no `trigger_kind`
    attribute at all -- the column simply doesn't exist on this model."""
    engine = create_engine("sqlite://")
    JournalEntrySQL.__table__.create(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    monkeypatch.setattr(worker, "get_session", lambda: session)
    monkeypatch.setattr(worker, "remove_session", lambda: None)

    write = _write_with_trigger_kind()
    data = write.model_dump()

    ok = worker._write_row(JournalEntrySQL, data)
    assert ok is True

    row = session.query(JournalEntrySQL).filter_by(entry_id=write.entry_id).first()
    assert row is not None
    assert row.author == "orion"
    assert not hasattr(row, "trigger_kind")
