from __future__ import annotations

from datetime import datetime, timezone

from app.endogenous_runtime_sql_reader import EndogenousRuntimeSqlReader


class _Conn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, query, params):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def mappings(self):
                return self

            def all(self):
                return self._rows

        return _Result(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Engine:
    def __init__(self, rows):
        self.rows = rows

    def connect(self):
        return _Conn(self.rows)


def test_runtime_reader_filters_and_limit_are_bounded() -> None:
    reader = EndogenousRuntimeSqlReader(enabled=False, database_url="sqlite://")
    reader._enabled = True
    reader._engine = _Engine(rows=[{"payload": {"runtime_record_id": "r1"}}])

    out = reader.runtime_records(
        limit=50000,
        invocation_surface="operator_review",
        workflow_type="reflective_journal",
        outcome="trigger",
        audit_status="ok",
        subject_ref="project:orion",
        created_after=datetime.now(timezone.utc),
    )
    assert out.source == "sql"
    assert out.rows[0]["runtime_record_id"] == "r1"
    assert out.filters["limit"] == 50000


def test_calibration_reader_profile_history_query() -> None:
    reader = EndogenousRuntimeSqlReader(enabled=False, database_url="sqlite://")
    reader._enabled = True
    reader._engine = _Engine(rows=[{"payload": {"audit_id": "a1", "profile_id": "p1", "event_type": "activated"}}])

    out = reader.calibration_audit(limit=20, profile_id="p1", event_type="activated")
    assert out.source == "sql"
    assert out.rows[0]["profile_id"] == "p1"
    assert out.rows[0]["event_type"] == "activated"
