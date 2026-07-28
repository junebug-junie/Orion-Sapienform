"""Regression coverage for digest.py::summarize_window()'s real DB queries.

Found in review (2026-07-28, PR fixing notify-digest's POSTGRES_URI): this service had
never once actually run summarize_window() against a real database in its test suite --
test_digest_topics.py only covers the Topic Foundry HTTP helpers, and
test_heartbeat_chassis.py explicitly no-ops init_models. This exercises the real query path
(severity/event_kind/source_service grouping, critical/warning event selection, failed
attempts, throttled/deduped counts) against a real (temp, file-backed) SQLite database
built from this service's own SQLAlchemy models -- independent of which backend
(SQLite/Postgres) POSTGRES_URI actually points at in production.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.digest as digest_mod
from app.db_models import NotificationAttemptDB, NotificationRequestDB
from orion.core.sql_router.db import Base

WINDOW_END = datetime(2026, 7, 28, 12, 0, 0)
WINDOW_START = WINDOW_END - timedelta(hours=24)


@pytest.fixture()
def temp_session(tmp_path, monkeypatch):
    db_path = tmp_path / "notify_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(
        bind=engine, tables=[NotificationRequestDB.__table__, NotificationAttemptDB.__table__]
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(digest_mod, "SessionLocal", session_factory)
    return session_factory


def _request(
    session_factory,
    *,
    notification_id: str,
    source_service: str,
    event_kind: str,
    severity: str,
    status: str,
    created_at: datetime,
) -> None:
    with session_factory() as db:
        db.add(
            NotificationRequestDB(
                notification_id=notification_id,
                source_service=source_service,
                event_kind=event_kind,
                severity=severity,
                title="t",
                context={},
                tags=[],
                recipient_group="juniper_primary",
                created_at=created_at,
                status=status,
            )
        )
        db.commit()


def _attempt(
    session_factory, *, attempt_id: str, notification_id: str, status: str, attempted_at: datetime
) -> None:
    with session_factory() as db:
        db.add(
            NotificationAttemptDB(
                attempt_id=attempt_id,
                notification_id=notification_id,
                channel="email",
                status=status,
                attempted_at=attempted_at,
            )
        )
        db.commit()


def test_summarize_window_counts_severity_and_top_kinds(temp_session) -> None:
    within = WINDOW_END - timedelta(hours=1)
    _request(
        temp_session,
        notification_id="n1",
        source_service="orion-actions",
        event_kind="orion.daily.pulse",
        severity="info",
        status="pending",
        created_at=within,
    )
    _request(
        temp_session,
        notification_id="n2",
        source_service="orion-actions",
        event_kind="orion.daily.pulse",
        severity="info",
        status="pending",
        created_at=within,
    )
    _request(
        temp_session,
        notification_id="n3",
        source_service="orion-substrate-runtime",
        event_kind="orion.system.error",
        severity="critical",
        status="pending",
        created_at=within,
    )

    summary = digest_mod.summarize_window(WINDOW_START, WINDOW_END)

    assert summary.severity_counts == {"info": 2, "critical": 1}
    assert ("orion.daily.pulse", 2) in summary.top_event_kinds
    assert len(summary.critical_events) == 1
    assert summary.critical_events[0].notification_id == "n3"


def test_summarize_window_excludes_outside_window(temp_session) -> None:
    outside = WINDOW_START - timedelta(hours=1)
    _request(
        temp_session,
        notification_id="old",
        source_service="orion-actions",
        event_kind="orion.daily.pulse",
        severity="info",
        status="pending",
        created_at=outside,
    )

    summary = digest_mod.summarize_window(WINDOW_START, WINDOW_END)

    assert summary.severity_counts == {}
    assert summary.top_event_kinds == []


def test_summarize_window_counts_throttled_and_deduped(temp_session) -> None:
    within = WINDOW_END - timedelta(hours=1)
    _request(
        temp_session,
        notification_id="t1",
        source_service="orion-actions",
        event_kind="orion.daily.pulse",
        severity="info",
        status="throttled",
        created_at=within,
    )
    _request(
        temp_session,
        notification_id="d1",
        source_service="orion-actions",
        event_kind="orion.daily.pulse",
        severity="info",
        status="deduped",
        created_at=within,
    )

    summary = digest_mod.summarize_window(WINDOW_START, WINDOW_END)

    assert summary.throttled_count == 1
    assert summary.deduped_count == 1


def test_summarize_window_includes_failed_attempts(temp_session) -> None:
    within = WINDOW_END - timedelta(hours=1)
    _attempt(
        temp_session,
        attempt_id="a1",
        notification_id="n1",
        status="failed",
        attempted_at=within,
    )
    _attempt(
        temp_session,
        attempt_id="a2",
        notification_id="n2",
        status="sent",
        attempted_at=within,
    )

    summary = digest_mod.summarize_window(WINDOW_START, WINDOW_END)

    assert len(summary.failed_attempts) == 1
    assert summary.failed_attempts[0].attempt_id == "a1"
