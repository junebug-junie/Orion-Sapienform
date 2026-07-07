"""SQL read path for action outcomes (Option C: bus -> sql-writer -> SQL read).

Uses an on-disk SQLite database as a stand-in for the shared Postgres store to
exercise `load_action_outcomes` when `ORION_ACTION_OUTCOME_DB_URL` is set, plus
the graceful file fallback when the SQL read fails.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

from orion.autonomy import action_outcomes
from orion.autonomy.action_outcomes import append_action_outcome, load_action_outcomes
from orion.autonomy.models import ActionOutcomeRefV1


def _make_db(path) -> str:
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE action_outcomes (
                action_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                kind TEXT NOT NULL,
                summary TEXT NOT NULL,
                success BOOLEAN NULL,
                surprise REAL NOT NULL DEFAULT 0.0,
                observed_at TIMESTAMP NULL,
                correlation_id TEXT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
    engine.dispose()
    return url


@pytest.fixture(autouse=True)
def _clear_engine_cache():
    action_outcomes._ENGINE_CACHE.clear()
    yield
    action_outcomes._ENGINE_CACHE.clear()


def test_load_from_sql_returns_chronological_outcomes(tmp_path, monkeypatch) -> None:
    url = _make_db(tmp_path / "outcomes.db")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_DB_URL", url)

    base = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    engine = create_engine(url)
    with engine.begin() as conn:
        for i in range(3):
            conn.execute(
                text(
                    "INSERT INTO action_outcomes "
                    "(action_id, subject, kind, summary, success, surprise, observed_at, correlation_id) "
                    "VALUES (:action_id, :subject, :kind, :summary, :success, :surprise, :observed_at, :corr)"
                ),
                {
                    "action_id": f"fetch-{i}",
                    "subject": "orion",
                    "kind": "web.fetch.readonly",
                    "summary": f"fetched {i} article(s)",
                    "success": i % 2 == 0,
                    "surprise": 0.1 * i,
                    "observed_at": base + timedelta(minutes=i),
                    "corr": f"wp-run-{i}",
                },
            )
    engine.dispose()

    loaded = load_action_outcomes(subject="orion")
    assert [o.action_id for o in loaded] == ["fetch-0", "fetch-1", "fetch-2"]
    assert loaded[-1].summary == "fetched 2 article(s)"
    assert loaded[0].kind == "web.fetch.readonly"


def test_load_from_sql_filters_by_subject(tmp_path, monkeypatch) -> None:
    url = _make_db(tmp_path / "outcomes.db")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_DB_URL", url)

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO action_outcomes (action_id, subject, kind, summary, surprise, observed_at) "
                "VALUES ('a', 'orion', 'web.fetch.readonly', 's', 0.0, :ts), "
                "('b', 'juniper', 'web.fetch.readonly', 's', 0.0, :ts)"
            ),
            {"ts": datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)},
        )
    engine.dispose()

    assert [o.action_id for o in load_action_outcomes(subject="orion")] == ["a"]
    assert [o.action_id for o in load_action_outcomes(subject="juniper")] == ["b"]


def test_load_falls_back_to_file_when_sql_unavailable(tmp_path, monkeypatch) -> None:
    # DB URL points at a nonexistent table -> read raises -> file fallback used.
    bad_url = f"sqlite:///{tmp_path / 'empty.db'}"
    create_engine(bad_url).dispose()  # create empty db, no table
    monkeypatch.setenv("ORION_ACTION_OUTCOME_DB_URL", bad_url)
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))

    ref = ActionOutcomeRefV1(
        action_id="fetch-file",
        kind="web.fetch.readonly",
        summary="fetched 1 article(s)",
        success=True,
        surprise=0.0,
        observed_at=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
    )
    append_action_outcome(subject="orion", outcome=ref)

    loaded = load_action_outcomes(subject="orion")
    assert [o.action_id for o in loaded] == ["fetch-file"]


def test_load_uses_file_when_db_url_unset(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ORION_ACTION_OUTCOME_DB_URL", raising=False)
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    ref = ActionOutcomeRefV1(
        action_id="fetch-nofdb",
        kind="web.fetch.readonly",
        summary="fetched 1 article(s)",
        observed_at=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
    )
    append_action_outcome(subject="orion", outcome=ref)
    loaded = load_action_outcomes(subject="orion")
    assert [o.action_id for o in loaded] == ["fetch-nofdb"]
