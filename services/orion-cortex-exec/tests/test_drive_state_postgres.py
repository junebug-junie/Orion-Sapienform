from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app import drive_state_postgres as dsp


@pytest.fixture(autouse=True)
def _reset_engine(monkeypatch: pytest.MonkeyPatch):
    dsp.reset_drive_state_postgres_engine_for_tests()
    monkeypatch.setenv(
        "ORION_ACTION_OUTCOME_DB_URL",
        "postgresql://postgres:postgres@localhost:5432/conjourney",
    )
    monkeypatch.setenv("CHAT_STANCE_DRIVE_STATE_FETCH_TIMEOUT_SEC", "0.4")
    yield
    dsp.reset_drive_state_postgres_engine_for_tests()


def test_row_to_stance_drive_state_maps_columns() -> None:
    mapped = dsp._row_to_stance_drive_state(
        {
            "dominant_drive": "coherence",
            "active_drives": ["coherence", "continuity"],
            "drive_pressures": {"coherence": 0.8, "continuity": 0.4},
            "summary": "pressure on coherence",
            "tension_kinds": ["drive_competition.coherence_continuity"],
        }
    )
    assert mapped == {
        "pressures": {"coherence": 0.8, "continuity": 0.4},
        "activations": {"coherence": True, "continuity": True},
        "dominant_drive": "coherence",
        "summary": "pressure on coherence",
        "tension_kinds": ["drive_competition.coherence_continuity"],
    }


def test_row_to_stance_drive_state_coerces_jsonb_strings() -> None:
    mapped = dsp._row_to_stance_drive_state(
        {
            "dominant_drive": "coherence",
            "active_drives": '["coherence"]',
            "drive_pressures": '{"coherence": 0.9}',
            "summary": "ok",
            "tension_kinds": '["unresolved_thread"]',
        }
    )
    assert mapped is not None
    assert mapped["activations"] == {"coherence": True}
    assert mapped["pressures"] == {"coherence": 0.9}
    assert mapped["tension_kinds"] == ["unresolved_thread"]


def test_quiet_tick_returns_none() -> None:
    assert (
        dsp._row_to_stance_drive_state(
            {
                "dominant_drive": None,
                "active_drives": [],
                "drive_pressures": {"coherence": 0.1},
                "summary": None,
                "tension_kinds": [],
            }
        )
        is None
    )


@pytest.mark.asyncio
async def test_fetch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_query() -> dict[str, Any]:
        return {
            "dominant_drive": "coherence",
            "active_drives": ["coherence"],
            "drive_pressures": {"coherence": 0.7},
            "summary": "ok",
            "tension_kinds": [],
        }

    monkeypatch.setattr(dsp, "_query_latest_drive_audit_row_sync", _fake_query)
    compact, diag = await dsp.fetch_drive_state_for_chat_stance("corr-1")
    assert compact is not None
    assert compact["dominant_drive"] == "coherence"
    assert compact["pressures"] == {"coherence": 0.7}
    assert diag["ok"] is True
    assert diag["reason"] == "success"
    assert diag["source"] == "drive_audits"


@pytest.mark.asyncio
async def test_fetch_dsn_unset_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORION_ACTION_OUTCOME_DB_URL", raising=False)
    monkeypatch.delenv("ENDOGENOUS_RUNTIME_SQL_DATABASE_URL", raising=False)
    compact, diag = await dsp.fetch_drive_state_for_chat_stance("corr-2")
    assert compact is None
    assert diag["reason"] == "dsn_unset"


@pytest.mark.asyncio
async def test_fetch_timeout_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_STANCE_DRIVE_STATE_FETCH_TIMEOUT_SEC", "0.05")

    def _slow() -> dict[str, Any]:
        import time

        time.sleep(0.2)
        return {"dominant_drive": "coherence", "active_drives": ["coherence"], "summary": "x"}

    monkeypatch.setattr(dsp, "_query_latest_drive_audit_row_sync", _slow)
    compact, diag = await dsp.fetch_drive_state_for_chat_stance("corr-3")
    assert compact is None
    assert diag["timed_out"] is True
    assert diag["reason"] == "timeout"


@pytest.mark.asyncio
async def test_fetch_exception_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> dict[str, Any]:
        raise RuntimeError("db down")

    monkeypatch.setattr(dsp, "_query_latest_drive_audit_row_sync", _boom)
    compact, diag = await dsp.fetch_drive_state_for_chat_stance("corr-4")
    assert compact is None
    assert diag["reason"] == "exception"
    assert diag["exception_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_fetch_no_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dsp, "_query_latest_drive_audit_row_sync", lambda: None)
    compact, diag = await dsp.fetch_drive_state_for_chat_stance("corr-5")
    assert compact is None
    assert diag["reason"] == "no_rows"


@pytest.mark.asyncio
async def test_fetch_no_meaningful_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dsp,
        "_query_latest_drive_audit_row_sync",
        lambda: {
            "dominant_drive": None,
            "active_drives": [],
            "drive_pressures": {"coherence": 0.1},
            "summary": None,
            "tension_kinds": [],
        },
    )
    compact, diag = await dsp.fetch_drive_state_for_chat_stance("corr-quiet")
    assert compact is None
    assert diag["reason"] == "no_meaningful_content"


def test_select_columns_subset_of_drive_audit_sql_model() -> None:
    """Guard against SELECT drift: fail-open would hide UndefinedColumn forever."""
    model_path = (
        Path(__file__).resolve().parents[3]
        / "services"
        / "orion-sql-writer"
        / "app"
        / "models"
        / "drive_audit.py"
    )
    source = model_path.read_text(encoding="utf-8")
    selected = (
        "dominant_drive",
        "active_drives",
        "drive_pressures",
        "summary",
        "tension_kinds",
        "observed_at",
        "created_at",
        "subject",
    )
    missing = [col for col in selected if col not in source]
    assert not missing, f"stance SELECT references missing drive_audits columns: {missing}"
    for col in selected:
        assert col in dsp.DRIVE_AUDITS_LATEST_QUERY_FOR_STANCE
