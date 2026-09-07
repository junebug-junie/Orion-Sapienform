from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from app import recent_attention_reader as rar


@pytest.fixture(autouse=True)
def _reset_engine(monkeypatch: pytest.MonkeyPatch):
    rar.reset_recent_attention_reader_engine_for_tests()
    monkeypatch.setenv(
        "SUBSTRATE_FELT_STATE_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/conjourney",
    )
    monkeypatch.setenv("ENABLE_RECENT_ATTENTION_CUE", "true")
    monkeypatch.setenv("RECENT_ATTENTION_CUE_FETCH_TIMEOUT_SEC", "0.8")
    yield
    rar.reset_recent_attention_reader_engine_for_tests()


@pytest.mark.asyncio
async def test_fetch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_fetch_sync() -> dict[str, Any]:
        return {
            "items": [
                {
                    "process": "cortex_turn",
                    "narrative": "Following the chat turn.",
                    "age_label": "moments ago",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "stale": False,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    monkeypatch.setattr(rar, "_fetch_sync", _fake_fetch_sync)
    cue = await rar.fetch_recent_attention_cue("corr-1")
    assert cue is not None
    assert cue["items"][0]["process"] == "cortex_turn"
    assert cue["stale"] is False


@pytest.mark.asyncio
async def test_fetch_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_RECENT_ATTENTION_CUE", "false")
    cue = await rar.fetch_recent_attention_cue("corr-2")
    assert cue is None


@pytest.mark.asyncio
async def test_fetch_dsn_unset_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUBSTRATE_FELT_STATE_DATABASE_URL", raising=False)
    monkeypatch.delenv("ENDOGENOUS_RUNTIME_SQL_DATABASE_URL", raising=False)
    cue = await rar.fetch_recent_attention_cue("corr-3")
    assert cue is None


@pytest.mark.asyncio
async def test_fetch_timeout_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECENT_ATTENTION_CUE_FETCH_TIMEOUT_SEC", "0.05")

    def _slow() -> dict[str, Any]:
        import time

        time.sleep(0.3)
        return {"items": [], "stale": True, "as_of": ""}

    monkeypatch.setattr(rar, "_fetch_sync", _slow)
    cue = await rar.fetch_recent_attention_cue("corr-4")
    assert cue is None


@pytest.mark.asyncio
async def test_fetch_exception_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> dict[str, Any]:
        raise RuntimeError("db down")

    monkeypatch.setattr(rar, "_fetch_sync", _boom)
    cue = await rar.fetch_recent_attention_cue("corr-5")
    assert cue is None


def test_limit_default() -> None:
    assert rar._limit() == 3


def test_limit_parses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECENT_ATTENTION_CUE_LIMIT", "5")
    assert rar._limit() == 5


def test_stale_after_sec_default() -> None:
    assert rar._stale_after_sec() == 900.0


def test_query_filters_empty_narrative_before_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """`reason_narrative` is NOT NULL DEFAULT '' on the live table -- an
    empty-narrative row is real, not absent. Filtering it must happen in SQL,
    before LIMIT, so an empty-narrative burst in the most recent rows can't
    starve the cue of real narrated rows sitting just past the window."""
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.all.return_value = []
    monkeypatch.setattr(rar, "_get_engine", lambda: engine)

    rar._fetch_sync()

    sql = str(conn.execute.call_args[0][0])
    where_clause, _, order_clause = sql.partition("ORDER BY")
    assert "reason_narrative <> ''" in where_clause
    assert "LIMIT" in order_clause


def test_dsn_falls_back_to_endogenous_runtime_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUBSTRATE_FELT_STATE_DATABASE_URL", raising=False)
    monkeypatch.setenv(
        "ENDOGENOUS_RUNTIME_SQL_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/conjourney",
    )
    assert rar._dsn() == "postgresql://postgres:postgres@localhost:5432/conjourney"
