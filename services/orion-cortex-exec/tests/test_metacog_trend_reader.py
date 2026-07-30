from __future__ import annotations

from typing import Any

import pytest

from app import metacog_trend_reader as mtr


@pytest.fixture(autouse=True)
def _reset_engine(monkeypatch: pytest.MonkeyPatch):
    mtr.reset_metacog_trend_reader_engine_for_tests()
    monkeypatch.setenv(
        "SUBSTRATE_FELT_STATE_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/conjourney",
    )
    monkeypatch.setenv("ENABLE_METACOG_TREND_CUE", "true")
    monkeypatch.setenv("METACOG_TREND_CUE_FETCH_TIMEOUT_SEC", "0.8")
    yield
    mtr.reset_metacog_trend_reader_engine_for_tests()


@pytest.mark.asyncio
async def test_fetch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_fetch_sync() -> dict[str, Any]:
        return {
            "prediction_error": {
                "node:substrate.execution": 0.297,
                "node:substrate.biometrics": 0.155,
            },
            "biometrics_induction": {
                "athena": {"channel": "cpu", "trend": 0.505},
            },
        }

    monkeypatch.setattr(mtr, "_fetch_sync", _fake_fetch_sync)
    cue = await mtr.fetch_recent_trend_signals("corr-1")
    assert cue is not None
    assert cue["prediction_error"]["node:substrate.execution"] == 0.297
    assert cue["biometrics_induction"]["athena"]["channel"] == "cpu"


@pytest.mark.asyncio
async def test_fetch_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_METACOG_TREND_CUE", "false")
    cue = await mtr.fetch_recent_trend_signals("corr-2")
    assert cue is None


@pytest.mark.asyncio
async def test_fetch_dsn_unset_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUBSTRATE_FELT_STATE_DATABASE_URL", raising=False)
    monkeypatch.delenv("ENDOGENOUS_RUNTIME_SQL_DATABASE_URL", raising=False)
    cue = await mtr.fetch_recent_trend_signals("corr-3")
    assert cue is None


@pytest.mark.asyncio
async def test_fetch_timeout_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METACOG_TREND_CUE_FETCH_TIMEOUT_SEC", "0.05")

    def _slow() -> dict[str, Any]:
        import time

        time.sleep(0.3)
        return {"prediction_error": {}, "biometrics_induction": {}}

    monkeypatch.setattr(mtr, "_fetch_sync", _slow)
    cue = await mtr.fetch_recent_trend_signals("corr-4")
    assert cue is None


@pytest.mark.asyncio
async def test_fetch_exception_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> dict[str, Any]:
        raise RuntimeError("db down")

    monkeypatch.setattr(mtr, "_fetch_sync", _boom)
    cue = await mtr.fetch_recent_trend_signals("corr-5")
    assert cue is None


def test_induction_nodes_defaults() -> None:
    assert mtr._induction_nodes() == ("athena", "atlas", "circe")


def test_induction_nodes_parses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METACOG_TREND_CUE_NODES", "athena, atlas")
    assert mtr._induction_nodes() == ("athena", "atlas")


def test_dsn_falls_back_to_endogenous_runtime_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUBSTRATE_FELT_STATE_DATABASE_URL", raising=False)
    monkeypatch.setenv(
        "ENDOGENOUS_RUNTIME_SQL_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/conjourney",
    )
    assert mtr._dsn() == "postgresql://postgres:postgres@localhost:5432/conjourney"
