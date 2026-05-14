from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app import main as app_main


def test_debug_queue_404_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_main.settings, "spark_introspection_queue_debug_enabled", False)

    async def _run() -> None:
        with pytest.raises(HTTPException) as ei:
            await app_main.spark_introspection_queue_debug(
                x_spark_introspection_debug_token=None,
            )
        assert ei.value.status_code == 404

    asyncio.run(_run())


def test_debug_queue_503_when_enabled_but_token_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_main.settings, "spark_introspection_queue_debug_enabled", True)
    monkeypatch.setattr(app_main.settings, "spark_introspection_queue_debug_token", "")

    async def _run() -> None:
        with pytest.raises(HTTPException) as ei:
            await app_main.spark_introspection_queue_debug(
                x_spark_introspection_debug_token=None,
            )
        assert ei.value.status_code == 503

    asyncio.run(_run())


def test_debug_queue_401_wrong_token_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_main.settings, "spark_introspection_queue_debug_enabled", True)
    monkeypatch.setattr(app_main.settings, "spark_introspection_queue_debug_token", "secret")

    async def _run() -> None:
        with pytest.raises(HTTPException) as ei:
            await app_main.spark_introspection_queue_debug(
                x_spark_introspection_debug_token="wrong",
            )
        assert ei.value.status_code == 401

    asyncio.run(_run())


def test_debug_queue_ok_with_header_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_main.settings, "spark_introspection_queue_debug_enabled", True)
    monkeypatch.setattr(app_main.settings, "spark_introspection_queue_debug_token", "secret")

    async def _run() -> None:
        with patch.object(app_main, "get_spark_queue_status", new=AsyncMock(return_value={"enabled": True})):
            out = await app_main.spark_introspection_queue_debug(
                x_spark_introspection_debug_token="secret",
            )
            assert out == {"enabled": True}

    asyncio.run(_run())


def test_debug_token_equal_uses_sha256_digest_compare() -> None:
    assert app_main._debug_token_equal("a", "a") is True
    assert app_main._debug_token_equal("a", "b") is False
    assert app_main._debug_token_equal("short", "much_longer_expected") is False
