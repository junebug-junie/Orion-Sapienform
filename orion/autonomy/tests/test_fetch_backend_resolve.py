from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orion.autonomy.fetch_backend_resolve import resolve_fetch_backend, resolve_firecrawl_api_key
from orion.autonomy.fetch_backends import firecrawl_search_backend


def test_resolve_firecrawl_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "direct-key")
    assert resolve_firecrawl_api_key() == "direct-key"


def test_resolve_firecrawl_api_key_from_fcc_file(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    env_file = tmp_path / ".fcc.env"
    env_file.write_text('FIRECRAWL_API_KEY="file-key"\n', encoding="utf-8")
    monkeypatch.setenv("ORION_FCC_ENV_PATH", str(env_file))
    assert resolve_firecrawl_api_key() == "file-key"


def test_resolve_fetch_backend_uses_firecrawl_when_key_present(monkeypatch) -> None:
    monkeypatch.setenv("ORION_EPISODE_FETCH_BACKEND", "auto")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "direct-key")
    backend = resolve_fetch_backend()
    assert backend.func is firecrawl_search_backend
    assert backend.keywords["api_key"] == "direct-key"


def test_resolve_fetch_backend_stub_mode(monkeypatch) -> None:
    from orion.autonomy.episode_fetch import default_fetch_backend

    monkeypatch.setenv("ORION_EPISODE_FETCH_BACKEND", "stub")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "direct-key")
    assert resolve_fetch_backend() is default_fetch_backend


@pytest.mark.asyncio
async def test_resolve_fetch_backend_without_key_returns_stub(monkeypatch) -> None:
    from orion.autonomy.episode_fetch import default_fetch_backend

    monkeypatch.setenv("ORION_EPISODE_FETCH_BACKEND", "auto")
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setenv("ORION_FCC_ENV_PATH", "/tmp/nonexistent-fcc.env")
    assert resolve_fetch_backend() is default_fetch_backend
