# services/orion-hub/tests/test_fcc_mcp_config.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def mcp_config() -> Any:
    import scripts.fcc_mcp_config as mod

    return mod


@pytest.fixture(autouse=True)
def _mock_mcp_toolchain(mcp_config: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_config, "shutil", mcp_config.shutil, raising=False)
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(mcp_config, "_probe_convex_version", lambda *a, **k: None)
    monkeypatch.setattr(mcp_config, "_probe_convex_auth", lambda *a, **k: None)


def test_render_injects_github_and_firecrawl_secrets(mcp_config: Any, tmp_path: Path) -> None:
    out = mcp_config.render_mcp_config(
        correlation_id="corr-1",
        fcc_env={
            "GITHUB_PAT": "ghp_test",
            "FIRECRAWL_API_KEY": "fc_test",
        },
        tmp_dir=tmp_path,
        include_aitown=False,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mcpServers"]["github"]["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_test"
    assert data["mcpServers"]["firecrawl"]["env"]["FIRECRAWL_API_KEY"] == "fc_test"
    assert "orion-aitown" not in data["mcpServers"]


def test_render_fails_without_github_pat(mcp_config: Any, tmp_path: Path) -> None:
    with pytest.raises(mcp_config.McpPreflightError) as exc:
        mcp_config.render_mcp_config(
            correlation_id="corr-2",
            fcc_env={"FIRECRAWL_API_KEY": "fc_test"},
            tmp_dir=tmp_path,
            include_aitown=False,
        )
    assert exc.value.error_code == "fcc_mcp_github_missing"


def test_render_fails_without_docker(mcp_config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: None if cmd == "docker" else f"/usr/bin/{cmd}")
    with pytest.raises(mcp_config.McpPreflightError) as exc:
        mcp_config.render_mcp_config(
            correlation_id="corr-docker",
            fcc_env={"GITHUB_PAT": "ghp_test", "FIRECRAWL_API_KEY": "fc_test"},
            tmp_dir=tmp_path,
            include_aitown=False,
        )
    assert exc.value.error_code == "fcc_mcp_docker_missing"


def test_render_includes_aitown_when_enabled(mcp_config: Any, tmp_path: Path) -> None:
    out = mcp_config.render_mcp_config(
        correlation_id="corr-3",
        fcc_env={
            "GITHUB_PAT": "ghp_test",
            "FIRECRAWL_API_KEY": "fc_test",
            "AITOWN_CONVEX_URL": "http://127.0.0.1:3210",
            "AITOWN_ADMIN_KEY": "admin",
            "AITOWN_WORLD_ID": "world-1",
        },
        tmp_dir=tmp_path,
        include_aitown=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "orion-aitown" in data["mcpServers"]
    assert data["mcpServers"]["orion-aitown"]["env"]["AITOWN_WORLD_ID"] == "world-1"


def test_render_fails_when_aitown_unreachable(mcp_config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_probe(*_a, **_k):
        raise mcp_config.McpPreflightError(error_code="fcc_mcp_aitown_unreachable", message="down")

    monkeypatch.setattr(mcp_config, "_probe_convex_version", _fail_probe)
    with pytest.raises(mcp_config.McpPreflightError) as exc:
        mcp_config.render_mcp_config(
            correlation_id="corr-4",
            fcc_env={
                "GITHUB_PAT": "ghp_test",
                "FIRECRAWL_API_KEY": "fc_test",
                "AITOWN_CONVEX_URL": "http://127.0.0.1:3210",
                "AITOWN_ADMIN_KEY": "admin",
                "AITOWN_WORLD_ID": "world-1",
            },
            tmp_dir=tmp_path,
            include_aitown=True,
        )
    assert exc.value.error_code == "fcc_mcp_aitown_unreachable"
