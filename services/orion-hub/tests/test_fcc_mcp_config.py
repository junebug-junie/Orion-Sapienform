# services/orion-hub/tests/test_fcc_mcp_config.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.fcc_mcp_config import McpPreflightError, render_mcp_config


def test_render_injects_github_and_firecrawl_secrets(tmp_path: Path) -> None:
    out = render_mcp_config(
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


def test_render_fails_without_github_pat(tmp_path: Path) -> None:
    with pytest.raises(McpPreflightError) as exc:
        render_mcp_config(
            correlation_id="corr-2",
            fcc_env={"FIRECRAWL_API_KEY": "fc_test"},
            tmp_dir=tmp_path,
            include_aitown=False,
        )
    assert exc.value.error_code == "fcc_mcp_github_missing"


def test_render_includes_aitown_when_enabled(tmp_path: Path) -> None:
    out = render_mcp_config(
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
