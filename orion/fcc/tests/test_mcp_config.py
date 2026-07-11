from __future__ import annotations

import json
from pathlib import Path

import pytest

import orion.fcc.mcp_config as mcp_config
from orion.fcc.claude_spawn import mcp_allowed_tool_patterns
from orion.fcc.mcp_config import McpPreflightError, render_mcp_config

_BASE_ENV = {"GITHUB_PAT": "ghp_test", "FIRECRAWL_API_KEY": "fc_test"}


def _patch_all_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")


def _render(tmp_path: Path, **kwargs) -> dict:
    path = render_mcp_config(
        correlation_id="corr-test",
        fcc_env=_BASE_ENV,
        tmp_dir=tmp_path / "render",
        **kwargs,
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_render_defaults_exclude_self_index_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_all_tools(monkeypatch)
    data = _render(tmp_path)
    assert set(data["mcpServers"]) == {"github", "firecrawl"}


def test_render_gitnexus_adds_server_without_extra_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_all_tools(monkeypatch)
    data = _render(tmp_path, include_gitnexus=True)
    assert data["mcpServers"]["gitnexus"] == {
        "type": "stdio",
        "command": "gitnexus",
        "args": ["mcp"],
    }


def test_render_gitnexus_missing_binary_is_specific_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mcp_config.shutil,
        "which",
        lambda cmd: None if cmd == "gitnexus" else f"/usr/bin/{cmd}",
    )
    with pytest.raises(McpPreflightError) as exc:
        _render(tmp_path, include_gitnexus=True)
    assert exc.value.error_code == "fcc_mcp_gitnexus_missing"


def test_render_context_mode_adds_server_with_scoped_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_all_tools(monkeypatch)
    storage = tmp_path / "ctx-data"
    data = _render(
        tmp_path,
        include_context_mode=True,
        context_mode_dir=str(storage),
        context_mode_project_dir="/mnt/scripts/Orion-Sapienform",
    )
    server = data["mcpServers"]["context-mode"]
    assert server["command"] == "context-mode"
    assert server["env"] == {
        "CONTEXT_MODE_PLATFORM": "claude-code",
        "CONTEXT_MODE_PROJECT_DIR": "/mnt/scripts/Orion-Sapienform",
        "CONTEXT_MODE_DIR": str(storage),
    }
    assert storage.is_dir()


def test_render_context_mode_missing_binary_is_specific_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mcp_config.shutil,
        "which",
        lambda cmd: None if cmd == "context-mode" else f"/usr/bin/{cmd}",
    )
    with pytest.raises(McpPreflightError) as exc:
        _render(
            tmp_path,
            include_context_mode=True,
            context_mode_dir=str(tmp_path / "ctx"),
            context_mode_project_dir="/repo",
        )
    assert exc.value.error_code == "fcc_mcp_context_mode_missing"


@pytest.mark.parametrize("bad_dir", ["", "relative/path"])
def test_render_context_mode_rejects_non_absolute_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_dir: str
) -> None:
    _patch_all_tools(monkeypatch)
    with pytest.raises(McpPreflightError) as exc:
        _render(
            tmp_path,
            include_context_mode=True,
            context_mode_dir=bad_dir,
            context_mode_project_dir="/repo",
        )
    assert exc.value.error_code == "fcc_mcp_context_mode_dir"


def test_render_context_mode_requires_project_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_all_tools(monkeypatch)
    with pytest.raises(McpPreflightError) as exc:
        _render(
            tmp_path,
            include_context_mode=True,
            context_mode_dir=str(tmp_path / "ctx"),
            context_mode_project_dir="",
        )
    assert exc.value.error_code == "fcc_mcp_context_mode_config"


def test_self_index_servers_get_allowed_tool_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_all_tools(monkeypatch)
    data = _render(
        tmp_path,
        include_gitnexus=True,
        include_context_mode=True,
        context_mode_dir=str(tmp_path / "ctx"),
        context_mode_project_dir="/repo",
    )
    patterns = mcp_allowed_tool_patterns(data["mcpServers"])
    assert "mcp__gitnexus" in patterns
    assert "mcp__context-mode" in patterns
