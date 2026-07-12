from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List

import pytest

from orion.harness import fcc_motor as motor


class _FakeStream:
    def __init__(self, lines: List[bytes]) -> None:
        self._lines = list(lines)
        self._idx = 0

    async def readline(self) -> bytes:
        if self._idx >= len(self._lines):
            return b""
        line = self._lines[self._idx]
        self._idx += 1
        await asyncio.sleep(0)
        return line


class _BlockingAfterLinesStream(_FakeStream):
    async def readline(self) -> bytes:
        if self._idx >= len(self._lines):
            await asyncio.sleep(3600)
            return b""
        return await super().readline()


class _FakeProc:
    def __init__(
        self,
        stdout_lines: List[str],
        returncode: int = 0,
        *,
        block_after_stdout: bool = False,
    ) -> None:
        stream_cls = _BlockingAfterLinesStream if block_after_stdout else _FakeStream
        self.stdout = stream_cls([ln.encode("utf-8") + b"\n" for ln in stdout_lines])
        self.stderr = _FakeStream([])
        self.returncode = returncode
        self.killed = False

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


def _fake_fcc_env(_path: Any) -> dict[str, str]:
    return {"MODEL_HAIKU": "claude-haiku-test"}


def test_normalize_fcc_model_id_forces_no_thinking_for_llamacpp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_FCC_FORCE_NO_THINKING_MODEL", raising=False)

    assert (
        motor.normalize_fcc_model_id_for_claude("llamacpp/chat")
        == "claude-3-freecc-no-thinking/llamacpp/chat"
    )
    assert (
        motor.normalize_fcc_model_id_for_claude("anthropic/llamacpp/chat")
        == "claude-3-freecc-no-thinking/llamacpp/chat"
    )
    assert (
        motor.normalize_fcc_model_id_for_claude(
            "claude-3-freecc-no-thinking/llamacpp/chat"
        )
        == "claude-3-freecc-no-thinking/llamacpp/chat"
    )


def test_normalize_fcc_model_id_can_disable_no_thinking_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_FCC_FORCE_NO_THINKING_MODEL", "false")

    assert motor.normalize_fcc_model_id_for_claude("llamacpp/chat") == "llamacpp/chat"


def test_stream_idle_timeout_defaults_below_turn_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_FCC_STREAM_IDLE_TIMEOUT_SEC", raising=False)
    assert motor._stream_idle_timeout_sec(900.0) == motor.DEFAULT_STREAM_IDLE_TIMEOUT_SEC


def test_stream_idle_timeout_can_be_disabled_to_turn_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_FCC_STREAM_IDLE_TIMEOUT_SEC", "0")
    assert motor._stream_idle_timeout_sec(900.0) == 900.0


@pytest.mark.asyncio
async def test_run_fcc_turn_surfaces_stream_idle_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _FakeProc(
        [
            '{"type":"system","subtype":"init"}',
            '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"ToolSearch"}]}}',
        ],
        block_after_stdout=True,
    )

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **k: None)
    monkeypatch.setenv("HARNESS_FCC_STREAM_IDLE_TIMEOUT_SEC", "0.01")

    events = []
    async for ev in motor.run_fcc_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-idle",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        events.append(ev)

    assert [ev["type"] for ev in events] == ["step", "step", "error"]
    assert events[-1]["error_code"] == "fcc_idle_timeout"
    assert events[-1]["steps_seen"] == 2
    assert events[-1]["last_event_type"] == "assistant"
    assert events[-1]["last_tool"] == "ToolSearch"
    assert proc.killed is True


@pytest.mark.asyncio
async def test_run_fcc_turn_enables_and_accumulates_partial_stream_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_argv: list = []
    proc = _FakeProc(
        [
            '{"type":"system","subtype":"init"}',
            '{"type":"stream_event","event":{"type":"message_start","message":{"id":"m1"}}}',
            '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}}',
            '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}}',
            '{"type":"stream_event","event":{"type":"content_block_stop","index":0}}',
            '{"type":"stream_event","event":{"type":"message_stop"}}',
        ]
    )

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured_argv.extend(args)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **k: None)
    monkeypatch.setenv("HARNESS_FCC_PARTIAL_PROGRESS_INTERVAL_SEC", "0")

    events = []
    async for ev in motor.run_fcc_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-partials",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        events.append(ev)

    assert "--include-partial-messages" in captured_argv
    assert events[-1]["type"] == "final"
    assert events[-1]["llm_response"] == "Hello"


@pytest.mark.asyncio
async def test_run_fcc_turn_partial_stream_timeout_is_unsafe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _FakeProc(
        [
            '{"type":"system","subtype":"init"}',
            '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"loop"}}}',
            '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"ing"}}}',
        ]
    )
    clock = {"value": 0.0}

    def fake_monotonic() -> float:
        clock["value"] += 2.0
        return clock["value"]

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **k: None)
    monkeypatch.setenv("HARNESS_FCC_PARTIAL_STREAM_TIMEOUT_SEC", "0.01")
    monkeypatch.setenv("HARNESS_FCC_PARTIAL_PROGRESS_INTERVAL_SEC", "0")

    events = []
    async for ev in motor.run_fcc_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-partial-timeout",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        events.append(ev)

    assert events[-1]["type"] == "error"
    assert events[-1]["error_code"] == "fcc_partial_stream_timeout"
    assert events[-1]["partial_unsafe"] is True
    assert events[-1]["llm_response"] == "looping"
    assert proc.killed is True


@pytest.mark.asyncio
async def test_run_fcc_turn_adds_mcp_config_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_argv: list = []

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured_argv.extend(args)
        return _FakeProc([
            '{"type":"result","result":"Done.","session_id":"s1"}',
        ])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **k: Path("/tmp/fake-mcp.json"))
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    fake_cfg = Path("/tmp/fake-mcp.json")
    fake_cfg.write_text(
        '{"mcpServers":{"github":{},"firecrawl":{}}}',
        encoding="utf-8",
    )

    async for _ in motor.run_fcc_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-mcp",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        pass

    assert "--mcp-config" in captured_argv
    assert "/tmp/fake-mcp.json" in [str(x) for x in captured_argv]
    assert "--allowedTools" in captured_argv
    idx = captured_argv.index("--allowedTools")
    assert captured_argv[idx + 1] == "mcp__github"
    assert captured_argv[idx + 2] == "mcp__firecrawl"
    dis_idx = captured_argv.index("--disallowedTools")
    assert captured_argv[dis_idx + 1] == "Bash(gh *)"


@pytest.mark.asyncio
async def test_run_fcc_turn_omits_mcp_config_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_argv: list = []

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured_argv.extend(args)
        return _FakeProc([
            '{"type":"result","result":"Done.","session_id":"s1"}',
        ])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.delenv("HARNESS_FCC_MCP_ENABLED", raising=False)

    async for _ in motor.run_fcc_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-no-mcp",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        pass

    assert "--mcp-config" not in captured_argv
    assert "--allowedTools" not in captured_argv


@pytest.mark.asyncio
async def test_run_fcc_turn_hook_mode_allows_context_mode_plugin_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_argv: list = []

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured_argv.extend(args)
        return _FakeProc([
            '{"type":"result","result":"Done.","session_id":"s1"}',
        ])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.delenv("HARNESS_FCC_GITNEXUS_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_FCC_CONTEXT_MODE_ENABLED", raising=False)
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "true")
    fake_cfg = tmp_path / "hooks-mcp.json"
    fake_cfg.write_text(
        '{"mcpServers":{"github":{},"firecrawl":{}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **k: fake_cfg)

    async for _ in motor.run_fcc_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-hooks-argv",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        pass

    assert "--allowedTools" in captured_argv
    idx = captured_argv.index("--allowedTools")
    assert captured_argv[idx + 1] == "mcp__github"
    assert captured_argv[idx + 2] == "mcp__firecrawl"
    assert captured_argv[idx + 3] == "mcp__plugin_context-mode_context-mode"


@pytest.mark.asyncio
async def test_run_fcc_turn_surfaces_mcp_preflight_error_code(monkeypatch: pytest.MonkeyPatch) -> None:
    from orion.fcc.mcp_config import McpPreflightError

    def _raise_preflight(**_kwargs: Any) -> Path:
        raise McpPreflightError(error_code="fcc_mcp_github_missing", message="Missing GITHUB_PAT in FCC env")

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        raise AssertionError("should not spawn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", _raise_preflight)
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")

    events = []
    async for ev in motor.run_fcc_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-preflight",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        events.append(ev)

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["error_code"] == "fcc_mcp_github_missing"


def test_maybe_render_mcp_config_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_FCC_MCP_ENABLED", raising=False)
    assert motor._maybe_render_mcp_config(correlation_id="corr-off") is None


def test_maybe_render_mcp_config_wires_orion_fcc_render(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import orion.fcc.mcp_config as mcp_config

    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.delenv("HARNESS_AITOWN_ENABLED", raising=False)
    monkeypatch.setattr(
        motor,
        "load_fcc_env",
        lambda _p: {"GITHUB_PAT": "ghp_motor_int", "FIRECRAWL_API_KEY": "fc_motor_int"},
    )
    monkeypatch.setattr(motor, "expand_env_path", lambda _p: Path("/fake/.fcc/.env"))
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    path = motor._maybe_render_mcp_config(correlation_id="corr-motor-render")
    try:
        assert path is not None
        assert path.name == "corr-motor-render.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["mcpServers"]["github"]["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_motor_int"
        assert data["mcpServers"]["firecrawl"]["env"]["FIRECRAWL_API_KEY"] == "fc_motor_int"
        assert "orion-aitown" not in data["mcpServers"]
    finally:
        mcp_config.cleanup_mcp_config(path)


def test_maybe_render_mcp_config_passes_github_toolsets_from_fcc_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import orion.fcc.mcp_config as mcp_config

    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.delenv("HARNESS_AITOWN_ENABLED", raising=False)
    monkeypatch.setattr(
        motor,
        "load_fcc_env",
        lambda _p: {
            "GITHUB_PAT": "ghp_motor_ts",
            "FIRECRAWL_API_KEY": "fc_motor_ts",
            "GITHUB_TOOLSETS": "repos,pull_requests,issues",
            "GITHUB_READ_ONLY": "0",
        },
    )
    monkeypatch.setattr(motor, "expand_env_path", lambda _p: Path("/fake/.fcc/.env"))
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    path = motor._maybe_render_mcp_config(correlation_id="corr-motor-toolsets")
    try:
        assert path is not None
        github_env = json.loads(path.read_text(encoding="utf-8"))["mcpServers"]["github"]["env"]
        assert github_env["GITHUB_TOOLSETS"] == "repos,pull_requests,issues"
        assert github_env["GITHUB_READ_ONLY"] == "0"
    finally:
        mcp_config.cleanup_mcp_config(path)


def test_maybe_render_mcp_config_defaults_github_toolsets_when_env_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import orion.fcc.mcp_config as mcp_config

    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.delenv("HARNESS_AITOWN_ENABLED", raising=False)
    monkeypatch.setattr(
        motor,
        "load_fcc_env",
        lambda _p: {"GITHUB_PAT": "ghp_motor_def", "FIRECRAWL_API_KEY": "fc_motor_def"},
    )
    monkeypatch.setattr(motor, "expand_env_path", lambda _p: Path("/fake/.fcc/.env"))
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    path = motor._maybe_render_mcp_config(correlation_id="corr-motor-toolset-default")
    try:
        assert path is not None
        github_env = json.loads(path.read_text(encoding="utf-8"))["mcpServers"]["github"]["env"]
        assert github_env["GITHUB_TOOLSETS"] == "repos,pull_requests"
        assert github_env["GITHUB_READ_ONLY"] == "1"
    finally:
        mcp_config.cleanup_mcp_config(path)


def test_maybe_render_mcp_config_wires_self_index_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    import orion.fcc.mcp_config as mcp_config

    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.delenv("HARNESS_AITOWN_ENABLED", raising=False)
    monkeypatch.setenv("HARNESS_FCC_GITNEXUS_ENABLED", "true")
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_ENABLED", "true")
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_DIR", str(tmp_path / "ctx-data"))
    monkeypatch.setenv("HARNESS_FCC_WORKSPACE", "/mnt/scripts/Orion-Sapienform")
    monkeypatch.setattr(
        motor,
        "load_fcc_env",
        lambda _p: {"GITHUB_PAT": "ghp_selfidx", "FIRECRAWL_API_KEY": "fc_selfidx"},
    )
    monkeypatch.setattr(motor, "expand_env_path", lambda _p: Path("/fake/.fcc/.env"))
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    path = motor._maybe_render_mcp_config(correlation_id="corr-self-index")
    try:
        assert path is not None
        servers = json.loads(path.read_text(encoding="utf-8"))["mcpServers"]
        assert servers["gitnexus"] == {"type": "stdio", "command": "gitnexus", "args": ["mcp"]}
        cm_env = servers["context-mode"]["env"]
        assert cm_env["CONTEXT_MODE_PROJECT_DIR"] == "/mnt/scripts/Orion-Sapienform"
        assert cm_env["CONTEXT_MODE_DIR"] == str(tmp_path / "ctx-data")
    finally:
        mcp_config.cleanup_mcp_config(path)


def test_maybe_render_mcp_config_hook_mode_skips_standalone_context_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    import orion.fcc.mcp_config as mcp_config

    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.delenv("HARNESS_AITOWN_ENABLED", raising=False)
    monkeypatch.setenv("HARNESS_FCC_GITNEXUS_ENABLED", "true")
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_ENABLED", "true")
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "true")
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_DIR", str(tmp_path / "ctx-data"))
    monkeypatch.setenv("HARNESS_FCC_WORKSPACE", "/mnt/scripts/Orion-Sapienform")
    monkeypatch.setattr(
        motor,
        "load_fcc_env",
        lambda _p: {"GITHUB_PAT": "ghp_hooks", "FIRECRAWL_API_KEY": "fc_hooks"},
    )
    monkeypatch.setattr(motor, "expand_env_path", lambda _p: Path("/fake/.fcc/.env"))
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    path = motor._maybe_render_mcp_config(correlation_id="corr-hooks-mode")
    try:
        assert path is not None
        servers = json.loads(path.read_text(encoding="utf-8"))["mcpServers"]
        assert "context-mode" not in servers
        assert "gitnexus" in servers
    finally:
        mcp_config.cleanup_mcp_config(path)


def test_maybe_render_mcp_config_self_index_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import orion.fcc.mcp_config as mcp_config

    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.delenv("HARNESS_AITOWN_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_FCC_GITNEXUS_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_FCC_CONTEXT_MODE_ENABLED", raising=False)
    monkeypatch.setattr(
        motor,
        "load_fcc_env",
        lambda _p: {"GITHUB_PAT": "ghp_defaults", "FIRECRAWL_API_KEY": "fc_defaults"},
    )
    monkeypatch.setattr(motor, "expand_env_path", lambda _p: Path("/fake/.fcc/.env"))
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    path = motor._maybe_render_mcp_config(correlation_id="corr-self-index-off")
    try:
        assert path is not None
        servers = json.loads(path.read_text(encoding="utf-8"))["mcpServers"]
        assert "gitnexus" not in servers
        assert "context-mode" not in servers
    finally:
        mcp_config.cleanup_mcp_config(path)


def test_harness_aitown_env_overrides_convex_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_AITOWN_CONVEX_URL", "http://host.docker.internal:3210")
    base = {"AITOWN_CONVEX_URL": "http://127.0.0.1:3210", "AITOWN_ADMIN_KEY": "k"}
    merged = motor._harness_aitown_env(base)
    assert merged["AITOWN_CONVEX_URL"] == "http://host.docker.internal:3210"
    assert merged["AITOWN_ADMIN_KEY"] == "k"


def test_build_subprocess_env_sets_context_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_FCC_MAX_CONTEXT_TOKENS", "65536")
    monkeypatch.setenv("HARNESS_FCC_FILE_READ_MAX_TOKENS", "8192")
    monkeypatch.setenv("HARNESS_FCC_AUTOCOMPACT_PCT_OVERRIDE", "70")
    env = motor._build_subprocess_env(fcc_server_url="http://127.0.0.1:8082", auth_token="tok")
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"
    assert env["CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"] == "8192"
    assert env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "70"


def test_build_subprocess_env_hook_mode_sets_context_mode_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.delenv("HARNESS_FCC_GITNEXUS_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_FCC_CONTEXT_MODE_ENABLED", raising=False)
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "true")
    monkeypatch.setenv("HARNESS_FCC_WORKSPACE", "/mnt/scripts/Orion-Sapienform")
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_DIR", str(tmp_path / "ctx-data"))
    env = motor._build_subprocess_env(fcc_server_url="http://127.0.0.1:8082", auth_token="tok")
    assert env["CONTEXT_MODE_PLATFORM"] == "claude-code"
    assert env["CONTEXT_MODE_PROJECT_DIR"] == "/mnt/scripts/Orion-Sapienform"
    assert env["CONTEXT_MODE_DIR"] == str(tmp_path / "ctx-data")


def test_build_subprocess_env_hook_mode_uses_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "true")
    monkeypatch.delenv("HARNESS_FCC_WORKSPACE", raising=False)
    monkeypatch.delenv("HARNESS_FCC_CONTEXT_MODE_DIR", raising=False)
    env = motor._build_subprocess_env(fcc_server_url="http://127.0.0.1:8082", auth_token="tok")
    assert env["CONTEXT_MODE_PLATFORM"] == "claude-code"
    assert env["CONTEXT_MODE_PROJECT_DIR"] == motor.os.getcwd()
    assert env["CONTEXT_MODE_DIR"] == "/var/lib/orion/context-mode"


def test_build_subprocess_env_no_context_mode_keys_without_hook_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", raising=False)
    monkeypatch.delenv("CONTEXT_MODE_PLATFORM", raising=False)
    monkeypatch.delenv("CONTEXT_MODE_PROJECT_DIR", raising=False)
    monkeypatch.delenv("CONTEXT_MODE_DIR", raising=False)
    env = motor._build_subprocess_env(fcc_server_url="http://127.0.0.1:8082", auth_token="tok")
    assert "CONTEXT_MODE_PLATFORM" not in env
    assert "CONTEXT_MODE_PROJECT_DIR" not in env
    assert "CONTEXT_MODE_DIR" not in env


def test_build_subprocess_env_enables_tool_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENABLE_TOOL_SEARCH", raising=False)
    env = motor._build_subprocess_env(fcc_server_url="http://127.0.0.1:8082", auth_token="tok")
    assert env["ENABLE_TOOL_SEARCH"] == "true"


def test_build_subprocess_env_preserves_tool_search_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "false")
    env = motor._build_subprocess_env(fcc_server_url="http://127.0.0.1:8082", auth_token="tok")
    assert env["ENABLE_TOOL_SEARCH"] == "false"


@pytest.mark.asyncio
async def test_run_fcc_turn_root_uses_dont_ask_permission_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_argv: list = []

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured_argv.extend(args)
        return _FakeProc([
            '{"type":"result","result":"Done.","session_id":"s1"}',
        ])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.setattr(motor.os, "geteuid", lambda: 0)
    monkeypatch.setenv("HARNESS_FCC_SKIP_PERMISSIONS", "true")

    async for _ in motor.run_fcc_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-dontask",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        pass

    assert "--permission-mode" in captured_argv
    assert "dontAsk" in captured_argv
    assert "--dangerously-skip-permissions" not in captured_argv


def test_should_skip_claude_permissions_env_false_overrides_non_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(motor.os, "geteuid", lambda: 1000)
    monkeypatch.setenv("HARNESS_FCC_SKIP_PERMISSIONS", "false")
    assert motor._should_skip_claude_permissions() is False
