from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.claude_runner import build_argv, run_claude_once  # noqa: E402


def test_build_argv_shape():
    argv = build_argv(
        "prompt text",
        claude_bin="claude",
        model="claude-sonnet-5",
        effort="medium",
        setting_sources_env_key="SELF_STUDY_ENRICHMENT_SETTING_SOURCES",
    )
    assert argv[0] == "claude"
    assert "-p" in argv
    assert "prompt text" in argv
    assert "--output-format" in argv and argv[argv.index("--output-format") + 1] == "json"
    assert "--model" in argv and argv[argv.index("--model") + 1] == "claude-sonnet-5"
    assert "--tools" in argv and argv[argv.index("--tools") + 1] == ""
    assert "--effort" in argv and argv[argv.index("--effort") + 1] == "medium"


def test_build_argv_no_tool_use_regardless_of_effort():
    argv = build_argv(
        "p",
        claude_bin="claude",
        model="claude-sonnet-5",
        effort="",
        setting_sources_env_key="SELF_STUDY_ENRICHMENT_SETTING_SOURCES",
    )
    assert "--effort" not in argv
    assert "--tools" in argv


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str, stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_claude_once_success_parses_result_field():
    fake_stdout = json.dumps({"result": "This cluster does X because Y."})
    with patch("app.claude_runner.subprocess.run", return_value=_FakeCompletedProcess(0, fake_stdout)):
        result = run_claude_once(
            "prompt",
            claude_bin="claude",
            model="claude-sonnet-5",
            effort="medium",
            setting_sources_env_key="SELF_STUDY_ENRICHMENT_SETTING_SOURCES",
            timeout_sec=5,
        )
    assert result.ok is True
    assert result.text == "This cluster does X because Y."


def test_run_claude_once_nonzero_exit_is_failure():
    with patch("app.claude_runner.subprocess.run", return_value=_FakeCompletedProcess(1, "", "boom")):
        result = run_claude_once(
            "prompt",
            claude_bin="claude",
            model="claude-sonnet-5",
            effort="medium",
            setting_sources_env_key="SELF_STUDY_ENRICHMENT_SETTING_SOURCES",
            timeout_sec=5,
        )
    assert result.ok is False
    assert "boom" in (result.error or "")


def test_run_claude_once_empty_result_is_failure_not_success():
    # No empty-shell cognition: raw_len=0 must never be treated as success.
    fake_stdout = json.dumps({"result": ""})
    with patch("app.claude_runner.subprocess.run", return_value=_FakeCompletedProcess(0, fake_stdout)):
        result = run_claude_once(
            "prompt",
            claude_bin="claude",
            model="claude-sonnet-5",
            effort="medium",
            setting_sources_env_key="SELF_STUDY_ENRICHMENT_SETTING_SOURCES",
            timeout_sec=5,
        )
    assert result.ok is False


def test_run_claude_once_timeout_is_failure():
    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=5)

    with patch("app.claude_runner.subprocess.run", side_effect=_raise):
        result = run_claude_once(
            "prompt",
            claude_bin="claude",
            model="claude-sonnet-5",
            effort="medium",
            setting_sources_env_key="SELF_STUDY_ENRICHMENT_SETTING_SOURCES",
            timeout_sec=5,
        )
    assert result.ok is False
    assert "timeout" in (result.error or "")


def test_run_claude_once_unparseable_json_is_failure():
    with patch("app.claude_runner.subprocess.run", return_value=_FakeCompletedProcess(0, "not json")):
        result = run_claude_once(
            "prompt",
            claude_bin="claude",
            model="claude-sonnet-5",
            effort="medium",
            setting_sources_env_key="SELF_STUDY_ENRICHMENT_SETTING_SOURCES",
            timeout_sec=5,
        )
    assert result.ok is False
