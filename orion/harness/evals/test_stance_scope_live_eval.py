from __future__ import annotations

import subprocess
import sys

import httpx
import pytest

from orion.harness.evals import stance_scope_live_eval as module
from orion.harness.evals.stance_scope_live_eval import (
    HUNT_TRIPWIRE,
    parse_tool_steps,
    run_once,
    score_run,
)

CORR = "f924c7b9-1c82-40d8-a6a2-5acb2edffbb3"
OTHER = "00000000-0000-0000-0000-000000000000"


def _line(corr: str, step: int, tool: str) -> str:
    return (
        "[ORION-HARNESS-GOV] 2026-09-28 23:36:39,000 - INFO - orion.harness.grammar_publish - "
        f"harness_grammar_step_published corr={corr} channel=orion:grammar:event "
        f"step={step} tool={tool} event_id=e-{step}"
    )


def test_parse_tool_steps_keeps_only_this_turn_in_step_order_without_none() -> None:
    log = "\n".join(
        [
            _line(CORR, 9, "mcp__orion-introspect__reading_results"),
            _line(OTHER, 3, "Bash"),
            _line(CORR, 7, "ToolSearch"),
            _line(CORR, 8, "none"),
            _line(CORR, 24, "Agent"),
            "unrelated line",
        ]
    )
    assert parse_tool_steps(log, CORR) == [
        "ToolSearch",
        "mcp__orion-introspect__reading_results",
        "Agent",
    ]


def test_score_run_flags_the_incident_shape_as_a_hunt() -> None:
    tools = ["ToolSearch"] + ["mcp__orion-introspect__reading_results"] * 6 + ["Agent"] + ["Bash"] * 19 + ["Read"] * 5
    result = score_run(tools, finished=False, reply_text="")
    assert result["introspect_calls"] == 6
    assert result["discovery_calls"] == 1
    assert result["other_tool_calls"] == 25
    assert result["hunt"] is True
    assert result["passed"] is False


def test_score_run_passes_a_focused_finished_turn() -> None:
    tools = ["ToolSearch", "mcp__orion-introspect__reading_results", "mcp__orion-introspect__reading_results"]
    result = score_run(tools, finished=True, reply_text="I read three GPU pieces.")
    assert result["hunt"] is False
    assert result["passed"] is True


def test_score_run_fails_when_the_tool_was_never_called() -> None:
    result = score_run(["Bash"], finished=True, reply_text="From memory, GPUs are fast.")
    assert result["passed"] is False


def test_hunt_tripwire_is_exclusive() -> None:
    assert score_run(["Bash"] * HUNT_TRIPWIRE, finished=True, reply_text="x")["hunt"] is False
    assert score_run(["Bash"] * (HUNT_TRIPWIRE + 1), finished=True, reply_text="x")["hunt"] is True


def test_run_once_connect_error_skips_lookup_and_cancel(monkeypatch) -> None:
    def raise_connect(*_args, **_kwargs):
        raise httpx.ConnectError("connection refused")

    lookups: list[str] = []
    cancels: list[str] = []

    def mock_lookup(sid: str) -> str:
        lookups.append(sid)
        return "c-1"

    monkeypatch.setattr(module.httpx, "post", raise_connect)
    monkeypatch.setattr(module, "_corr_for_session", mock_lookup)
    monkeypatch.setattr(module, "_cancel", cancels.append)
    monkeypatch.setattr(module, "_governor_log", lambda _since: "")
    result = run_once("http://127.0.0.1:8080", 30.0)
    assert "ConnectError" in str(result["error"])
    assert result["corr_lookup"] is None
    assert result["hunt"] is None
    assert result["passed"] is False
    assert lookups == []
    assert cancels == []


def test_run_once_remote_protocol_error_looks_up_and_cancels(monkeypatch) -> None:
    def raise_protocol(*_args, **_kwargs):
        raise httpx.RemoteProtocolError(
            "peer closed connection", request=httpx.Request("POST", "http://127.0.0.1:8080/api/chat")
        )

    cancels: list[str] = []
    monkeypatch.setattr(module.httpx, "post", raise_protocol)
    monkeypatch.setattr(module, "_corr_for_session", lambda _sid: "c-1")
    monkeypatch.setattr(module, "_cancel", cancels.append)
    monkeypatch.setattr(module, "_governor_log", lambda _since: "")
    result = run_once("http://127.0.0.1:8080", 30.0)
    assert "RemoteProtocolError" in str(result["error"])
    assert result["corr_lookup"] == "ok"
    assert result["correlation_id"] == "c-1"
    assert cancels == ["c-1"]
    assert result["passed"] is False


def test_run_once_successful_turn_has_no_lookup(monkeypatch) -> None:
    class _Resp:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"correlation_id": "c-1", "llm_response": "I read three GPU pieces.", "type": "final"}

    monkeypatch.setattr(module.httpx, "post", lambda *_a, **_k: _Resp())
    monkeypatch.setattr(
        module,
        "_governor_log",
        lambda _since: _line("c-1", 1, "mcp__orion-introspect__reading_results"),
    )
    result = run_once("http://127.0.0.1:8080", 30.0)
    assert result["corr_lookup"] is None
    assert result["error"] is None
    assert result["hunt"] is False
    assert result["passed"] is True


def test_main_rejects_zero_runs(monkeypatch) -> None:
    def fail_run(*_args, **_kwargs):
        raise AssertionError("run_once must not be called")

    monkeypatch.setattr(module, "run_once", fail_run)
    monkeypatch.setattr(sys, "argv", ["stance_scope_live_eval", "--runs", "0"])
    with pytest.raises(SystemExit) as excinfo:
        module.main()
    assert excinfo.value.code == 2


def test_run_once_timeout_missing_corr(monkeypatch) -> None:
    def raise_timeout(*_args, **_kwargs):
        raise httpx.TimeoutException("timed out")

    cancel_called: list[str] = []

    def mock_cancel(corr: str) -> None:
        cancel_called.append(corr)

    monkeypatch.setattr(module.httpx, "post", raise_timeout)
    monkeypatch.setattr(module, "_corr_for_session", lambda _sid: None)
    monkeypatch.setattr(module, "_cancel", mock_cancel)
    monkeypatch.setattr(module, "_governor_log", lambda _since: "")
    result = run_once("http://127.0.0.1:8080", 30.0)
    assert result["corr_lookup"] == "missing"
    assert result["hunt"] is None
    assert result["passed"] is False
    assert cancel_called == []


def test_run_once_timeout_cancel_failure(monkeypatch) -> None:
    def raise_timeout(*_args, **_kwargs):
        raise httpx.TimeoutException("timed out")

    cancel_called: list[str] = []

    def mock_cancel(corr: str) -> None:
        cancel_called.append(corr)
        raise subprocess.CalledProcessError(1, "x")

    monkeypatch.setattr(module.httpx, "post", raise_timeout)
    monkeypatch.setattr(module, "_corr_for_session", lambda _sid: "c-1")
    monkeypatch.setattr(module, "_cancel", mock_cancel)
    monkeypatch.setattr(module, "_governor_log", lambda _since: "")
    result = run_once("http://127.0.0.1:8080", 30.0)
    assert "CalledProcessError" in str(result["error"])
    assert result["corr_lookup"] == "ok"
    assert cancel_called == ["c-1"]
