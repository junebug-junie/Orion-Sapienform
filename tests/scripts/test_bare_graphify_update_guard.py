from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hooks"))

from bare_graphify_update_guard import (  # noqa: E402
    _escape_hatch_set,
    _evaluate,
    _statement_is_bare_graphify_update,
)


@pytest.mark.parametrize(
    "statement,expect_blocked",
    [
        ("graphify update .", True),
        ("graphify update", True),
        ("graphify update services/", True),
        ("/home/athena/.local/bin/graphify update .", True),
        ("scripts/safe_graphify_update.sh", False),
        ("./scripts/safe_graphify_update.sh", False),
        ("bash scripts/safe_graphify_update.sh", False),
        ("graphify query 'how does X work'", False),
        ("graphify path A B", False),
        ("graphify explain foo", False),
        ("graphify prs --worktrees", False),
        ("graphify query update", False),
        ("graphify query how to update the graph", False),
        ("graphify explain update", False),
        ("pytest services/foo/tests -q", False),
        ("git status", False),
    ],
)
def test_statement_is_bare_graphify_update(statement: str, expect_blocked: bool) -> None:
    result = _statement_is_bare_graphify_update(statement)
    assert result == expect_blocked, f"{statement!r} -> {result!r}"


def test_quoted_graphify_update_is_not_detected() -> None:
    assert _evaluate('echo "graphify update ."') is None


def test_escape_hatch_prefix() -> None:
    assert _escape_hatch_set("ORION_ALLOW_GRAPH_SHRINK=1 graphify update .")


def test_escape_hatch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_ALLOW_GRAPH_SHRINK", "1")
    assert _escape_hatch_set("graphify update .")


def test_evaluate_blocks_bare_update() -> None:
    assert _evaluate("graphify update .") is not None


def test_evaluate_allows_safe_wrapper() -> None:
    assert _evaluate("scripts/safe_graphify_update.sh") is None


def test_evaluate_allows_escape_hatch_on_same_statement() -> None:
    assert _evaluate("ORION_ALLOW_GRAPH_SHRINK=1 graphify update .") is None


def test_evaluate_blocks_second_statement_in_chain() -> None:
    assert _evaluate("git status && graphify update .") is not None


def test_evaluate_allows_first_statement_if_only_safe_wrapper() -> None:
    assert _evaluate("scripts/safe_graphify_update.sh && pytest -q") is None


def test_evaluate_escape_hatch_scoped_to_its_own_statement() -> None:
    """Escape hatch on statement 2 must not authorize statement 1."""
    assert _evaluate("graphify update .; ORION_ALLOW_GRAPH_SHRINK=1 echo ok") is not None


def _run_main(payload: dict, *, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "hooks" / "bare_graphify_update_guard.py")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )


def test_main_blocks_graphify_update() -> None:
    proc = _run_main({"tool_input": {"command": "graphify update ."}})
    decision = json.loads(proc.stdout)
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "graphify-safety" in decision["hookSpecificOutput"]["permissionDecisionReason"]


def test_main_allows_safe_wrapper() -> None:
    proc = _run_main({"tool_input": {"command": "scripts/safe_graphify_update.sh"}})
    assert proc.stdout == ""


def test_evaluate_allows_graphify_query_with_update_word() -> None:
    assert _evaluate("graphify query how to update the graph") is None


def test_evaluate_escape_hatch_env_allows_bare_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_ALLOW_GRAPH_SHRINK", "1")
    assert _evaluate("graphify update .") is None
