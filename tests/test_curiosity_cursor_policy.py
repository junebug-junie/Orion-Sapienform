"""Read-only Cursor Agent CLI argv policy (orion/curiosity/cursor_policy.py).

Moved here from services/orion-curiosity-peer/tests when that service dropped
Cursor (#3397). The policy still guards the grading-side Cursor caller in
orion/curiosity/supervisor.py, so its safety tests stay.
"""

from __future__ import annotations

import pytest

from orion.curiosity.cursor_policy import (
    assert_read_only_cli_argv,
    build_cursor_agent_argv,
)


def test_build_argv_is_ask_print_mode() -> None:
    argv = build_cursor_agent_argv(
        agent_bin="/opt/cursor-agent/versions/x/cursor-agent",
        prompt="sealed",
        workspace="/repo",
        model="composer-2.5",
    )
    assert_read_only_cli_argv(argv)
    assert argv[0].endswith("cursor-agent")
    assert "-p" in argv
    assert argv[argv.index("--mode") + 1] == "ask"
    assert "--workspace" in argv
    assert argv[argv.index("--workspace") + 1] == "/repo"
    assert "--trust" in argv
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "text"
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "composer-2.5"
    assert argv[-1] == "sealed"
    for bad in ("--force", "--yolo", "--approve-mcps", "--plan"):
        assert bad not in argv


def test_policy_rejects_force_yolo_approve_mcps() -> None:
    base = build_cursor_agent_argv(
        agent_bin="cursor-agent",
        prompt="x",
        workspace="/repo",
    )
    for bad in ("--force", "-f", "--yolo", "--approve-mcps"):
        with pytest.raises(ValueError) as exc:
            assert_read_only_cli_argv([*base[:-1], bad, base[-1]])
        assert bad in str(exc.value) or "forbidden" in str(exc.value).lower()


def test_policy_rejects_plan_mode() -> None:
    argv = [
        "cursor-agent",
        "-p",
        "--mode",
        "plan",
        "--workspace",
        "/repo",
        "--trust",
        "prompt",
    ]
    with pytest.raises(ValueError) as exc:
        assert_read_only_cli_argv(argv)
    assert "ask" in str(exc.value).lower()


def test_policy_rejects_missing_print_or_trust_or_workspace() -> None:
    good = build_cursor_agent_argv(
        agent_bin="cursor-agent", prompt="x", workspace="/repo"
    )
    no_print = [t for t in good if t not in ("-p", "--print")]
    with pytest.raises(ValueError):
        assert_read_only_cli_argv(no_print)

    no_trust = [t for t in good if t != "--trust"]
    with pytest.raises(ValueError):
        assert_read_only_cli_argv(no_trust)

    # Drop --workspace and its value.
    idx = good.index("--workspace")
    no_ws = good[:idx] + good[idx + 2 :]
    with pytest.raises(ValueError):
        assert_read_only_cli_argv(no_ws)
