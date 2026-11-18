"""Orion's repo sub-agent reaches FCC turns through --agents, not project settings (#3650)."""

from __future__ import annotations

import json

import pytest

from orion.harness import fcc_motor as motor


def _argv(**kw):
    return motor.build_claude_argv(claude_bin="claude", model_id="m", prompt="p", correlation_id="c", **kw)


def test_normal_turn_lists_orion_repo_agent_without_project_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_FCC_REPO_AGENT_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_FCC_SETTING_SOURCES", raising=False)
    argv = _argv()
    agents = json.loads(argv[argv.index("--agents") + 1])
    assert list(agents) == ["orion-repo-agent"]
    # Project scope (repo CLAUDE.md, hooks, settings) stays off.
    assert argv[argv.index("--setting-sources") + 1] == "user,local"


def test_warm_pool_process_gets_the_same_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_FCC_REPO_AGENT_ENABLED", raising=False)
    argv = motor.build_claude_argv(claude_bin="claude", model_id="m", prompt=None, correlation_id="c")
    assert "orion-repo-agent" in json.loads(argv[argv.index("--agents") + 1])


def test_reading_only_turn_gets_no_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_FCC_REPO_AGENT_ENABLED", raising=False)
    assert "--agents" not in _argv(reading_only=True)


@pytest.mark.parametrize("value", ["false", "0", "off"])
def test_flag_off_drops_the_agent(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HARNESS_FCC_REPO_AGENT_ENABLED", value)
    assert "--agents" not in _argv()
