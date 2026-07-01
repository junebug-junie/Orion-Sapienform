"""Tests for the agent_repl mode + runner dispatch."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

CTX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(CTX_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def test_agent_repl_is_a_valid_mode():
    from orion.schemas.context_exec import ContextExecRequestV1

    req = ContextExecRequestV1(text="what does orion-hub do?", mode="agent_repl")
    assert req.mode == "agent_repl"
