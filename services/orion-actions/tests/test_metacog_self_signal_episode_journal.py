"""Prose once per self_signal_episode: orion-actions journals the open row like
every metacog trigger, but not the close row (spec 2026-10-07 section D)."""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
for p in (SERVICE_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.main import should_journal_metacog_trigger  # noqa: E402
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1  # noqa: E402


def _t(kind: str, phase: str | None) -> MetacogTriggerV1:
    return MetacogTriggerV1(trigger_kind=kind, reason="r", upstream={"phase": phase} if phase else {})


def test_close_row_is_not_journaled():
    assert not should_journal_metacog_trigger(_t("self_signal_episode", "close"))


def test_open_row_and_other_kinds_are_journaled():
    assert should_journal_metacog_trigger(_t("self_signal_episode", "open"))
    assert should_journal_metacog_trigger(_t("transport", "close"))
    assert should_journal_metacog_trigger(_t("telemetry_anomaly", None))


def test_handler_consults_the_gate_before_dispatching():
    tree = ast.parse(Path(SERVICE_DIR, "app", "main.py").read_text())
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_handle_journal_metacog"
    )
    calls = [
        getattr(n.func, "id", None) for n in ast.walk(handler) if isinstance(n, ast.Call)
    ]
    assert calls.index("should_journal_metacog_trigger") < calls.index("_dispatch_journal")
