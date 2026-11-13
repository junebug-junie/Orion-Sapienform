"""Pytest entry for evals/run_curiosity_focus_eval.py (queue -> take -> kickoff -> outcome -> read back, offline)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).with_name("run_curiosity_focus_eval.py")


def _eval():
    spec = importlib.util.spec_from_file_location("run_curiosity_focus_eval", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_queued_focus_reaches_one_run_and_reads_back_truthfully():
    assert _eval().main([]) == 0
