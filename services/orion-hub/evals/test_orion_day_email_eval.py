"""Pytest entry for evals/run_orion_day_email_eval.py (fixture day, offline)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).with_name("run_orion_day_email_eval.py")


def _eval():
    spec = importlib.util.spec_from_file_location("run_orion_day_email_eval", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fixture_letter_shows_every_material_item_in_full():
    assert _eval().main([]) == 0
