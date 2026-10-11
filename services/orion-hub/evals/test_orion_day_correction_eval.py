"""Pytest entry for evals/run_orion_day_correction_eval.py (fixture letter and priors, offline)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).with_name("run_orion_day_correction_eval.py")


def _eval():
    spec = importlib.util.spec_from_file_location("run_orion_day_correction_eval", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_correction_reaches_one_run_and_the_untouched_prior_is_not_counted():
    assert _eval().main([]) == 0
