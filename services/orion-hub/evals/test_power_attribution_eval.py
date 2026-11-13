"""Run the heat-archaeology fixture eval as a test so CI gates on it."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def test_power_attribution_fixture_eval_passes():
    path = Path(__file__).with_name("run_power_attribution_eval.py")
    spec = importlib.util.spec_from_file_location("run_power_attribution_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scores = mod.run()
    assert scores["pass"], scores
