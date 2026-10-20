from __future__ import annotations

import numpy as np

from app.grading import grade


def test_perfect_forecast_has_full_skill_and_base_rate_has_none():
    hits = [True] * 20 + [False] * 80
    perfect = grade([1.0] * 20 + [0.0] * 80, hits)
    assert perfect["brier"] == 0 and perfect["skill"] == 1 and perfect["base_rate"] == 0.2
    climatology = grade([0.2] * 100, hits)
    assert abs(climatology["skill"]) < 1e-12


def test_a_forecast_that_runs_low_shows_in_the_calibration_bins():
    rng = np.random.default_rng(0)
    truth = rng.random(2000)
    hits = rng.random(2000) < truth
    g = grade(truth * 0.3, hits)          # says 30% of the real chance
    top = g["bins"][-1]
    assert top["observed"] > top["mean_p"] + 0.3 and g["skill"] < grade(truth, hits)["skill"]


def test_empty_input():
    assert grade([], [])["n"] == 0
