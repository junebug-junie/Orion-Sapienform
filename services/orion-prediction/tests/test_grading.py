from __future__ import annotations

import numpy as np

from orion.prediction.grading import grade


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


# --- sorting_score (within-group AUC) ---------------------------------------------------------------
import pytest  # noqa: E402

from orion.prediction.grading import sorting_score  # noqa: E402


def test_sorting_score_counts_pairs_and_ties():
    # pairs (hit, miss): (0.9,0.1) win, (0.9,0.5) win, (0.5,0.1) win, (0.5,0.5) tie -> 3.5 / 4
    out = sorting_score([0.9, 0.5, 0.1, 0.5], [True, True, False, False])
    assert out["pairs"] == 4 and out["auc"] == pytest.approx(0.875)


def test_knowing_the_group_earns_nothing_inside_it():
    # Group a always happens and gets 0.9, group b never and gets 0.1: perfect across groups,
    # and inside each group there is no pair to sort, so there is no score at all.
    out = sorting_score([0.9] * 10 + [0.1] * 10, [True] * 10 + [False] * 10, ["a"] * 10 + ["b"] * 10)
    assert out == {"auc": None, "pairs": 0, "p_value": None}
    assert sorting_score([0.9] * 10 + [0.1] * 10, [True] * 10 + [False] * 10)["auc"] == 1.0


def test_constant_probability_is_a_coin_flip_and_not_significant():
    out = sorting_score([0.4] * 60, [True, False] * 30)
    assert out["auc"] == 0.5 and out["p_value"] == pytest.approx(0.5)


def test_real_sorting_is_significant_and_a_base_rate_shift_does_not_move_it():
    import random
    rng = random.Random(1)
    p = [rng.random() for _ in range(300)]
    hits = [rng.random() < x for x in p]
    out = sorting_score(p, hits)
    assert out["auc"] > 0.7 and out["p_value"] < 1e-6
    # Every probability halved (badly calibrated after a base-rate shift): same order, same score.
    assert sorting_score([x / 2 for x in p], hits)["auc"] == pytest.approx(out["auc"])
