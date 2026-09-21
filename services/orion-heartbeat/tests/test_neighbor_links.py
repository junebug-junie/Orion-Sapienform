"""Pre-registered fail criteria for neighbor-link windows.

docs/research/preregistration/2026-09-20-heartbeat-neighbor-links.md
"""
from __future__ import annotations

from app.substrate.neighbor_links import (
    classify_window,
    classify_window_firing,
    decide_probe,
    decide_probe_v2,
    pair_threshold,
    score_pair,
    window_medians,
)


def test_median_split_puts_high_high_in_cofire() -> None:
    med_a, med_b, med_o = window_medians([1, 3, 5], [2, 4, 6], [0, 10, 20])
    assert med_a == 3
    assert med_b == 4
    assert med_o == 10
    assert classify_window(5, 6, 0, median_a=med_a, median_b=med_b, median_other=med_o) == "cofire"


def test_elsewhere_needs_both_low_and_other_high() -> None:
    assert (
        classify_window(1, 1, 20, median_a=3, median_b=4, median_other=10) == "elsewhere"
    )
    assert classify_window(1, 1, 5, median_a=3, median_b=4, median_other=10) == "neither"
    assert classify_window(1, 8, 20, median_a=3, median_b=4, median_other=10) == "neither"


def test_unverified_below_ten_windows() -> None:
    score = score_pair(
        organ_a="orion-biometrics",
        organ_b="orion-cortex-exec",
        site_a=1,
        site_b=2,
        i_cofire=[0.2] * 9,
        i_elsewhere=[0.1] * 20,
    )
    assert score.unverified is True
    assert score.holds is False


def test_absolute_threshold_when_scale_is_large() -> None:
    assert pair_threshold(0.20, 0.10) == 0.05
    score = score_pair(
        organ_a="orion-biometrics",
        organ_b="orion-cortex-exec",
        site_a=1,
        site_b=2,
        i_cofire=[0.20] * 10,
        i_elsewhere=[0.10] * 10,
    )
    assert score.unverified is False
    assert score.holds is True
    assert abs(score.delta - 0.10) < 1e-12


def test_absolute_fail_when_delta_tiny_on_large_scale() -> None:
    score = score_pair(
        organ_a="orion-cortex-exec",
        organ_b="orion-bus",
        site_a=2,
        site_b=3,
        i_cofire=[0.201] * 10,
        i_elsewhere=[0.200] * 10,
    )
    assert score.holds is False


def test_relative_threshold_when_scale_is_tiny() -> None:
    # means 0.008 / 0.004 → relative need 0.25 * 0.004 = 0.001
    assert abs(pair_threshold(0.008, 0.004) - 0.001) < 1e-12
    score = score_pair(
        organ_a="orion-biometrics",
        organ_b="orion-cortex-exec",
        site_a=1,
        site_b=2,
        i_cofire=[0.008] * 10,
        i_elsewhere=[0.004] * 10,
    )
    assert score.holds is True


def test_relative_fail_when_cofire_barely_above_elsewhere() -> None:
    score = score_pair(
        organ_a="orion-biometrics",
        organ_b="orion-cortex-exec",
        site_a=1,
        site_b=2,
        i_cofire=[0.0041] * 10,
        i_elsewhere=[0.0040] * 10,
    )
    assert score.holds is False


def test_thermometer_when_both_pairs_fail() -> None:
    weak = score_pair(
        organ_a="orion-biometrics",
        organ_b="orion-cortex-exec",
        site_a=1,
        site_b=2,
        i_cofire=[0.20] * 10,
        i_elsewhere=[0.20] * 10,
    )
    weak2 = score_pair(
        organ_a="orion-cortex-exec",
        organ_b="orion-bus",
        site_a=2,
        site_b=3,
        i_cofire=[0.11] * 10,
        i_elsewhere=[0.10] * 10,
    )
    decided = decide_probe(n_windows=40, pairs=[weak, weak2])
    assert decided.thermometer is True
    assert decided.relational is False


def test_relational_when_both_pairs_hold() -> None:
    strong = score_pair(
        organ_a="orion-biometrics",
        organ_b="orion-cortex-exec",
        site_a=1,
        site_b=2,
        i_cofire=[0.20] * 10,
        i_elsewhere=[0.10] * 10,
    )
    strong2 = score_pair(
        organ_a="orion-cortex-exec",
        organ_b="orion-bus",
        site_a=2,
        site_b=3,
        i_cofire=[0.30] * 10,
        i_elsewhere=[0.10] * 10,
    )
    decided = decide_probe(n_windows=40, pairs=[strong, strong2])
    assert decided.relational is True
    assert decided.thermometer is False


def test_mixed_when_only_one_pair_holds() -> None:
    strong = score_pair(
        organ_a="orion-biometrics",
        organ_b="orion-cortex-exec",
        site_a=1,
        site_b=2,
        i_cofire=[0.20] * 10,
        i_elsewhere=[0.10] * 10,
    )
    weak = score_pair(
        organ_a="orion-cortex-exec",
        organ_b="orion-bus",
        site_a=2,
        site_b=3,
        i_cofire=[0.11] * 10,
        i_elsewhere=[0.10] * 10,
    )
    decided = decide_probe(n_windows=40, pairs=[strong, weak])
    assert decided.mixed is True
    assert decided.thermometer is False
    assert "orion-biometrics/orion-cortex-exec" in decided.reason


def test_unverified_below_min_windows() -> None:
    strong = score_pair(
        organ_a="orion-biometrics",
        organ_b="orion-cortex-exec",
        site_a=1,
        site_b=2,
        i_cofire=[0.20] * 10,
        i_elsewhere=[0.10] * 10,
    )
    decided = decide_probe(n_windows=39, pairs=[strong, strong])
    assert decided.reason.startswith("UNVERIFIED")
    assert decided.thermometer is False
    assert decided.relational is False


def test_v2_zero_is_quiet_not_cofire() -> None:
    assert classify_window_firing(0, 0, 7) == "elsewhere"
    assert classify_window_firing(0, 0, 3) == "neither"
    assert classify_window_firing(1, 1, 0) == "cofire"
    assert classify_window_firing(4, 0, 10) == "neither"
    assert classify_window_firing(0, 5, 10) == "neither"


def test_v2_decides_on_exec_bus_even_if_bio_exec_unverified() -> None:
    bio = score_pair(
        organ_a="orion-biometrics",
        organ_b="orion-cortex-exec",
        site_a=1,
        site_b=2,
        i_cofire=[0.2] * 9,
        i_elsewhere=[0.0] * 3,
    )
    exec_bus = score_pair(
        organ_a="orion-cortex-exec",
        organ_b="orion-bus",
        site_a=2,
        site_b=3,
        i_cofire=[0.20] * 10,
        i_elsewhere=[0.05] * 10,
    )
    assert bio.unverified is True
    decided = decide_probe_v2(n_windows=40, pairs=[bio, exec_bus])
    assert decided.relational is True
    assert decided.thermometer is False


def test_v2_thermometer_when_exec_bus_fails() -> None:
    bio = score_pair(
        organ_a="orion-biometrics",
        organ_b="orion-cortex-exec",
        site_a=1,
        site_b=2,
        i_cofire=[0.20] * 10,
        i_elsewhere=[0.01] * 10,
    )
    exec_bus = score_pair(
        organ_a="orion-cortex-exec",
        organ_b="orion-bus",
        site_a=2,
        site_b=3,
        i_cofire=[0.20] * 10,
        i_elsewhere=[0.20] * 10,
    )
    decided = decide_probe_v2(n_windows=40, pairs=[bio, exec_bus])
    assert decided.thermometer is True
    assert decided.relational is False


def test_v2_unverified_when_exec_bus_cells_empty() -> None:
    bio = score_pair(
        organ_a="orion-biometrics",
        organ_b="orion-cortex-exec",
        site_a=1,
        site_b=2,
        i_cofire=[0.20] * 10,
        i_elsewhere=[0.01] * 10,
    )
    exec_bus = score_pair(
        organ_a="orion-cortex-exec",
        organ_b="orion-bus",
        site_a=2,
        site_b=3,
        i_cofire=[0.20] * 9,
        i_elsewhere=[],
    )
    decided = decide_probe_v2(n_windows=40, pairs=[bio, exec_bus])
    assert decided.reason.startswith("UNVERIFIED")
    assert decided.thermometer is False
    assert decided.relational is False
