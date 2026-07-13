"""Spec for cross-drive tension detection (`drive_tension.py`).

Tension definition under test — "inverse_coactivation": a tension
`(drive_a, drive_b)` is reported when `activations[drive_a] is True` (A is
currently dominant) and `pressures[drive_b] < DriveMathConfig.
deactivate_threshold` (0.42; B is currently suppressed). The threshold
comparison is strict (`<`): a pressure exactly equal to the threshold does
NOT count as suppressed. `magnitude = pressures[drive_a] * (1 -
pressures[drive_b])`.

This is a pure, standalone module — it is not wired into `DriveEngine`,
`bus_worker.py`, or any live path. These tests exercise the function in
isolation with synthetic pressure/activation vectors and known expected
outcomes; nothing here touches the bus, DB, or schema registry.
"""
from __future__ import annotations

from orion.spark.concept_induction.drive_tension import (
    INVERSE_COACTIVATION,
    DriveTensionV1,
    detect_drive_tensions,
)
from orion.spark.concept_induction.drives import DRIVE_KEYS, DriveMathConfig

THRESHOLD = DriveMathConfig().deactivate_threshold  # 0.42


def _pairs(tensions: list[DriveTensionV1]) -> set[tuple[str, str]]:
    return {(t.drive_a, t.drive_b) for t in tensions}


def test_no_tension_when_nothing_active() -> None:
    """All pressures moderate, nothing active -> no tensions at all."""
    pressures = {k: 0.5 for k in DRIVE_KEYS}
    activations = {k: False for k in DRIVE_KEYS}

    result = detect_drive_tensions(pressures, activations)

    assert result == []


def test_no_tension_when_active_but_nothing_low() -> None:
    """A drive active, but no other drive dips below the suppressed bar ->
    no tensions, even though something is active."""
    pressures = {k: 0.5 for k in DRIVE_KEYS}
    pressures["capability"] = 0.9
    activations = {k: False for k in DRIVE_KEYS}
    activations["capability"] = True

    result = detect_drive_tensions(pressures, activations)

    assert result == []


def test_single_pair_tension() -> None:
    """capability active + high, relational genuinely low, everything else
    moderate -> exactly one tension: (capability, relational)."""
    pressures = {k: 0.5 for k in DRIVE_KEYS}
    pressures["capability"] = 0.8
    pressures["relational"] = 0.1
    activations = {k: False for k in DRIVE_KEYS}
    activations["capability"] = True

    result = detect_drive_tensions(pressures, activations)

    assert len(result) == 1
    tension = result[0]
    assert tension.drive_a == "capability"
    assert tension.drive_b == "relational"
    assert tension.tension_kind == INVERSE_COACTIVATION
    # magnitude = 0.8 * (1 - 0.1) = 0.72
    assert abs(tension.magnitude - 0.72) < 1e-9
    assert 0.0 <= tension.magnitude <= 1.0


def test_multiple_simultaneous_tensions() -> None:
    """Two drives active + high, two drives genuinely low, two moderate and
    inactive -> all four cross-pairs qualify, nothing else does."""
    pressures = {
        "coherence": 0.5,       # moderate, inactive
        "continuity": 0.5,      # moderate, inactive
        "capability": 0.8,      # active + high
        "autonomy": 0.7,        # active + high
        "relational": 0.1,      # low
        "predictive": 0.2,      # low
    }
    activations = {
        "coherence": False,
        "continuity": False,
        "capability": True,
        "autonomy": True,
        "relational": False,
        "predictive": False,
    }

    result = detect_drive_tensions(pressures, activations)

    expected_pairs = {
        ("capability", "relational"),
        ("capability", "predictive"),
        ("autonomy", "relational"),
        ("autonomy", "predictive"),
    }
    assert _pairs(result) == expected_pairs
    # capability and autonomy are both active but neither is "low", so
    # neither direction of that pair should appear.
    assert ("capability", "autonomy") not in _pairs(result)
    assert ("autonomy", "capability") not in _pairs(result)
    # every emitted tension is correctly typed and in range.
    for tension in result:
        assert tension.tension_kind == INVERSE_COACTIVATION
        assert 0.0 <= tension.magnitude <= 1.0


def test_boundary_exactly_at_threshold_is_excluded() -> None:
    """pressures[drive_b] == deactivate_threshold exactly -> NOT suppressed,
    no tension. This locks in the exclusive (`<`, not `<=`) boundary."""
    pressures = {k: 0.5 for k in DRIVE_KEYS}
    pressures["capability"] = 0.8
    pressures["relational"] = THRESHOLD  # exactly on the line
    activations = {k: False for k in DRIVE_KEYS}
    activations["capability"] = True

    result = detect_drive_tensions(pressures, activations)

    assert result == []


def test_boundary_just_below_threshold_is_included() -> None:
    """A hair below the threshold -> suppressed, tension fires."""
    pressures = {k: 0.5 for k in DRIVE_KEYS}
    pressures["capability"] = 0.8
    pressures["relational"] = THRESHOLD - 1e-6
    activations = {k: False for k in DRIVE_KEYS}
    activations["capability"] = True

    result = detect_drive_tensions(pressures, activations)

    assert _pairs(result) == {("capability", "relational")}


def test_empty_inputs_return_empty_list() -> None:
    """Empty dicts must not raise; there is nothing to compare."""
    assert detect_drive_tensions({}, {}) == []


def test_missing_key_in_pressures_is_skipped_not_raised() -> None:
    """drive_a is active but has no pressure reading at all (key present in
    activations, absent from pressures) -> that drive cannot participate as
    drive_a; no crash, no bogus tension."""
    activations = {"capability": True}
    pressures = {"relational": 0.1}  # "capability" missing entirely

    result = detect_drive_tensions(pressures, activations)

    assert result == []


def test_missing_key_in_activations_is_treated_as_inactive() -> None:
    """A drive with a pressure reading but no entry in activations can still
    serve as drive_b (suppressed side) for another active drive, but can
    never itself be drive_a since it has no True activation on record."""
    pressures = {"capability": 0.8, "relational": 0.1}
    activations = {"capability": True}  # "relational" absent

    result = detect_drive_tensions(pressures, activations)

    assert _pairs(result) == {("capability", "relational")}
