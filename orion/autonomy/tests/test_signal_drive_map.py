"""Task 3: structural signal->drive map validates and stays keyword-free."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from orion.autonomy import signal_drive_map as sdm_mod
from orion.autonomy.signal_drive_map import (
    SignalDriveMapError,
    load_signal_drive_map,
)
from orion.spark.concept_induction.drives import DRIVE_KEYS


def test_shipped_map_validates() -> None:
    m = load_signal_drive_map()
    assert "biometrics_state" in m.signal_kinds()
    assert "failure_event" in m.signal_kinds()
    for kind in m.signal_kinds():
        for rule in m.rules_for(kind):
            assert rule.worse in {"up", "down"}
            assert rule.drives
            assert all(d in DRIVE_KEYS for d in rule.drives)


def test_unmapped_kind_returns_empty() -> None:
    m = load_signal_drive_map()
    assert m.rules_for("scene_state") == []
    assert m.rules_for("totally_unknown_kind") == []
    assert m.match("scene_state", "salience") is None


def test_exact_match_wins() -> None:
    m = load_signal_drive_map()
    rule = m.match("spark_signal", "coherence")
    assert rule is not None and rule.worse == "down"
    assert "coherence" in rule.drives


def test_suffix_match_for_dynamic_biometric_dims() -> None:
    """Real biometrics dims are '<metric>_level' — suffix rule must catch them."""
    m = load_signal_drive_map()
    rule = m.match("biometrics_state", "heart_rate_level")
    assert rule is not None and rule.worse == "down"
    assert "capability" in rule.drives
    vol = m.match("biometrics_state", "hrv_volatility")
    assert vol is not None and vol.worse == "up"
    # A dimension with no matching suffix maps nothing.
    assert m.match("biometrics_state", "confidence") is None


def test_rejects_unknown_drive(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        "version: 1\nsignal_kinds:\n  x:\n    d:\n      worse: up\n      drives: {not_a_drive: 0.5}\n"
    )
    with pytest.raises(SignalDriveMapError):
        load_signal_drive_map(p)


def test_rejects_bad_worse(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        "version: 1\nsignal_kinds:\n  x:\n    d:\n      worse: sideways\n      drives: {capability: 0.5}\n"
    )
    with pytest.raises(SignalDriveMapError):
        load_signal_drive_map(p)


def test_rejects_weight_out_of_range(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        "version: 1\nsignal_kinds:\n  x:\n    d:\n      worse: up\n      drives: {capability: 1.5}\n"
    )
    with pytest.raises(SignalDriveMapError):
        load_signal_drive_map(p)


def test_no_text_matching_in_module() -> None:
    """Grep-guard: the mapping module must not inspect natural-language signal
    fields. It reads only typed keys (kind, dimension, worse, drives)."""
    src = inspect.getsource(sdm_mod)
    for banned in (".summary", ".notes", "re.search", "re.match", "lower()", "keyword"):
        assert banned not in src, f"map module must not touch {banned!r}"
