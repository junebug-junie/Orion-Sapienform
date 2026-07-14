from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import drive_state_divergence_audit as dsa  # noqa: E402
from orion.autonomy.models import AutonomyStateV2  # noqa: E402


def _autonomy_state(*, drive_pressures: dict, active_drives: list[str]) -> AutonomyStateV2:
    return AutonomyStateV2(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        source="reducer",
        drive_pressures=drive_pressures,
        active_drives=active_drives,
    )


def _mock_store(load_drive_state_return):
    """Build a mock LocalProfileStore class whose instances' load_drive_state
    returns `load_drive_state_return` (or raises, if it's an exception instance)."""
    instance = Mock()
    if isinstance(load_drive_state_return, Exception):
        instance.load_drive_state.side_effect = load_drive_state_return
    else:
        instance.load_drive_state.return_value = load_drive_state_return
    cls = Mock(return_value=instance)
    return cls


# --------------------------------------------------------------------------
# load_drive_state_v1 / load_autonomy_state loader wrappers
# --------------------------------------------------------------------------

def test_load_drive_state_v1_missing_store_returns_none_no_error():
    with patch.object(dsa, "LocalProfileStore", _mock_store({})):
        state, error = dsa.load_drive_state_v1("/tmp/does-not-matter.json", "orion")
    assert state is None
    assert error is None


def test_load_drive_state_v1_present():
    raw = {"pressures": {"coherence": 0.5}, "activations": {"coherence": True}, "updated_at": "2026-07-13T00:00:00+00:00"}
    with patch.object(dsa, "LocalProfileStore", _mock_store(raw)):
        state, error = dsa.load_drive_state_v1("/tmp/does-not-matter.json", "orion")
    assert state == raw
    assert error is None


def test_load_drive_state_v1_raises_is_caught():
    with patch.object(dsa, "LocalProfileStore", _mock_store(RuntimeError("disk full"))):
        state, error = dsa.load_drive_state_v1("/tmp/does-not-matter.json", "orion")
    assert state is None
    assert error is not None
    assert "disk full" in error


def test_load_autonomy_state_none():
    with patch.object(dsa, "load_autonomy_state_v2", return_value=None):
        state, error = dsa.load_autonomy_state("orion")
    assert state is None
    assert error is None


def test_load_autonomy_state_present():
    expected = _autonomy_state(drive_pressures={"coherence": 0.4}, active_drives=["coherence"])
    with patch.object(dsa, "load_autonomy_state_v2", return_value=expected):
        state, error = dsa.load_autonomy_state("orion")
    assert state is expected
    assert error is None


def test_load_autonomy_state_raises_is_caught():
    with patch.object(dsa, "load_autonomy_state_v2", side_effect=RuntimeError("db down")):
        state, error = dsa.load_autonomy_state("orion")
    assert state is None
    assert error is not None
    assert "db down" in error


# --------------------------------------------------------------------------
# compare_drives
# --------------------------------------------------------------------------

def test_compare_drives_both_present_real_divergence():
    drive_raw = {
        "pressures": {
            "coherence": 0.80,
            "continuity": 0.10,
            "capability": 0.50,
            "relational": 0.20,
            "predictive": 0.30,
            "autonomy": 0.60,
        },
        "activations": {
            "coherence": True,
            "continuity": False,
            "capability": True,
            "relational": False,
            "predictive": False,
            "autonomy": True,
        },
    }
    autonomy_state = _autonomy_state(
        drive_pressures={
            "coherence": 0.20,
            "continuity": 0.15,
            "capability": 0.50,
            "relational": 0.90,
            "predictive": 0.30,
            "autonomy": 0.05,
        },
        # active_drives disagrees with drive_state.v1 on coherence and autonomy,
        # agrees on continuity/capability/relational/predictive.
        active_drives=["capability"],
    )

    per_drive, summary = dsa.compare_drives(drive_raw, autonomy_state)

    assert per_drive["coherence"]["abs_diff"] == pytest.approx(0.60)
    assert per_drive["continuity"]["abs_diff"] == pytest.approx(0.05)
    assert per_drive["capability"]["abs_diff"] == pytest.approx(0.0)
    assert per_drive["relational"]["abs_diff"] == pytest.approx(0.70)
    assert per_drive["predictive"]["abs_diff"] == pytest.approx(0.0)
    assert per_drive["autonomy"]["abs_diff"] == pytest.approx(0.55)

    assert summary["drives_compared_pressure"] == 6
    assert summary["max_abs_diff_drive"] == "relational"
    assert summary["max_abs_diff"] == pytest.approx(0.70)
    assert summary["mean_abs_diff"] == pytest.approx((0.60 + 0.05 + 0.0 + 0.70 + 0.0 + 0.55) / 6)

    # activation agreement: coherence True vs not-in-active_drives(False) -> disagree
    assert per_drive["coherence"]["activation_agree"] is False
    # continuity False vs not-in-active_drives(False) -> agree
    assert per_drive["continuity"]["activation_agree"] is True
    # capability True vs in active_drives(True) -> agree
    assert per_drive["capability"]["activation_agree"] is True
    # relational False vs not-in-active_drives(False) -> agree
    assert per_drive["relational"]["activation_agree"] is True
    # predictive False vs not-in-active_drives(False) -> agree
    assert per_drive["predictive"]["activation_agree"] is True
    # autonomy True vs not-in-active_drives(False) -> disagree
    assert per_drive["autonomy"]["activation_agree"] is False

    assert summary["activation_drives_compared"] == 6
    assert summary["activation_agreements"] == 4
    assert summary["activation_disagreements"] == 2


def test_compare_drives_autonomy_missing():
    drive_raw = {
        "pressures": {k: 0.5 for k in dsa.DRIVE_KEYS},
        "activations": {k: True for k in dsa.DRIVE_KEYS},
    }
    per_drive, summary = dsa.compare_drives(drive_raw, None)
    for key in dsa.DRIVE_KEYS:
        assert per_drive[key]["pressure_autonomy_state_v2"] is None
        assert per_drive[key]["abs_diff"] is None
        assert per_drive[key]["active_autonomy_state_v2"] is None
        assert per_drive[key]["activation_agree"] is None
        assert per_drive[key]["pressure_drive_state_v1"] == 0.5
    assert summary["drives_compared_pressure"] == 0
    assert summary["mean_abs_diff"] is None
    assert summary["max_abs_diff_drive"] is None
    assert summary["activation_drives_compared"] == 0


def test_compare_drives_drive_state_missing():
    autonomy_state = _autonomy_state(
        drive_pressures={k: 0.3 for k in dsa.DRIVE_KEYS},
        active_drives=list(dsa.DRIVE_KEYS),
    )
    per_drive, summary = dsa.compare_drives(None, autonomy_state)
    for key in dsa.DRIVE_KEYS:
        assert per_drive[key]["pressure_drive_state_v1"] is None
        assert per_drive[key]["abs_diff"] is None
        assert per_drive[key]["activation_drive_state_v1"] is None
        assert per_drive[key]["activation_agree"] is None
        assert per_drive[key]["pressure_autonomy_state_v2"] == 0.3
    assert summary["drives_compared_pressure"] == 0
    assert summary["activation_drives_compared"] == 0


def test_compare_drives_corrupted_pressure_value_does_not_crash():
    # LocalProfileStore's JSON file is unvalidated (no pydantic model) -- a
    # hand-edited or corrupted store could hold a non-numeric pressure value.
    # This must degrade that single key to "unavailable", not raise.
    drive_raw = {
        "pressures": {"coherence": "not-a-number", "continuity": 0.4},
        "activations": {"coherence": True, "continuity": False},
    }
    autonomy_state = _autonomy_state(
        drive_pressures={"coherence": 0.5, "continuity": 0.5},
        active_drives=[],
    )
    per_drive, summary = dsa.compare_drives(drive_raw, autonomy_state)
    assert per_drive["coherence"]["pressure_drive_state_v1"] is None
    assert per_drive["coherence"]["abs_diff"] is None
    assert per_drive["continuity"]["abs_diff"] == pytest.approx(0.1)
    assert summary["drives_compared_pressure"] == 1


def test_compare_drives_both_missing():
    per_drive, summary = dsa.compare_drives(None, None)
    for key in dsa.DRIVE_KEYS:
        assert per_drive[key]["pressure_drive_state_v1"] is None
        assert per_drive[key]["pressure_autonomy_state_v2"] is None
        assert per_drive[key]["abs_diff"] is None
    assert summary["drives_compared_pressure"] == 0
    assert summary["mean_abs_diff"] is None
    assert summary["activation_drives_compared"] == 0


# --------------------------------------------------------------------------
# main() end-to-end (loaders mocked)
# --------------------------------------------------------------------------

def test_main_json_both_present_exits_zero(capsys):
    drive_raw = {
        "pressures": {k: 0.4 for k in dsa.DRIVE_KEYS},
        "activations": {k: False for k in dsa.DRIVE_KEYS},
        "updated_at": "2026-07-13T00:00:00+00:00",
    }
    autonomy_state = _autonomy_state(
        drive_pressures={k: 0.6 for k in dsa.DRIVE_KEYS},
        active_drives=[],
    )
    with patch.object(dsa, "LocalProfileStore", _mock_store(drive_raw)), patch.object(
        dsa, "load_autonomy_state_v2", return_value=autonomy_state
    ):
        exit_code = dsa.main(["--json"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["drive_state_v1"]["available"] is True
    assert out["autonomy_state_v2"]["available"] is True
    assert out["summary"]["drives_compared_pressure"] == 6
    for key in dsa.DRIVE_KEYS:
        assert out["per_drive"][key]["abs_diff"] == pytest.approx(0.2)


def test_main_prose_both_missing_exits_zero(capsys):
    with patch.object(dsa, "LocalProfileStore", _mock_store({})), patch.object(
        dsa, "load_autonomy_state_v2", return_value=None
    ):
        exit_code = dsa.main([])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "drive_state.v1: UNAVAILABLE" in out
    assert "autonomy_state_v2: UNAVAILABLE" in out
    assert "Cannot compute divergence: both signals are unavailable." in out


def test_main_prose_one_missing_exits_zero_reports_na(capsys):
    autonomy_state = _autonomy_state(
        drive_pressures={k: 0.6 for k in dsa.DRIVE_KEYS},
        active_drives=["capability"],
    )
    with patch.object(dsa, "LocalProfileStore", _mock_store({})), patch.object(
        dsa, "load_autonomy_state_v2", return_value=autonomy_state
    ):
        exit_code = dsa.main([])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "drive_state.v1: UNAVAILABLE" in out
    assert "autonomy_state_v2: available" in out
    assert "n/a" in out


# --------------------------------------------------------------------------
# store_path provenance warning (2026-07-13 incident: an operator ran this
# with neither --store-path nor $CONCEPT_STORE_PATH set, silently landed on
# DEFAULT_CONCEPT_STORE_PATH, and a stale dev-leftover file sitting at that
# path was misread as a 24h+-stale live drive_state.v1 signal.)
# --------------------------------------------------------------------------

def test_main_warns_when_store_path_falls_back_to_default(capsys, monkeypatch):
    monkeypatch.delenv("CONCEPT_STORE_PATH", raising=False)
    with patch.object(dsa, "LocalProfileStore", _mock_store({})), patch.object(
        dsa, "load_autonomy_state_v2", return_value=None
    ):
        exit_code = dsa.main([])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "WARNING:" in out
    assert "almost certainly NOT the path the live Docker container writes to" in out
    assert dsa.DEFAULT_CONCEPT_STORE_PATH in out


def test_main_no_warning_when_store_path_given_explicitly(capsys, monkeypatch):
    monkeypatch.delenv("CONCEPT_STORE_PATH", raising=False)
    with patch.object(dsa, "LocalProfileStore", _mock_store({})), patch.object(
        dsa, "load_autonomy_state_v2", return_value=None
    ):
        exit_code = dsa.main(["--store-path", "/data/concept-induction-state.json"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "WARNING:" not in out


def test_main_no_warning_when_env_var_set(capsys, monkeypatch):
    monkeypatch.setenv("CONCEPT_STORE_PATH", "/mnt/graphdb/orion/concepts/concept-induction-state.json")
    with patch.object(dsa, "LocalProfileStore", _mock_store({})), patch.object(
        dsa, "load_autonomy_state_v2", return_value=None
    ):
        exit_code = dsa.main([])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "WARNING:" not in out


def test_main_json_reports_store_path_source_and_warning(monkeypatch, capsys):
    monkeypatch.delenv("CONCEPT_STORE_PATH", raising=False)
    with patch.object(dsa, "LocalProfileStore", _mock_store({})), patch.object(
        dsa, "load_autonomy_state_v2", return_value=None
    ):
        exit_code = dsa.main(["--json"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["drive_state_v1"]["store_path_source"] == "default_fallback"
    assert out["drive_state_v1"]["store_path_warning"] is not None
    assert out["drive_state_v1"]["store_path"] == dsa.DEFAULT_CONCEPT_STORE_PATH


def test_main_json_store_path_source_env(monkeypatch, capsys):
    monkeypatch.setenv("CONCEPT_STORE_PATH", "/mnt/graphdb/orion/concepts/concept-induction-state.json")
    with patch.object(dsa, "LocalProfileStore", _mock_store({})), patch.object(
        dsa, "load_autonomy_state_v2", return_value=None
    ):
        exit_code = dsa.main(["--json"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["drive_state_v1"]["store_path_source"] == "env:CONCEPT_STORE_PATH"
    assert out["drive_state_v1"]["store_path_warning"] is None
    assert out["drive_state_v1"]["store_path"] == "/mnt/graphdb/orion/concepts/concept-induction-state.json"


def test_main_json_store_path_source_cli(monkeypatch, capsys):
    monkeypatch.delenv("CONCEPT_STORE_PATH", raising=False)
    with patch.object(dsa, "LocalProfileStore", _mock_store({})), patch.object(
        dsa, "load_autonomy_state_v2", return_value=None
    ):
        exit_code = dsa.main(["--json", "--store-path", "/data/concept-induction-state.json"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["drive_state_v1"]["store_path_source"] == "cli:--store-path"
    assert out["drive_state_v1"]["store_path_warning"] is None
    assert out["drive_state_v1"]["store_path"] == "/data/concept-induction-state.json"


def test_main_never_crashes_on_loader_exceptions(capsys):
    with patch.object(dsa, "LocalProfileStore", _mock_store(RuntimeError("boom"))), patch.object(
        dsa, "load_autonomy_state_v2", side_effect=RuntimeError("boom2")
    ):
        exit_code = dsa.main([])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "boom" in out
    assert "boom2" in out
