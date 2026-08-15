"""Fleet power reaches Orion's metacog cue (ROADMAP B1 consumer).

Without this, B1 is schema + producer + storage with no reducer and no cognition consumer --
the empty-shell shape the repo contract bans. This is the step that makes Orion's own power
draw something it can actually see, in watts rather than as a normalised band.
"""
from __future__ import annotations

import json

from app.executor import _metacog_biometrics_cue


def _ctx(cluster: dict) -> dict:
    return {"biometrics": {"status": "ok", "cluster": cluster}}


def test_fleet_watts_reaches_the_cue_in_watts():
    cue = json.loads(_ctx and _metacog_biometrics_cue(_ctx({
        "constraint": "NONE",
        "measurements": {"chassis_watts": 663.0, "gpu_watts_total": 316.63},
    })))
    assert cue["fleet_watts"] == 663
    assert "fleet_watts_partial" not in cue


def test_a_partial_fleet_total_says_which_nodes_are_missing():
    """663 W is real and excludes circe, which has no reachable BMC. A consumer reading the
    number without the exclusion is reading a partial sum as a complete one."""
    cue = json.loads(_metacog_biometrics_cue(_ctx({
        "measurements": {"chassis_watts": 663.0},
        "measurements_missing": {"chassis_watts": ["circe"]},
    })))
    assert cue["fleet_watts"] == 663
    assert cue["fleet_watts_partial"] == ["circe"]


def test_no_measurements_means_no_fleet_key_rather_than_zero():
    cue = json.loads(_metacog_biometrics_cue(_ctx({"constraint": "NONE"})))
    assert "fleet_watts" not in cue
    assert "fleet_watts_partial" not in cue


def test_measurements_without_chassis_watts_is_absent_not_zero():
    """Every node lacking a BMC -- the cluster has GPU watts but no chassis figure."""
    cue = json.loads(_metacog_biometrics_cue(_ctx({
        "measurements": {"gpu_watts_total": 172.96, "gpu_count": 3.0},
    })))
    assert "fleet_watts" not in cue


def test_non_numeric_chassis_watts_is_ignored():
    cue = json.loads(_metacog_biometrics_cue(_ctx({"measurements": {"chassis_watts": "lots"}})))
    assert "fleet_watts" not in cue


def test_missing_biometrics_context_still_degrades_cleanly():
    cue = json.loads(_metacog_biometrics_cue({}))
    assert cue["status"] == "missing"


def test_cue_stays_valid_json_under_the_char_budget():
    """The cue has a size cap with two fallback shapes; adding a key must not break them."""
    from app.executor import _METACOG_BIOMETRICS_CUE_DRAFT_MAX_CHARS

    cue = _metacog_biometrics_cue(_ctx({
        "constraint": "POWER",
        "measurements": {"chassis_watts": 663.0},
        "measurements_missing": {"chassis_watts": ["circe"]},
    }))
    assert len(cue) <= _METACOG_BIOMETRICS_CUE_DRAFT_MAX_CHARS
    json.loads(cue)


# ------------------------------------------------------- binding constraint (ROADMAP B2)

def test_the_peak_reaches_the_cue_with_its_channel_and_node():
    """strain in this same cue reads 0.44 on a node whose power is pegged at 1.000. Orion
    reading only that concludes its body is fine while a resource is full."""
    cue = json.loads(_metacog_biometrics_cue(_ctx({
        "composites": {"strain": 0.44},
        "peak_pressure": 1.0,
        "peak_pressure_channel": "power",
        "peak_pressure_node": "athena",
    })))
    assert cue["peak"] == 1.0
    assert cue["peak_at"] == "athena.power"


def test_the_channel_survives_a_missing_node():
    cue = json.loads(_metacog_biometrics_cue(_ctx({
        "peak_pressure": 0.772, "peak_pressure_channel": "disk_capacity",
    })))
    assert cue["peak"] == 0.77
    assert cue["peak_at"] == "disk_capacity"


def test_no_peak_means_no_key_rather_than_zero():
    cue = json.loads(_metacog_biometrics_cue(_ctx({"constraint": "NONE"})))
    assert "peak" not in cue and "peak_at" not in cue


def test_a_non_numeric_peak_is_ignored():
    cue = json.loads(_metacog_biometrics_cue(_ctx({"peak_pressure": "high"})))
    assert "peak" not in cue


def test_a_calm_fleet_reports_a_real_zero_peak():
    """0.0 is a reading, not an absence -- it must reach the cue."""
    cue = json.loads(_metacog_biometrics_cue(_ctx({
        "peak_pressure": 0.0, "peak_pressure_channel": "cpu", "peak_pressure_node": "atlas",
    })))
    assert cue["peak"] == 0.0
    assert cue["peak_at"] == "atlas.cpu"


def test_the_cue_stays_under_budget_with_every_field_present():
    from app.executor import _METACOG_BIOMETRICS_CUE_DRAFT_MAX_CHARS

    cue = _metacog_biometrics_cue(_ctx({
        "constraint": "POWER",
        "composites": {"strain": 0.44, "homeostasis": 0.56, "stability": 0.98},
        "measurements": {"chassis_watts": 786.0},
        "measurements_missing": {"chassis_watts": ["circe"]},
        "peak_pressure": 1.0, "peak_pressure_channel": "power", "peak_pressure_node": "athena",
    }))
    assert len(cue) <= _METACOG_BIOMETRICS_CUE_DRAFT_MAX_CHARS
    json.loads(cue)
