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
