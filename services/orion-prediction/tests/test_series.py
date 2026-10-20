from __future__ import annotations

import pytest

from app.models import split_seasonalities
from app.series import SeriesConfigError, duration_min, load_registry, parse_registry
from tests.conftest import REGISTRY_PATH
from tests.helpers import spec, spec_doc


def test_repo_registry_loads_and_declares_the_cabinet_target():
    reg = load_registry(REGISTRY_PATH)
    t = reg.get("athena.cabinet_temp_c")
    assert t.role == "target" and t.grain_min == 5
    assert t.threshold("shed").value == 34.0
    assert t.threshold("experiment_guard").max_probability == 0.05
    assert set(t.covariates) == {"athena.chassis_watts", "athena.cabinet_cooling_watts"}
    assert all(s.consumers for s in reg.series.values())


def test_monthly_is_refused_on_48_days_and_weekly_kept():
    t = load_registry(REGISTRY_PATH).get("athena.cabinet_temp_c")
    usable, refused = split_seasonalities(t, 48 * 288)
    assert "monthly" in refused
    assert {s.name for s in usable} == {"daily", "weekly"}
    usable, refused = split_seasonalities(t, 60 * 288)
    assert refused == []


def test_weekly_needs_two_full_cycles():
    s = spec(seasonalities=[{"name": "weekly", "period": "7d"}])
    assert split_seasonalities(s, 13 * 288)[1] == ["weekly"]
    assert split_seasonalities(s, 14 * 288)[1] == []


@pytest.mark.parametrize("value,expected", [("5min", 5), ("1h", 60), ("2d", 2880)])
def test_duration(value, expected):
    assert duration_min(value) == expected


@pytest.mark.parametrize("over,msg", [
    ({"grain": "7min"}, "multiple"),
    ({"horizons": ["5min", "1h", "1h"]}, "duplicate horizons"),
    ({"consumers": []}, "consumers"),
    ({"covariates": ["test.nope"]}, "unknown covariate"),
    ({"seasonalities": [{"name": "odd", "period": "13min"}]}, "multiple"),
    ({"source": {"kind": "biometrics", "node": "n", "key": "Bad-Key"}}, "key"),
    ({"transform": "logit"}, "transform"),
    ({"thresholds": [{"name": "hot", "value": 31.0, "window": "12h"}]}, "longer than the longest horizon"),
])
def test_bad_registry_refused(over, msg):
    with pytest.raises(SeriesConfigError, match=msg):
        parse_registry(spec_doc(**over))


def test_version_must_be_one():
    with pytest.raises(SeriesConfigError):
        parse_registry({"version": 2, "series": []})
