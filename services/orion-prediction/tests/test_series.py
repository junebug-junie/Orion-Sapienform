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
    assert set(t.covariates) == {"athena.chassis_watts", "circe.chassis_watts", "hecate.chassis_watts",
                                 "athena.cabinet_cooling_watts", "athena.central_ac_proxy"}
    proxy = reg.get("athena.central_ac_proxy")
    assert proxy.source.kind == "weather" and proxy.lag_min == 180 and proxy.above == 75.0
    assert reg.get("hecate.chassis_watts").zero_before_first and reg.get("circe.chassis_watts").source.kind == "cluster"
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
    ({"source": {"kind": "weather", "key": "humidity"}}, "temp_f"),
    ({"source": {"kind": "cluster", "node": "Circe!", "key": "w"}}, "cluster source"),
    ({"source": {"kind": "office_env", "node": "pan-1", "key": "sample_age_sec"}}, "office_env source"),
    ({"source": {"kind": "office_env", "node": "pan-1", "key": "office_temp_c; DROP TABLE x"}}, "office_env source"),
    ({"source": {"kind": "office_env", "node": "pan 1;", "key": "office_temp_c"}}, "office_env source"),
    ({"source": {"kind": "office_env", "key": "office_temp_c"}}, "office_env source"),
    ({"thresholds": [{"name": "hot", "value": 31.0, "window": "12h"}]}, "longer than the longest horizon"),
])
def test_bad_registry_refused(over, msg):
    with pytest.raises(SeriesConfigError, match=msg):
        parse_registry(spec_doc(**over))


def test_version_must_be_one():
    with pytest.raises(SeriesConfigError):
        parse_registry({"version": 2, "series": []})


def test_cabinet_thresholds_are_pinned_to_the_thermal_owner():
    """Body economy Patch B: the cabinet lines come from orion/autonomy/thermal_gate.py, not literals.
    Values written by hand: a moved owner line or a registry that stops reading it fails here."""
    t = load_registry(REGISTRY_PATH).get("athena.cabinet_temp_c")
    assert (t.threshold("shed").value, t.threshold("shed").window_min) == (34.0, 30)
    assert (t.threshold("experiment_guard").value, t.threshold("experiment_guard").window_min) == (32.0, 30)


def test_a_literal_cabinet_line_is_refused():
    doc = spec_doc(source={"kind": "biometrics", "node": "athena", "key": "cabinet_temp_c"},
                   thresholds=[{"name": "shed", "value": 34.0, "window": "30min"}])
    with pytest.raises(SeriesConfigError, match="value_from"):
        parse_registry(doc)


@pytest.mark.parametrize("bad", [
    {"name": "x", "window": "30min"},
    {"name": "x", "value": 1.0, "value_from": "thermal_gate.hot", "window": "30min"},
    {"name": "x", "value_from": "thermal_gate.nope", "window": "30min"},
])
def test_value_from_is_exactly_one_known_owner(bad):
    with pytest.raises(SeriesConfigError):
        parse_registry(spec_doc(thresholds=[bad]))


def test_repo_registry_declares_the_young_office_target_without_wiring_it_into_the_cabinet():
    reg = load_registry(REGISTRY_PATH)
    o = reg.get("office.temp_c")
    assert o.role == "target" and o.source.kind == "office_env"
    assert (o.source.node, o.source.key) == ("pan-1", "office_temp_c")
    assert o.grain_min == 5 and o.agg == "mean" and o.max_fill_min == 15 and o.train_window_min == 21 * 1440
    assert o.train_from.isoformat() == "2026-10-28T22:16:00+00:00"
    assert [(a.isoformat(), b.isoformat()) for a, b in o.exclude] == [
        ("2026-10-28T22:16:00+00:00", "2026-10-29T00:10:00+00:00")]   # placement + sensor settling
    assert o.horizons_min == reg.get("athena.cabinet_temp_c").horizons_min
    assert o.thresholds == () and o.covariates == () and o.consumers == ("prediction-scoreboard",)
    assert "office.temp_c" not in reg.get("athena.cabinet_temp_c").covariates
    assert {t.series_id for t in reg.targets()} == {"athena.cabinet_temp_c", "office.temp_c"}
    # Daily needs 2 days, weekly 14: both refused at first, weekly alone at ~4 days.
    assert split_seasonalities(o, 3 * 12)[1] == ["daily", "weekly"]
    assert split_seasonalities(o, 4 * 288)[1] == ["weekly"]
    assert split_seasonalities(o, 14 * 288)[1] == []
