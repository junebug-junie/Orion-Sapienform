"""Blind-detector watch (orion/vision_detection_rate_watch.py) and its wiring
into the ladder report. Fixtures are real rows of DETECTION_RATE_SQL read
from live vision_scene_inventory on 2026-10-29 (d, windows, windows with a
detection; d=0 is the trailing 3 h ending at the given time)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion import substrate_ladder_liveness as ll
from orion import vision_detection_rate_watch as dr

# Walkway, 3 h ending 2026-10-22 18:00Z: the incident (detector half-blind).
INCIDENT_10_22 = [(0, 1604, 92), (1, 1234, 996), (2, 1247, 1125), (3, 1201, 1061),
                  (4, 1170, 1060), (5, 201, 147), (6, 1294, 528), (7, 1183, 1158)]
# Walkway, 3 h ending 2026-10-28 08:00Z (02:00 local): a normal night.
NORMAL_NIGHT_10_28 = [(0, 1320, 2), (1, 911, 1), (2, 1405, 518), (3, 1122, 0),
                      (4, 1131, 1), (5, 1152, 0), (6, 1142, 3), (7, 1119, 0)]
# Walkway, 3 h ending 2026-10-26 15:00Z: the DHCP outage, no windows at all.
OUTAGE_10_26 = [(0, 0, 0), (1, 1606, 216), (2, 1603, 119), (3, 1511, 26),
                (4, 1509, 16), (5, 1469, 885), (6, 1507, 233), (7, 1145, 434)]
# Walkway, 3 h ending 2026-10-28 18:00Z: a normal day whose baseline holds the
# two blind days (d=5, d=6) and a missing day (d=2).
NORMAL_DAY_10_28 = [(0, 1539, 1515), (1, 1383, 849), (2, 0, 0), (3, 1594, 1341),
                    (4, 1610, 1056), (5, 1607, 193), (6, 1604, 92), (7, 1234, 996)]


def _eval(rows, stream="walkway"):
    cur, base = dr.split_rows(rows)
    return dr.evaluate_detection_rate(stream, cur, base)


def test_incident_is_red():
    r = _eval(INCIDENT_10_22)
    assert r.status == "collapsed" and r.red
    assert r.current.share == pytest.approx(92 / 1604)
    # Day 5 (201 windows) is still >= min_baseline_windows; all 7 count.
    assert r.baseline_days_used == 7
    assert "gone blind" in r.summary()


def test_normal_night_reads_calm_not_unknown():
    r = _eval(NORMAL_NIGHT_10_28)
    assert r.status == "ok" and not r.red
    assert "quiet hours" in r.reason


def test_outage_reads_unknown_not_calm():
    r = _eval(OUTAGE_10_26)
    assert r.status == "unknown" and not r.red


def test_normal_day_with_blind_days_in_baseline_is_ok():
    r = _eval(NORMAL_DAY_10_28)
    assert r.status == "ok"
    assert r.baseline_days_used == 6  # the empty day is skipped, not read as 0


def test_thin_baseline_is_unknown():
    rows = [(0, 1500, 10), (1, 1500, 1400), (2, 50, 40), (3, 0, 0)]
    assert _eval(rows).status == "unknown"


def test_ratio_boundary():
    cfg = dr.DetectionRateConfig()
    base = [dr.SpanCount(1000, 800)] * 3  # median 0.8 -> red under 0.2
    assert dr.evaluate_detection_rate("w", dr.SpanCount(1000, 200), base, cfg).status == "ok"
    assert dr.evaluate_detection_rate("w", dr.SpanCount(1000, 199), base, cfg).status == "collapsed"


def test_sql_is_bounded_and_parameterised():
    assert "make_interval" in dr.DETECTION_RATE_SQL
    assert dr.DETECTION_RATE_SQL.count("%s") == len(dr.sql_params("walkway", datetime.now(timezone.utc)))


def test_streams_match_camera_input_rungs():
    assert set(dr.DETECTION_RATE_STREAMS) == set(ll.CAMERA_INPUT_STREAMS_EXPECTED_ONLINE)


def _report(*rows_by_stream):
    rep = ll.LadderReport()
    rep.detection_rates = [_eval(rows, stream) for stream, rows in rows_by_stream]
    return rep


def test_ladder_report_red_key_and_warning_severity():
    rep = _report(("walkway", INCIDENT_10_22), ("cam0", NORMAL_DAY_10_28))
    assert rep.red
    assert rep.red_keys() == ["rung:detection_rate:walkway"]
    assert "rung:detection_rate:cam0" in rep.green_keys()
    assert rep.severity() == "warning"
    assert "gone blind" in rep.alert_message()
    d = rep.to_dict()["detection_rates"]
    assert {x["rung"] for x in d} == {"detection_rate:walkway", "detection_rate:cam0"}


def test_unknown_is_neither_red_nor_green():
    rep = _report(("walkway", OUTAGE_10_26))
    assert not rep.red
    assert "rung:detection_rate:walkway" not in rep.green_keys()


def test_severity_stays_critical_when_a_ladder_rung_is_also_red():
    rep = _report(("walkway", INCIDENT_10_22))
    rep.rungs = [ll.RungResult("field_state", "stale", None, None, 900.0)]
    assert rep.severity() == "critical"
