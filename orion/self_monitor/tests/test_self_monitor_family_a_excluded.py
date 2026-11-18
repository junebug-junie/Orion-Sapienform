"""The self-monitor's forecast source skips orion-prediction's family A reading rows.

Family A (services/orion-prediction/app/family_a_live.py) stores ~23k per-reading rows a day in
prediction_forecast under series_id 'family_a:<reading>'. Grading them is a later, explicit decision;
until then the self-monitor must not pick them up as ~40 new forecast faculties.
"""
from __future__ import annotations

import re

from orion.self_monitor.collect import SOURCES

FAMILY_A_PREFIX = "family_a:"   # services/orion-prediction/app/family_a_live.py PREFIX


def _forecast_sql() -> str:
    return next(sql for name, sql, _adapter in SOURCES if name == "prediction_forecast")


def test_forecast_source_excludes_the_family_a_prefix():
    sql = _forecast_sql()
    m = re.search(r"left\(series_id,\s*(\d+)\)\s*<>\s*'([^']*)'", sql)
    assert m, sql
    assert m.group(2) == FAMILY_A_PREFIX
    assert int(m.group(1)) == len(FAMILY_A_PREFIX)


def test_forecast_source_still_reads_single_series_champions():
    sql = _forecast_sql()
    assert "role = 'champion'" in sql and "scored_at >= :since" in sql
