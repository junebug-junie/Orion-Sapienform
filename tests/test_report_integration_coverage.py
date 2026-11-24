"""The live coverage report's pure half (scripts/report_integration_coverage.py build_report / render)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("report_integration_coverage", REPO / "scripts/report_integration_coverage.py")
rep_mod = importlib.util.module_from_spec(_spec)
sys.modules["report_integration_coverage"] = rep_mod
_spec.loader.exec_module(rep_mod)

from orion.integration.organs import load_organs  # noqa: E402

HI = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
LO = HI - timedelta(hours=24)


def test_a_cite_by_another_organ_closes_one_by_its_own_organ_does_not():
    rep = rep_mod.build_report(
        kind_counts={"reading_turn": 4, "gpu_wait": 2, "immune_finding": 1},
        uses=[("reading_turn:a", "reading_turn", "prime"), ("gpu_wait:b", "gpu_wait", "body")],
        cites_made={"prime": 1, "body": 1}, reads={"prime": (3, 2)}, lo=LO, hi=HI, registry=load_organs(),
        missing_tables=[])
    by = {o["organ"]: o for o in rep["organs"]}
    assert by["reading"]["closed"] and by["reading"]["cites_received"] == 1
    assert not by["body"]["closed"] and by["body"]["cites_received"] == 0
    assert by["prime"]["now_reads"] == 3 and by["prime"]["reads_no_cite"] == 2 and by["prime"]["cites_made"] == 1
    assert rep["closed_organs"] == ["reading"] and rep["used_by_other"] == 1 and rep["experiences"] == 6
    assert rep["unregistered_kinds"] == {"immune_finding": 1}
    text = rep_mod.render(rep)
    assert "experience use rate: 1 / 6 = 16.67%; closed organs: 1 of" in text
    assert "unregistered kinds (no organ claims them): {'immune_finding': 1}" in text


def test_missing_tables_are_named_and_read_as_zero():
    rep = rep_mod.build_report(kind_counts={"reading_turn": 2}, uses=[], cites_made={}, reads={}, lo=LO, hi=HI,
                               registry=load_organs(), missing_tables=["experience_shown", "experience_use"])
    assert rep["experience_use_rate"] == 0.0 and rep["closed_count"] == 0
    assert "missing tables (migration manual_migration_experience_now_v1.sql not applied): experience_shown, " \
           "experience_use" in rep_mod.render(rep)


def test_reads_split_closed_with_no_cite_from_never_closed():
    """#3498: "read and cited nothing" (closed, no use) is not "never closed" (the model never saw the read)."""
    rep = rep_mod.build_report(kind_counts={"reading_turn": 2}, uses=[], cites_made={},
                               reads={"reverie": (5, 2, 3)}, lo=LO, hi=HI, registry=load_organs(), missing_tables=[])
    by = {o["organ"]: o for o in rep["organs"]}
    assert (by["reverie"]["now_reads"], by["reverie"]["reads_no_cite"], by["reverie"]["reads_unclosed"]) == (5, 2, 3)
    assert by["prime"]["reads_unclosed"] == 0
    line = next(ln for ln in rep_mod.render(rep).splitlines() if ln.startswith("reverie"))
    assert line.split()[1:5] == ["0", "5", "2", "3"]
    assert "unclosed" in rep_mod.render(rep).splitlines()[1]
