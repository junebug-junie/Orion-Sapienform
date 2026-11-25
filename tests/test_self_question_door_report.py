"""The door's weekly report (scripts/analysis/self_question_door_report.py): the dressing rate and the unjudged
share, with None (never 0) when a week has no base."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))

from self_question_door_report import weekly  # noqa: E402


def test_dressing_rate_counts_demoted_against_everything_admitted():
    rows = [{"week": "2026-11-23", "door": "admitted", "n": 6}, {"week": "2026-11-23", "door": "demoted", "n": 2},
            {"week": "2026-11-23", "door": "routed", "n": 4}, {"week": "2026-11-23", "door": "unjudged", "n": 1},
            {"week": "2026-11-23", "door": "held", "n": 3}]
    (w,) = weekly(rows)
    assert w["dressing_rate"] == 2 / 8
    assert w["unjudged_share"] == 1 / 13  # held rows are not judged yet


def test_a_week_with_nothing_admitted_reads_none_not_zero():
    (w,) = weekly([{"week": "2026-11-30", "door": "routed", "n": 3}])
    assert w["dressing_rate"] is None and w["unjudged_share"] == 0.0
    assert weekly([]) == []
