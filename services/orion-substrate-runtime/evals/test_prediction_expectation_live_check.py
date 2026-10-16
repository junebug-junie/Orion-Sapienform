"""The rung-6 live check's pure summarizer flags flat/thin data and reports what
the tuner would propose, on synthetic readings."""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_prediction_expectation_live_check import summarize  # noqa: E402

T0 = datetime(2026, 10, 16, tzinfo=timezone.utc)


def _timed(values):
    return [(T0 + timedelta(minutes=i), v) for i, v in enumerate(values)]


def test_flat_domain_is_flagged() -> None:
    s = summarize("execution", _timed([0.0] * 600), None)
    assert s["coverage"] and not s["not_flat"]
    assert s["share_at_min"] == 1.0


def test_thin_domain_reports_insufficient_readings() -> None:
    s = summarize("chat", _timed([0.01, 0.02] * 20), None)
    assert s["coverage"] is False
    assert s["assessment"] == "insufficient_readings"


def test_steady_domain_reports_a_candidate_against_the_live_default() -> None:
    r = random.Random(2)
    s = summarize("execution", _timed([abs(0.02 + r.gauss(0, 0.005)) for _ in range(700)]), None)
    assert s["live_is_default"] is True
    assert isinstance(s["assessment"], dict)
    assert s["assessment"]["fit_gain_lower_bound"] > 0.0
