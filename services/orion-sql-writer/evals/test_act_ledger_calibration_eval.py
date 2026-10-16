"""Eval: the act ledger's binary EffectPosterior converges to the real outcome rate.

Fixture: 0/1 outcome sequences rebuilt read-only from 60 days of live
chat_history_log on 2026-10-16 (act_ledger_calibration.py --live), no text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import act_ledger_calibration as cal  # noqa: E402


def test_posterior_converges_to_empirical_rate_on_recorded_outcomes():
    seqs = json.loads(cal.FIXTURE.read_text())
    assert seqs and all(len(v) >= 100 for v in seqs.values())
    for name, seq in seqs.items():
        r = cal.replay(seq)
        assert abs(r["final_mean"] - r["empirical"]) < 0.02, (name, r)
        # Not saturated: the horizons were chosen so the outcome is near max entropy.
        assert 0.2 < r["empirical"] < 0.8, (name, r)


def test_posterior_tracks_a_skewed_rate():
    seq = ([1] * 9 + [0]) * 30  # a 90% producer, like paintings
    r = cal.replay(seq)
    assert abs(r["final_mean"] - 0.9) < 0.02
    assert r["brier"] < r["base_brier"], "a skewed rate must beat the cold coin-flip prior"
