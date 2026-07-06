"""Phase H — efficacy metric unit tests (pure reductions)."""
from __future__ import annotations

from orion.reverie.efficacy import (
    action_usefulness_rate,
    pressure_discharge_rate,
    recall_delta,
)


def test_pressure_discharge_rate_counts_drops():
    # 2 of 3 chains discharged (after < before).
    assert pressure_discharge_rate([0.9, 0.8, 0.5], [0.4, 0.9, 0.2]) == 2 / 3


def test_pressure_discharge_rate_none_on_empty():
    assert pressure_discharge_rate([], []) is None


def test_pressure_discharge_rate_aligns_on_shorter():
    assert pressure_discharge_rate([0.9, 0.8], [0.1]) == 1.0


def test_action_usefulness_rate_case_insensitive():
    assert action_usefulness_rate(["Useful", "neutral", "HELPED"]) == 2 / 3


def test_action_usefulness_rate_none_on_empty():
    assert action_usefulness_rate([]) is None


def test_recall_delta_reports_wins_as_negative():
    d = recall_delta(
        latency_ms_before=120.0, latency_ms_after=80.0,
        graph_size_before=1000, graph_size_after=900,
    )
    assert d.latency_ms_delta == -40.0  # faster
    assert d.graph_size_delta == -100  # smaller
