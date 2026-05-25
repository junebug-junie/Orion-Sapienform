from pathlib import Path

from orion.attention.field_attention.policy import load_attention_policy
from orion.attention.field_attention.scoring import (
    clamp01,
    compute_salience,
    urgency_score,
    weighted_pressure,
)

REPO = Path(__file__).resolve().parents[1]
POLICY = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")


def test_execution_load_raises_node_pressure() -> None:
    pressure, dominant = weighted_pressure(
        {"execution_load": 1.0, "reasoning_load": 0.35},
        POLICY.node_channel_weights,
    )
    assert pressure > 0.5
    assert "execution_load" in dominant


def test_failure_pressure_high_urgency() -> None:
    score = urgency_score({"failure_pressure": 1.0}, POLICY.node_channel_weights)
    assert score >= 0.9


def test_availability_reduces_pressure() -> None:
    pressure, _ = weighted_pressure({"availability": 1.0}, POLICY.node_channel_weights)
    assert pressure < 0.5


def test_expected_offline_suppression_reduces_pressure() -> None:
    pressure, _ = weighted_pressure(
        {"expected_offline_suppression": 1.0},
        POLICY.node_channel_weights,
    )
    assert pressure < 0.4


def test_capability_execution_pressure_raises_salience() -> None:
    pressure, _ = weighted_pressure(
        {"execution_pressure": 1.0},
        POLICY.capability_channel_weights,
    )
    assert pressure > 0.5


def test_capability_available_capacity_reduces() -> None:
    pressure, _ = weighted_pressure(
        {"available_capacity": 1.0},
        POLICY.capability_channel_weights,
    )
    assert pressure < 0.5


def test_scores_clamped() -> None:
    assert clamp01(2.0) == 1.0
    assert clamp01(-1.0) == 0.0
    salience = compute_salience(
        pressure_score=2.0,
        novelty_score=0.0,
        urgency_score=0.0,
        confidence_score=0.0,
        weights=POLICY.weights,
    )
    assert salience <= 1.0
