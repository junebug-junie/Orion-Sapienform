"""The routing-decision record must be resolvable and carry the gate's effect.

``chat_reflective_lane_threshold`` is the one knob Orion can turn about its own
behaviour, and the gate it drives left no trace anywhere -- its inputs went into
an in-memory options dict nothing read. That absence is why the mutation loop
justified routing changes with graph-review telemetry the threshold cannot
affect. This record is the missing evidence.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.routing_decision import RoutingDecisionRecordV1

REPO_ROOT = Path(__file__).resolve().parents[3]


def _record(**overrides) -> RoutingDecisionRecordV1:
    base = dict(
        execution_depth_before_gate=2,
        execution_depth=0,
        decision_confidence=0.5,
        routing_threshold=0.58,
        gate_demoted=True,
    )
    base.update(overrides)
    return RoutingDecisionRecordV1(**base)


def test_the_schema_resolves_from_the_registry_the_bus_actually_reads() -> None:
    """This repo has two registry dicts and only one is consulted at runtime.

    Registering in SCHEMA_REGISTRY alone leaves resolve() raising Unknown
    schema_id, so a published envelope fails validation at the consumer. That
    happened while writing this patch and was caught only by calling resolve().
    """
    assert resolve("RoutingDecisionRecordV1") is RoutingDecisionRecordV1
    assert SCHEMA_REGISTRY["RoutingDecisionRecordV1"].kind == "routing.decision.record.v1"


def test_the_channel_declares_the_same_schema_and_a_real_consumer() -> None:
    channels = yaml.safe_load((REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text())
    entry = next(c for c in channels["channels"] if c["name"] == "orion:routing:decision")

    assert entry["schema_id"] == "RoutingDecisionRecordV1"
    assert entry["producer_services"] == ["orion-cortex-orch"]
    # A producer with no consumer is a dead contract; sql-writer persists it.
    assert "orion-sql-writer" in entry["consumer_services"]


def test_a_demotion_is_visible_without_comparing_to_anything_else() -> None:
    """gate_demoted is the outcome a routing mutation claims to move."""
    demoted = _record()
    assert demoted.gate_demoted is True
    assert demoted.execution_depth_before_gate == 2
    assert demoted.execution_depth == 0


def test_both_sides_of_the_comparison_are_recorded() -> None:
    """Orion controls the threshold, not the confidence.

    Storing only the outcome would make a threshold change and a drift in
    confidence indistinguishable after the fact.
    """
    r = _record(
        decision_confidence=0.61,
        routing_threshold=0.58,
        gate_demoted=False,
        execution_depth=2,
    )
    assert r.decision_confidence == 0.61
    assert r.routing_threshold == 0.58


def test_the_record_carries_no_message_content() -> None:
    """It answers how Orion decided, not what was said, so it is safe to query."""
    fields = set(RoutingDecisionRecordV1.model_fields)
    for leak in ("text", "message", "user_message", "raw_user_text", "content", "prompt"):
        assert leak not in fields


def test_out_of_range_values_are_rejected_not_silently_clamped() -> None:
    with pytest.raises(Exception):
        _record(decision_confidence=1.5)
    with pytest.raises(Exception):
        _record(routing_threshold=-0.1)
