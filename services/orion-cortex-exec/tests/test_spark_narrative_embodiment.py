from __future__ import annotations

from datetime import datetime, timezone

from app.spark_narrative import spark_embodiment_hint, spark_embodiment_narrative
from orion.schemas.telemetry.spark import SparkStateSnapshotV1

NOW = datetime(2026, 7, 12, 23, 0, tzinfo=timezone.utc)


def _snapshot(**overrides) -> SparkStateSnapshotV1:
    base = dict(
        source_service="spark-introspector",
        source_node="athena",
        producer_boot_id="boot-1",
        seq=1,
        snapshot_ts=NOW,
    )
    base.update(overrides)
    return SparkStateSnapshotV1(**base)


def test_embodiment_hint_names_real_node() -> None:
    snap = _snapshot(dominant_node="node:atlas", dominant_node_reason="node gpu_pressure is elevated")

    hint = spark_embodiment_hint(snap)

    assert hint == {
        "dominant_node": "node:atlas",
        "dominant_node_reason": "node gpu_pressure is elevated",
    }


def test_embodiment_hint_none_when_no_qualifying_node() -> None:
    snap = _snapshot()  # dominant_node/dominant_node_reason default to None

    hint = spark_embodiment_hint(snap)

    assert hint == {
        "dominant_node": "none",
        "dominant_node_reason": "no node currently salient",
    }


def test_embodiment_narrative_names_node_without_prefix() -> None:
    snap = _snapshot(dominant_node="node:atlas", dominant_node_reason="node gpu_pressure is elevated")

    narrative = spark_embodiment_narrative(snap)

    assert "atlas" in narrative
    assert "node:atlas" not in narrative  # human-readable, prefix stripped
    assert "node gpu_pressure is elevated" in narrative
    assert "not a mood" in narrative  # explicit against fabrication/category error


def test_embodiment_narrative_honest_when_no_node() -> None:
    snap = _snapshot()

    narrative = spark_embodiment_narrative(snap)

    assert "No single hardware node" in narrative
    # Must not fabricate a node name when none is present.
    assert "atlas" not in narrative.lower()
    assert "circe" not in narrative.lower()
    assert "athena" not in narrative.lower()
    assert "prometheus" not in narrative.lower()


def test_embodiment_narrative_handles_missing_reason() -> None:
    snap = _snapshot(dominant_node="node:circe", dominant_node_reason=None)

    narrative = spark_embodiment_narrative(snap)

    assert "circe" in narrative
    # No dangling empty parens when reason is absent.
    assert "()" not in narrative
