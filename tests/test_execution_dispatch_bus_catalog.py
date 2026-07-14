"""Bus catalog coverage for orion-execution-dispatch-runtime as a cortex-exec producer."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"


def _channel_entry(name: str) -> dict:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8")) or {}
    for entry in doc.get("channels") or []:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    raise AssertionError(f"Missing channel catalog entry: {name}")


def test_execution_dispatch_runtime_is_background_exec_request_producer() -> None:
    entry = _channel_entry("orion:cortex:exec:request:background")
    assert entry["schema_id"] == "CortexExecRequestPayload"
    assert entry["consumer_services"] == ["orion-cortex-exec"]
    assert "orion-execution-dispatch-runtime" in (entry.get("producer_services") or [])


def test_execution_dispatch_runtime_is_action_outcome_producer() -> None:
    entry = _channel_entry("orion:autonomy:action:outcome")
    assert entry["schema_id"] == "ActionOutcomeEmitV1"
    assert "orion-sql-writer" in entry["consumer_services"]
    assert "orion-spark-concept-induction" in (entry.get("producer_services") or [])
    assert "orion-execution-dispatch-runtime" in (entry.get("producer_services") or [])


def test_concept_induction_is_recall_service_request_producer() -> None:
    """recall.query.readonly (P4): the autonomy tick issues an inline recall RPC
    from orion-spark-concept-induction's ConceptWorker before the readonly fetch."""
    entry = _channel_entry("orion:exec:request:RecallService")
    assert entry["schema_id"] == "RecallQueryV1"
    assert entry["consumer_services"] == ["orion-recall"]
    assert "orion-spark-concept-induction" in (entry.get("producer_services") or [])
