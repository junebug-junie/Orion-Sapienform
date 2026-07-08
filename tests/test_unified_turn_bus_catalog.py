"""Bus catalog + schema registry alignment for unified Orion turn."""

from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"

UNIFIED_TURN_CHANNELS: dict[str, str] = {
    "orion:thought:request": "StanceReactRequestV1",
    "orion:thought:result:*": "ThoughtEventV1",
    "orion:thought:artifact": "ThoughtEventV1",
    "orion:harness:run:request": "HarnessRunRequestV1",
    "orion:harness:run:result:*": "HarnessRunV1",
    "orion:harness:run:artifact": "HarnessRunV1",
    "orion:harness:run:step": "HarnessRunStepV1",
    "orion:substrate:finalize_appraisal:request": "HarnessDraftMoleculeV1",
    "orion:substrate:finalize_appraisal:result:*": "SubstrateFinalizeAppraisalV1",
    "orion:harness:verdict:artifact": "HarnessVerdictMoleculeV1",
    "orion:substrate:turn_outcome": "HarnessTurnOutcomeMoleculeV1",
    "orion:substrate:post_turn_closure": "HarnessPostTurnClosureV1",
}


def _channel_index() -> dict[str, dict]:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8"))
    return {c["name"]: c for c in (doc.get("channels") or []) if c.get("name")}


def test_unified_turn_channels_exist_with_registry_schema_ids() -> None:
    channels = _channel_index()
    for name, schema_id in UNIFIED_TURN_CHANNELS.items():
        assert name in channels, f"missing channel catalog entry for {name!r}"
        entry = channels[name]
        assert entry["schema_id"] == schema_id
        assert schema_id in SCHEMA_REGISTRY, f"{schema_id!r} not in schema registry"
        assert resolve(schema_id) is SCHEMA_REGISTRY[schema_id].model


def test_harness_governor_produces_cortex_exec_requests() -> None:
    channels = _channel_index()
    entry = channels["orion:cortex:exec:request:background"]
    assert "orion-harness-governor" in entry["producer_services"]
    legacy = channels["orion:cortex:exec:request"]
    assert "orion-harness-governor" not in legacy["producer_services"]
    assert "orion-thought" in legacy["producer_services"]


def test_harness_governor_consumes_cortex_exec_results() -> None:
    channels = _channel_index()
    entry = channels["orion:exec:result:*"]
    assert "orion-harness-governor" in entry["consumer_services"]
