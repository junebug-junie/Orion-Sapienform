"""Recall diagnostic schemas are registered and validate production shapes."""

from __future__ import annotations

from pathlib import Path

import yaml

from orion.core.contracts.recall import (
    RecallAdapterDiagnosticsV1,
    RecallDebugV1,
    RecallVectorPolicyPathV1,
    RecallVectorPolicyV1,
)
from orion.schemas.registry import _REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"


def test_recall_diagnostic_schemas_in_registry() -> None:
    for schema_id in (
        "RecallDebugV1",
        "RecallVectorPolicyV1",
        "RecallVectorPolicyPathV1",
        "RecallSourceGatingV1",
        "RecallAdapterDiagnosticsV1",
    ):
        assert schema_id in _REGISTRY
        assert resolve(schema_id) is _REGISTRY[schema_id]


def test_vector_policy_path_roundtrip() -> None:
    raw = {
        "allowed": False,
        "reason": "disabled_profile_vector_top_k_zero",
        "profile": "assist.light.v1",
        "vector_top_k": 0,
        "RECALL_ENABLE_VECTOR": True,
        "path": "main",
    }
    model = RecallVectorPolicyPathV1.model_validate(raw)
    dumped = model.model_dump(by_alias=True)
    assert dumped["RECALL_ENABLE_VECTOR"] is True
    assert dumped["reason"] == "disabled_profile_vector_top_k_zero"


def test_recall_debug_validates_vector_policy_subset() -> None:
    debug = RecallDebugV1.model_validate(
        {
            "vector_policy": {
                "main": {
                    "allowed": False,
                    "reason": "disabled_global",
                    "profile": "reflect.v1",
                    "vector_top_k": 8,
                    "RECALL_ENABLE_VECTOR": False,
                    "path": "main",
                }
            },
            "source_gating": {"vector": "disabled_by_profile_or_global", "sql_timeline": "enabled"},
        }
    )
    assert debug.vector_policy is not None
    assert debug.vector_policy.main is not None
    assert debug.vector_policy.main.allowed is False


def test_recall_telemetry_channel_still_recall_decision_v1() -> None:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8"))
    channels = doc.get("channels") or []
    entry = next(c for c in channels if c.get("name") == "orion:recall:telemetry")
    assert entry["schema_id"] == "RecallDecisionV1"
    assert "RecallDecisionV1" in _REGISTRY


def test_adapter_diagnostics_shape() -> None:
    RecallAdapterDiagnosticsV1.model_validate(
        {
            "recall_fragments_seen": 2,
            "recall_fragments_mapped": 1,
            "recall_fragments_dropped": 1,
            "dropped_counts_by_reason": {"unsupported_source": 1},
            "mapped_counts_by_source": {"sql_timeline": 1},
            "mapped_counts_by_node_type": {"event.chat_timeline": 1},
            "original_sources_seen": ["sql_timeline", "vector"],
        }
    )
