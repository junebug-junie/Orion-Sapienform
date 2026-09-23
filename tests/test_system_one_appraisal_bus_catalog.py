"""Bus catalog alignment for the operational System One appraisal channel."""

from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.system_one_appraisal import (
    SYSTEM_ONE_APPRAISAL_CHANNEL,
    SYSTEM_ONE_APPRAISAL_KIND,
    SystemOneAppraisalFrameV1,
)

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"


def test_system_one_appraisal_channel_registered() -> None:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8")) or {}
    channels = doc.get("channels") or []
    entry = next((c for c in channels if c.get("name") == SYSTEM_ONE_APPRAISAL_CHANNEL), None)
    assert entry is not None
    assert entry["schema_id"] == "SystemOneAppraisalFrameV1"
    assert entry["message_kind"] == SYSTEM_ONE_APPRAISAL_KIND
    assert "orion-substrate-runtime" in entry["producer_services"]
    assert "SystemOneAppraisalFrameV1" in SCHEMA_REGISTRY
    assert resolve("SystemOneAppraisalFrameV1") is SystemOneAppraisalFrameV1
    assert SCHEMA_REGISTRY["SystemOneAppraisalFrameV1"].model is SystemOneAppraisalFrameV1
    assert SCHEMA_REGISTRY["SystemOneAppraisalFrameV1"].kind == SYSTEM_ONE_APPRAISAL_KIND
