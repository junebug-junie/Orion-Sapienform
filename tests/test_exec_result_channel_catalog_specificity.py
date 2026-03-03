from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"


def _catalog_names() -> set[str]:
    raw = yaml.safe_load(CHANNELS_YAML.read_text()) or {}
    channels = raw.get("channels", [])
    return {entry.get("name") for entry in channels if isinstance(entry, dict) and entry.get("name")}


def test_exec_result_channels_have_specific_wildcards() -> None:
    names = _catalog_names()
    assert "orion:exec:result:LLMGatewayService:*" in names
    assert "orion:exec:result:PlannerReactService:*" in names
    assert "orion:exec:result:AgentChainService:*" in names
