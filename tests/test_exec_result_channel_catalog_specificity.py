from __future__ import annotations

from pathlib import Path

import yaml

from orion.core.bus.enforce import ChannelCatalogEnforcer


ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"


def _catalog_names() -> set[str]:
    raw = yaml.safe_load(CHANNELS_YAML.read_text()) or {}
    channels = raw.get("channels", [])
    return {entry.get("name") for entry in channels if isinstance(entry, dict) and entry.get("name")}


def test_exec_result_channels_have_specific_wildcards() -> None:
    names = _catalog_names()
    assert "orion:verb:result:*" in names
    assert "orion:exec:result:LLMGatewayService:*" in names
    assert "orion:exec:result:PlannerReactService:*" in names
    assert "orion:exec:result:AgentChainService:*" in names
    assert "orion:exec:result:PadRpc:*" in names
    assert "orion:exec:result:StateService:*" in names


def test_channel_enforcer_accepts_dedicated_verb_result_reply_channel() -> None:
    raw = yaml.safe_load(CHANNELS_YAML.read_text()) or {}
    channels = raw.get("channels", [])
    catalog = {entry["name"]: entry for entry in channels if isinstance(entry, dict) and entry.get("name")}
    enforcer = ChannelCatalogEnforcer(enforce=True, catalog=catalog)

    entry = enforcer.entry_for("orion:verb:result:corr-123:req-456")

    assert entry is not None
    assert entry["schema_id"] == "VerbResultV1"
