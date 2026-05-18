"""Bus catalog entries for Mind LLM synthesis (intake + per-run reply channels)."""

from __future__ import annotations

from pathlib import Path

import pytest

from orion.core.bus.catalog_loader import load_channel_catalog
from orion.core.bus.enforce import ChannelCatalogEnforcer

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"


def test_mind_llm_reply_channel_pattern_cataloged() -> None:
    catalog = load_channel_catalog()
    entry = catalog.get("orion:mind:llm:reply:*")
    assert entry is not None, "orion:mind:llm:reply:* must be in channel catalog"
    assert entry.get("schema_id") == "ChatResultPayload"
    assert "orion-llm-gateway" in (entry.get("producer_services") or [])
    assert "orion-mind" in (entry.get("consumer_services") or [])


def test_mind_llm_reply_concrete_channel_matches_wildcard() -> None:
    enforcer = ChannelCatalogEnforcer(enforce=True, catalog=load_channel_catalog())
    channel = "orion:mind:llm:reply:550e8400-e29b-41d4-a716-446655440000"
    enforcer.validate(channel)
    entry = enforcer.entry_for(channel)
    assert entry is not None
    assert entry.get("schema_id") == "ChatResultPayload"


def test_orion_mind_produces_llm_gateway_intake() -> None:
    catalog = load_channel_catalog()
    entry = catalog["orion:exec:request:LLMGatewayService"]
    producers = entry.get("producer_services") or []
    assert "orion-mind" in producers


def test_mind_llm_intake_literal_in_catalog() -> None:
    catalog = load_channel_catalog()
    assert "orion:exec:request:LLMGatewayService" in catalog


@pytest.mark.parametrize(
    "channel",
    [
        "orion:mind:llm:reply:abc",
        "orion:exec:request:LLMGatewayService",
        "orion:mind:artifact",
    ],
)
def test_mind_llm_channels_pass_prefix_guardrail(channel: str) -> None:
    assert channel.startswith("orion:")
