"""Bus catalog alignment for stance_context_prepare (unified-turn latency L4)."""

from __future__ import annotations

from pathlib import Path

import yaml

from orion.core.bus.enforce import ChannelCatalogEnforcer
from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.stance_context_prepare import (
    STANCE_CONTEXT_PREPARE_REQUEST_KIND,
    STANCE_CONTEXT_PREPARE_RESULT_KIND,
    STANCE_CONTEXT_PREPARE_RESULT_PREFIX,
    StanceContextPrepareRequestV1,
    StanceContextPrepareResultV1,
    stance_context_prepare_channel,
)

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"
EXEC_COMPOSE = ROOT / "services" / "orion-cortex-exec" / "docker-compose.yml"


def _channels() -> dict:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8")) or {}
    return {c["name"]: c for c in doc.get("channels") or []}


def _exec_request_channels() -> list[str]:
    """Every exec request channel a cortex-exec container listens on."""
    class _Loader(yaml.SafeLoader):
        pass

    # Compose's `!override` / `!reset` tags.
    _Loader.add_multi_constructor("!", lambda loader, suffix, node: None)
    compose = yaml.load(EXEC_COMPOSE.read_text(encoding="utf-8"), Loader=_Loader) or {}
    found = {"orion:cortex:exec:request"}
    for svc in (compose.get("services") or {}).values():
        env = svc.get("environment") or {}
        value = env.get("CHANNEL_EXEC_REQUEST") if isinstance(env, dict) else None
        if isinstance(value, str) and not value.startswith("$"):
            found.add(value)
    return sorted(found)


def test_every_exec_lane_has_a_registered_prepare_channel() -> None:
    channels = _channels()
    lanes = _exec_request_channels()
    assert len(lanes) >= 4
    for exec_channel in lanes:
        prepare = stance_context_prepare_channel(exec_channel)
        assert prepare is not None, exec_channel
        entry = channels.get(prepare)
        assert entry is not None, prepare
        assert entry["schema_id"] == "StanceContextPrepareRequestV1"
        assert entry["message_kind"] == STANCE_CONTEXT_PREPARE_REQUEST_KIND
        assert entry["producer_services"] == ["orion-thought"]
        assert entry["consumer_services"] == ["orion-cortex-exec"]
        assert entry["single_consumer"] is True


def test_result_channel_resolves_to_the_result_schema() -> None:
    enforcer = ChannelCatalogEnforcer()
    entry = enforcer.entry_for(f"{STANCE_CONTEXT_PREPARE_RESULT_PREFIX}:abc-123")
    assert entry is not None
    assert entry["schema_id"] == "StanceContextPrepareResultV1"
    assert entry["message_kind"] == STANCE_CONTEXT_PREPARE_RESULT_KIND


def test_schemas_registered_in_both_registries() -> None:
    for schema_id, model, kind in (
        ("StanceContextPrepareRequestV1", StanceContextPrepareRequestV1, STANCE_CONTEXT_PREPARE_REQUEST_KIND),
        ("StanceContextPrepareResultV1", StanceContextPrepareResultV1, STANCE_CONTEXT_PREPARE_RESULT_KIND),
    ):
        assert resolve(schema_id) is model
        assert SCHEMA_REGISTRY[schema_id].model is model
        assert SCHEMA_REGISTRY[schema_id].kind == kind
