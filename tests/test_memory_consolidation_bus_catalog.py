from pathlib import Path

import yaml

from orion.schemas.memory_consolidation import ChatHistorySparkMetaPatchV1, MemoryTurnPersistedV1
from orion.schemas.registry import resolve

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = {
    "orion:memory:turn:persisted": ("MemoryTurnPersistedV1", "memory.turn.persisted.v1"),
    "orion:chat:history:spark_meta:patch": ("ChatHistorySparkMetaPatchV1", "chat.history.spark_meta.patch.v1"),
}


def _channels():
    doc = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text(encoding="utf-8")) or {}
    return {e["name"]: e for e in doc.get("channels") or [] if isinstance(e, dict) and "name" in e}


def test_memory_consolidation_schemas_registered():
    assert resolve("MemoryTurnPersistedV1") is MemoryTurnPersistedV1
    assert resolve("ChatHistorySparkMetaPatchV1") is ChatHistorySparkMetaPatchV1


def test_memory_consolidation_channels_cataloged():
    ch = _channels()
    for name, (schema_id, message_kind) in CHANNELS.items():
        entry = ch[name]
        assert entry["schema_id"] == schema_id
        assert entry["message_kind"] == message_kind


def test_memory_consolidation_cortex_and_llm_channels_cataloged():
    ch = _channels()
    cortex = ch["orion:cortex:request"]
    assert "orion-memory-consolidation" in cortex["producer_services"]
    llm_req = ch["orion:exec:request:LLMGatewayService"]
    assert "orion-memory-consolidation" in llm_req["producer_services"]
    llm_res = ch["orion:exec:result:LLMGatewayService:*"]
    assert "orion-memory-consolidation" in llm_res["consumer_services"]
