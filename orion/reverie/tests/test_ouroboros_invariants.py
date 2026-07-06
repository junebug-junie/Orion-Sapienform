"""Cross-cutting ouroboros invariants for the reverie/dream/compaction weave.

The plan's non-negotiable safety property: no process reads its own output kind.
These tests assert it at the contract level (channels.yaml) across the whole
weave, plus that every weave kind is registered and resolvable — a single place
that fails if a future patch wires a self-subscribing loop or drops a schema.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve

_REPO = Path(__file__).resolve().parents[3]
_CHANNELS = _REPO / "orion" / "bus" / "channels.yaml"

# The weave's channels and their canonical schema_id.
WEAVE_CHANNELS = {
    "orion:reverie:thought": "SpontaneousThoughtV1",
    "orion:reverie:chain": "ReverieChainV1",
    "orion:dream:compaction-request": "CompactionRequestV1",
    "orion:dream:compaction-delta": "MemoryCompactionDeltaV1",
    "orion:reverie:resonance-alert": "ResonanceAlertV1",
}


def _weave_channel_entries() -> dict[str, dict]:
    doc = yaml.safe_load(_CHANNELS.read_text(encoding="utf-8"))

    def walk(o):
        if isinstance(o, dict):
            if o.get("name") in WEAVE_CHANNELS:
                yield o
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)

    return {e["name"]: e for e in walk(doc)}


def test_every_weave_kind_is_registered_and_resolvable():
    for schema_id in WEAVE_CHANNELS.values():
        assert schema_id in SCHEMA_REGISTRY, f"{schema_id} not in SCHEMA_REGISTRY"
        assert resolve(schema_id).__name__ == schema_id  # runtime _REGISTRY path too


def test_no_process_reads_its_own_output_kind():
    """A producer of a weave channel must never also be listed as a consumer of
    that same channel — that would be a process reading its own output kind."""
    entries = _weave_channel_entries()
    for name in WEAVE_CHANNELS:
        entry = entries.get(name)
        assert entry is not None, f"weave channel {name} missing from channels.yaml"
        producers = set(entry.get("producer_services") or [])
        consumers = set(entry.get("consumer_services") or [])
        overlap = producers & consumers
        assert not overlap, f"{name}: {overlap} both produces and consumes its own kind"


def test_dangerous_channels_have_no_live_consumer():
    """The memory-touching / dispatch-adjacent channels stay dead-ended: no
    service consumes the compaction delta or the request queue (applier is gated
    off, Phase G). '*' or a concrete consumer here would be a live apply path."""
    entries = _weave_channel_entries()
    for name in ("orion:dream:compaction-request", "orion:dream:compaction-delta"):
        consumers = entries[name].get("consumer_services") or []
        assert consumers == [], f"{name} must have no consumer while G is gated off"


def test_channel_schema_ids_match_the_weave_contract():
    entries = _weave_channel_entries()
    for name, schema_id in WEAVE_CHANNELS.items():
        assert entries[name].get("schema_id") == schema_id
