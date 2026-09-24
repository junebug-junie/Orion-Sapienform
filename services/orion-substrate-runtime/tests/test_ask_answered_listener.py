"""orion:ask:answered -> substrate EntityNodeV1 (walkway camera idea 3)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.cognitive_substrate import EntityNodeV1
from orion.schemas.ask import OrionAskAnsweredV1
from orion.substrate.adapters.vision_individual import (
    entity_node_id_for_individual,
    entity_type_for_kind,
    map_vision_individual_label_to_substrate,
)
from orion.substrate.store import InMemorySubstrateGraphStore


def _event(*, answer="the mail carrier", status="answered", source_kind="vision_individual", ask_id="ask-1", ref="ind-1"):
    return OrionAskAnsweredV1(
        ask_id=ask_id,
        status=status,
        answer=answer,
        answered_at=datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc),
        source_kind=source_kind,
        source_ref=ref,
    )


def _apply(event, store):
    return _apply_with_kind(event, store, ("person", "walkway"))


def _apply_with_kind(event, store, kind):
    """Run the handler with vision_individual lookup stubbed to ``kind``."""
    import app.ask_answered_listener as mod

    orig = mod.lookup_individual
    mod.lookup_individual = lambda _engine, _id: kind
    try:
        return mod.apply_answer_to_substrate(event, get_store=lambda: store, get_engine=lambda: object())
    finally:
        mod.lookup_individual = orig


def test_entity_type_mapping():
    assert entity_type_for_kind("person") == "person"
    assert entity_type_for_kind("Dog") == "animal"
    assert entity_type_for_kind("bicycle") == "vehicle"
    assert entity_type_for_kind("mailbox") == "mailbox"
    assert entity_type_for_kind(None) == "unknown"


def test_adapter_builds_one_user_asserted_entity():
    rec = map_vision_individual_label_to_substrate(
        individual_id="ind-1", label="  the  mail carrier ", kind="person", ask_id="ask-1",
        answered_at=datetime(2026, 9, 24, tzinfo=timezone.utc), stream_id="walkway",
    )
    (node,) = rec.nodes
    assert isinstance(node, EntityNodeV1)
    assert node.label == "the mail carrier"
    assert node.entity_type == "person"
    assert node.provenance.source_kind == "vision_individual"
    assert node.provenance.authority == "user_asserted"
    assert "ind-1" in node.provenance.evidence_refs
    assert node.metadata["individual_id"] == "ind-1"
    assert node.subject_ref == "vision_individual:ind-1"
    assert map_vision_individual_label_to_substrate(
        individual_id="ind-1", label="   ", kind=None, ask_id="a", answered_at=datetime.now(timezone.utc)
    ) is None


def test_answer_creates_entity_in_store():
    store = InMemorySubstrateGraphStore()
    out = _apply(_event(), store)
    assert out["outcome"] == "created"
    node = store.get_node_by_id(entity_node_id_for_individual("ind-1"))
    assert isinstance(node, EntityNodeV1)
    assert node.label == "the mail carrier"
    assert node.entity_type == "person"
    assert node.provenance.source_kind == "vision_individual"
    assert node.metadata["stream_id"] == "walkway"


def test_same_answer_twice_is_idempotent():
    store = InMemorySubstrateGraphStore()
    _apply(_event(), store)
    out = _apply(_event(ask_id="ask-2"), store)
    assert out["outcome"] == "updated"
    entities = [n for n in store.snapshot().nodes.values() if n.node_kind == "entity"]
    assert len(entities) == 1
    # Newest ask wins, not the first one (merge_node would have kept ask-1).
    assert entities[0].metadata["label_ask_id"] == "ask-2"
    assert entities[0].aliases == []


def test_capitalization_only_rename_is_kept():
    store = InMemorySubstrateGraphStore()
    _apply(_event(answer="rex"), store)
    out = _apply(_event(answer="Rex", ask_id="ask-2"), store)
    assert out["outcome"] == "relabelled"
    node = store.get_node_by_id(entity_node_id_for_individual("ind-1"))
    assert node.label == "Rex" and node.aliases == ["rex"]


def test_late_kind_fills_in_type_and_stream():
    store = InMemorySubstrateGraphStore()
    _apply_with_kind(_event(), store, (None, None))
    _apply_with_kind(_event(ask_id="ask-2"), store, ("dog", "walkway"))
    node = store.get_node_by_id(entity_node_id_for_individual("ind-1"))
    assert node.entity_type == "animal"
    assert node.metadata["stream_id"] == "walkway"


def test_rename_updates_label_and_keeps_old_as_alias():
    store = InMemorySubstrateGraphStore()
    _apply(_event(answer="the mail carrier"), store)
    out = _apply(_event(answer="Dana", ask_id="ask-2"), store)
    assert out["outcome"] == "relabelled"
    node = store.get_node_by_id(entity_node_id_for_individual("ind-1"))
    assert node.label == "Dana"
    assert "the mail carrier" in node.aliases
    assert node.metadata["label_ask_id"] == "ask-2"


def test_two_individuals_with_same_name_stay_separate():
    store = InMemorySubstrateGraphStore()
    _apply(_event(answer="Bob", ref="ind-1"), store)
    _apply(_event(answer="Bob", ref="ind-2", ask_id="ask-2"), store)
    entities = [n for n in store.snapshot().nodes.values() if n.node_kind == "entity"]
    assert len(entities) == 2


def test_missing_kind_still_writes_unknown_type():
    store = InMemorySubstrateGraphStore()
    out = _apply_with_kind(_event(), store, (None, None))
    assert out["entity_type"] == "unknown"


@pytest.mark.parametrize(
    "event",
    [
        _event(status="dismissed", answer=None),
        _event(source_kind="curiosity_prior"),
        _event(answer="   "),
    ],
)
def test_ignored_events_write_nothing(event):
    store = InMemorySubstrateGraphStore()
    assert _apply(event, store) is None
    assert not store.snapshot().nodes


def test_no_store_is_fail_open():
    import app.ask_answered_listener as mod

    assert mod.apply_answer_to_substrate(_event(), get_store=lambda: None, get_engine=lambda: None) is None


def test_lookup_individual_fail_open_on_db_error():
    from app.ask_answered_listener import lookup_individual

    class _BoomEngine:
        def connect(self):
            raise RuntimeError("relation vision_individual does not exist")

    assert lookup_individual(_BoomEngine(), "ind-1") == (None, None)
    assert lookup_individual(None, "ind-1") == (None, None)


@pytest.mark.asyncio
async def test_bus_message_end_to_end_and_wrong_kind_ignored(monkeypatch):
    import app.ask_answered_listener as mod

    monkeypatch.setattr(mod, "lookup_individual", lambda _e, _i: ("dog", "walkway"))
    store = InMemorySubstrateGraphStore()
    env = BaseEnvelope(
        kind="orion.ask.answered.v1",
        source=ServiceRef(name="orion-hub"),
        payload=_event(answer="Rex").model_dump(mode="json"),
    )
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=env, error=None)
    out = await mod._handle_bus_message(bus, {"data": b"x"}, get_store=lambda: store, get_engine=lambda: None)
    assert out["entity_type"] == "animal" and out["label"] == "Rex"

    wrong = env.model_copy(update={"kind": "something.else"})
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=wrong, error=None)
    assert await mod._handle_bus_message(bus, {"data": b"x"}, get_store=lambda: store, get_engine=lambda: None) is None


def test_settings_defaults():
    from app.settings import Settings

    s = Settings(POSTGRES_URI="postgresql://orion:orion@localhost:5432/orion")
    assert s.channel_ask_answered == "orion:ask:answered"
    assert s.enable_ask_answered_listener is True
