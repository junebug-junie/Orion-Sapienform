"""Subject routing for concept induction must align with Graph autonomy (relationship vs orion)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.spark.concept_induction.identity import RELATIONSHIP_SUBJECT, SELF_SUBJECT, resolve_subject_identity


def _env(payload: dict, *, source_name: str = "orion-hub") -> BaseEnvelope:
    return BaseEnvelope(
        kind="chat.history",
        source=ServiceRef(name=source_name, node="n1", version="1"),
        correlation_id=uuid4(),
        created_at=datetime.now(timezone.utc),
        payload=payload,
    )


def test_chat_history_turn_resolves_to_relationship_not_orion_hub() -> None:
    env = _env(
        {
            "prompt": "hello",
            "response": "hi there",
            "source": "hub_ws",
            "session_id": "s1",
        },
        source_name="orion-hub",
    )
    assert resolve_subject_identity(env, "orion:chat:history:turn") == RELATIONSHIP_SUBJECT


def test_orion_in_non_hub_source_still_self_for_legacy_envelopes() -> None:
    env = _env({"content": "internal note"}, source_name="orion-cortex-exec")
    assert resolve_subject_identity(env, "orion:internal:note") == SELF_SUBJECT


def test_explicit_subject_wins() -> None:
    env = _env({"subject": "orion", "prompt": "a", "response": "b"}, source_name="orion-hub")
    assert resolve_subject_identity(env, "orion:chat:history:turn") == "orion"
