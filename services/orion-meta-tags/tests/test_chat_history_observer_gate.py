"""Regression test for the chat.history observer-gate fix.

Commit aaf66d58 ("Enforce metacog draft/enrich patch contracts") moved
should_route_to_triage's Juniper-observer check in front of the
chat.history/social.turn.stored.v1 branch as an apparent side effect of an
unrelated refactor. ChatHistoryTurnV1 (orion-hub's real chat.history
payload) has no `observer` field, so this unconditionally skipped every
real chat turn for ~6 months. This test locks in the fix: chat.history
bypasses that gate; the generic collapse-mirror path still requires it.

app.main can't be imported directly in this environment -- a pre-existing
numpy/spacy/thinc binary incompatibility in the shared test venv (unrelated
to this change, confirmed via traceback in earlier sessions) blocks it.
Mocking spacy out of sys.modules before import sidesteps that entirely,
since the failure is inside spacy's own C-extension import chain, not
anything this test exercises.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(SERVICE_ROOT)]


@pytest.fixture
def main_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORION_BUS_URL", "redis://example.test/0")

    # A fake entity with real `.text` content (not just an empty ents=[])
    # so the extraction line (`entities = [ent.text for ent in doc.ents]`)
    # is actually exercised, not just left permanently dead under the mock.
    fake_doc = SimpleNamespace(ents=[SimpleNamespace(text="Circe")])
    fake_nlp = MagicMock(return_value=fake_doc)
    fake_spacy = MagicMock()
    fake_spacy.load.return_value = fake_nlp
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)

    # monkeypatch.delitem (not plain `del`) so teardown restores whatever
    # was in sys.modules before this fixture ran -- a bare `del` here would
    # leave the fake-spacy-backed module installed for any later test in the
    # same pytest process that does a plain `import app.main`.
    monkeypatch.delitem(sys.modules, "app.main", raising=False)
    monkeypatch.delitem(sys.modules, "app.settings", raising=False)
    import app.main as m

    m.meta_tagger = MagicMock()
    m.meta_tagger.bus.publish = AsyncMock()
    m.settings.RECALL_FALKOR_TAG_ENTITY_ENABLED = False
    return m


def _envelope(main_module, *, kind: str, payload: dict):
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    return BaseEnvelope(
        kind=kind,
        source=ServiceRef(name="hub", node="athena", version="0.4.0"),
        payload=payload,
    )


@pytest.mark.asyncio
async def test_chat_history_turn_with_no_observer_still_publishes(main_module) -> None:
    """The actual regression: a real hub chat.history payload (no `observer`
    field, matching ChatHistoryTurnV1's real schema) must NOT be skipped."""
    envelope = _envelope(
        main_module,
        kind="chat.history",
        payload={"session_id": "sess-1", "prompt": "hi", "response": "hello"},
    )
    await main_module.handle_triage_event(envelope)
    main_module.meta_tagger.bus.publish.assert_awaited_once()
    call_args = main_module.meta_tagger.bus.publish.call_args
    assert call_args[0][0] == main_module.settings.CHANNEL_EVENTS_TAGGED_CHAT


@pytest.mark.asyncio
async def test_generic_collapse_event_without_observer_is_still_skipped(main_module) -> None:
    """The generic (non-chat) collapse-mirror path must still require a
    Juniper observer -- this fix does not weaken that gate."""
    envelope = _envelope(
        main_module,
        kind="collapse.mirror.entry",
        payload={"id": "c1", "text": "some collapse text", "observer": None},
    )
    await main_module.handle_triage_event(envelope)
    main_module.meta_tagger.bus.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_generic_collapse_event_with_juniper_observer_still_publishes(main_module) -> None:
    envelope = _envelope(
        main_module,
        kind="collapse.mirror.entry",
        payload={"id": "c2", "text": "some collapse text", "observer": "Juniper"},
    )
    await main_module.handle_triage_event(envelope)
    main_module.meta_tagger.bus.publish.assert_awaited_once()
    call_args = main_module.meta_tagger.bus.publish.call_args
    assert call_args[0][0] == main_module.settings.CHANNEL_EVENTS_TAGGED


@pytest.mark.asyncio
async def test_chat_history_turn_writes_to_falkor_when_flag_enabled(main_module, monkeypatch) -> None:
    """Closes the coverage gap noted in PR #1180: the asyncio.to_thread
    wiring for the Falkor write was previously untestable in this
    environment. Confirms it actually fires end-to-end for a real
    chat.history turn, now that app.main can be imported via the spacy
    mock above."""
    from orion.graph.falkor_client import RecordingFalkorClient

    fake_client = RecordingFalkorClient()
    main_module.settings.RECALL_FALKOR_TAG_ENTITY_ENABLED = True
    monkeypatch.setattr(main_module, "_falkor_client", fake_client)

    envelope = _envelope(
        main_module,
        kind="chat.history",
        payload={"session_id": "sess-2", "prompt": "hi Circe", "response": "hello"},
    )
    await main_module.handle_triage_event(envelope)

    main_module.meta_tagger.bus.publish.assert_awaited_once()
    assert fake_client.calls, "expected at least one Cypher write to the Falkor client"
    assert any("MERGE (t:ChatTurn" in cypher for cypher, _ in fake_client.calls)
    # The fake NER doc (see main_module fixture) returns one entity, "Circe".
    # Confirms the extraction line actually feeds real content through to
    # the Falkor write, not just an always-empty ents=[] that would pass
    # even if `ent.text` were swapped for the wrong attribute.
    entity_calls = [params for cypher, params in fake_client.calls if "MENTIONS_ENTITY" in cypher]
    assert entity_calls and "circe" in entity_calls[0]["names"]


@pytest.mark.asyncio
async def test_social_turn_stored_bypasses_gate_like_chat_history(main_module) -> None:
    """social.turn.stored.v1 shares handle_triage_event's chat branch (see
    module docstring) -- must not regress independently of chat.history."""
    envelope = _envelope(
        main_module,
        kind="social.turn.stored.v1",
        payload={"session_id": "sess-3", "prompt": "hi", "response": "hello"},
    )
    await main_module.handle_triage_event(envelope)
    main_module.meta_tagger.bus.publish.assert_awaited_once()
    call_args = main_module.meta_tagger.bus.publish.call_args
    assert call_args[0][0] == main_module.settings.CHANNEL_EVENTS_TAGGED_CHAT
