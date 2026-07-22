"""Regression tests for chat/social-turn tagging in handle_triage_event.

Two fixes live here:

1. Observer-gate scoping (fix/meta-tags-chat-history-observer-gate, merged).
   Commit aaf66d58 ("Enforce metacog draft/enrich patch contracts") moved
   should_route_to_triage's Juniper-observer check in front of the
   chat.history/social.turn.stored.v1 branch as an apparent side effect of
   an unrelated refactor. ChatHistoryTurnV1 (orion-hub's real chat.history
   payload) has no `observer` field, so this unconditionally skipped every
   real chat turn for ~6 months. Fixed by scoping the gate to the generic
   collapse-mirror branch only (restoring commit ebd3b9d9's structure).

2. Kill the Fuseki dual-write (this branch). Once (1) was fixed, both the
   Fuseki `chat_tagging` enrichment copy (via orion-rdf-writer, consuming
   orion:tags:chat:enriched) and the Falkor Phase 2 write went live
   simultaneously -- redundant materializations of the same data. FalkorDB
   is now the sole persistence target: no bus publish for chat.history or
   social.turn.stored.v1 at all anymore. The Falkor write, previously
   chat.history-gated only, now also covers social.turn.stored.v1 (real
   live traffic from orion-social-memory) so it doesn't lose persistence.
   See services/orion-meta-tags/README.md.

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

from orion.graph.falkor_client import RecordingFalkorClient  # noqa: E402


@pytest.fixture
def main_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORION_BUS_URL", "redis://example.test/0")

    # A fake entity with real `.text`/`.label_` content (not just an empty
    # ents=[]) so the extraction line (app.main._named_entities, which
    # filters by ent.label_) is actually exercised, not just left permanently
    # dead under the mock. label_="PERSON" is one of _KEEP_ENTITY_LABELS.
    fake_doc = SimpleNamespace(ents=[SimpleNamespace(text="Circe", label_="PERSON")])
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
    # Falkor is the sole persistence path now (settings.py default is True)
    # -- default it on here too, with a recording fake client, so tests
    # observe real writes rather than a silent no-op.
    m.settings.RECALL_FALKOR_TAG_ENTITY_ENABLED = True
    m._falkor_client = RecordingFalkorClient()
    return m


def _envelope(main_module, *, kind: str, payload: dict):
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    return BaseEnvelope(
        kind=kind,
        source=ServiceRef(name="hub", node="athena", version="0.4.0"),
        payload=payload,
    )


@pytest.mark.asyncio
async def test_chat_history_turn_with_no_observer_writes_to_falkor_not_bus(main_module) -> None:
    """The gate-scoping regression: a real hub chat.history payload (no
    `observer` field, matching ChatHistoryTurnV1's real schema) must NOT be
    skipped. And the dual-write-kill regression: it must land in Falkor,
    not get published to the bus at all."""
    envelope = _envelope(
        main_module,
        kind="chat.history",
        payload={"session_id": "sess-1", "prompt": "hi Circe", "response": "hello"},
    )
    await main_module.handle_triage_event(envelope)

    main_module.meta_tagger.bus.publish.assert_not_awaited()
    calls = main_module._falkor_client.calls
    turn_calls = [params for cypher, params in calls if "MERGE (t:ChatTurn" in cypher]
    assert turn_calls and turn_calls[0]["source_kind"] == "chat.history"
    entity_calls = [params for cypher, params in calls if "MENTIONS_ENTITY" in cypher]
    assert entity_calls and "circe" in entity_calls[0]["names"]


@pytest.mark.asyncio
async def test_chat_history_drops_numeric_and_temporal_ner_labels(main_module) -> None:
    """Live-verified against the real en_core_web_trf model (2026-07-19):
    'first'->ORDINAL, 'the day'->DATE, 'a moment ago'->TIME, '8GB'->QUANTITY,
    '7680.0'->CARDINAL all leaked into the FalkorDB Entity graph as noise
    because entity extraction never checked ent.label_. 'P4'->PRODUCT is a
    real entity and must survive the same filter."""
    main_module.nlp = MagicMock(
        return_value=SimpleNamespace(
            ents=[
                SimpleNamespace(text="P4", label_="PRODUCT"),
                SimpleNamespace(text="first", label_="ORDINAL"),
                SimpleNamespace(text="the day", label_="DATE"),
                SimpleNamespace(text="a moment ago", label_="TIME"),
                SimpleNamespace(text="8GB", label_="QUANTITY"),
                SimpleNamespace(text="7680.0", label_="CARDINAL"),
            ]
        )
    )
    envelope = _envelope(
        main_module,
        kind="chat.history",
        payload={"session_id": "sess-2", "prompt": "the P4 ran fine", "response": "ok"},
    )
    await main_module.handle_triage_event(envelope)

    entity_calls = [params for cypher, params in main_module._falkor_client.calls if "MENTIONS_ENTITY" in cypher]
    assert entity_calls and entity_calls[0]["names"] == ["p4"]


@pytest.mark.asyncio
async def test_social_turn_stored_also_writes_to_falkor(main_module) -> None:
    """social.turn.stored.v1 (real live traffic from orion-social-memory)
    shares this branch and was never wired to Falkor before this fix --
    only to the now-killed Fuseki channel. Confirms it isn't losing
    persistence: same ChatTurn write shape as chat.history."""
    envelope = _envelope(
        main_module,
        kind="social.turn.stored.v1",
        payload={"session_id": "sess-3", "prompt": "hi", "response": "hello"},
    )
    await main_module.handle_triage_event(envelope)

    main_module.meta_tagger.bus.publish.assert_not_awaited()
    calls = main_module._falkor_client.calls
    turn_calls = [params for cypher, params in calls if "MERGE (t:ChatTurn" in cypher]
    assert turn_calls and turn_calls[0]["source_kind"] == "social.turn.stored.v1"


@pytest.mark.asyncio
async def test_falkor_write_failure_is_swallowed_and_logged(main_module, caplog) -> None:
    """The sole-persistence-path consequence of a Falkor write failure: no
    exception propagates out of handle_triage_event (one bad write must not
    take down triage intake for every subsequent event), but it's now logged
    at ERROR (not WARNING) since there's no Fuseki fallback to catch it."""
    main_module._falkor_client = MagicMock()
    main_module._falkor_client.graph_query.side_effect = RuntimeError("falkor down")

    envelope = _envelope(
        main_module,
        kind="chat.history",
        payload={"session_id": "sess-5", "prompt": "hi", "response": "hello"},
    )
    with caplog.at_level("ERROR"):
        await main_module.handle_triage_event(envelope)  # must not raise

    main_module.meta_tagger.bus.publish.assert_not_awaited()
    assert any("falkor_recall_write_failed_data_lost" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_chat_history_falkor_write_skipped_when_flag_disabled(main_module) -> None:
    """Documents the real consequence of RECALL_FALKOR_TAG_ENTITY_ENABLED=false
    now that Falkor is the only persistence path: nothing is written or
    published anywhere for this turn. Not a bug -- the flag's meaning
    genuinely changed (see settings.py's comment) -- but worth locking in
    as intentional, not an accidental silent drop."""
    main_module.settings.RECALL_FALKOR_TAG_ENTITY_ENABLED = False
    envelope = _envelope(
        main_module,
        kind="chat.history",
        payload={"session_id": "sess-4", "prompt": "hi", "response": "hello"},
    )
    await main_module.handle_triage_event(envelope)

    main_module.meta_tagger.bus.publish.assert_not_awaited()
    assert not main_module._falkor_client.calls


@pytest.mark.asyncio
async def test_generic_collapse_event_without_observer_is_still_skipped(main_module) -> None:
    """The generic (non-chat) collapse-mirror path must still require a
    Juniper observer -- untouched by either fix in this file."""
    envelope = _envelope(
        main_module,
        kind="collapse.mirror.entry",
        payload={"id": "c1", "text": "some collapse text", "observer": None},
    )
    await main_module.handle_triage_event(envelope)
    main_module.meta_tagger.bus.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_generic_collapse_event_with_juniper_observer_still_publishes(main_module) -> None:
    """The generic collapse-mirror path still publishes to the bus
    (orion:tags:enriched) -- only chat/social-turn tagging moved to
    Falkor-only. orion-rdf-writer and orion-sql-writer both still consume
    this channel."""
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
async def test_collapse_triage_falkor_write_skipped_when_flag_disabled_by_default(main_module) -> None:
    """RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED defaults False (dark/additive,
    unlike the chat/social flag) -- the bus publish (Fuseki's only path in)
    must still fire even though Falkor doesn't."""
    envelope = _envelope(
        main_module,
        kind="collapse.mirror.entry",
        payload={"id": "c3", "collapse_id": "collapse_c3", "text": "some collapse text", "observer": "Juniper"},
    )
    await main_module.handle_triage_event(envelope)
    main_module.meta_tagger.bus.publish.assert_awaited_once()
    assert not main_module._falkor_client.calls


@pytest.mark.asyncio
async def test_collapse_triage_writes_to_falkor_and_still_publishes_when_flag_enabled(main_module) -> None:
    """Additive, not a swap: with the flag on, both the Falkor write AND the
    bus publish (Fuseki via orion-rdf-writer) must happen -- this workload
    has no Falkor-side historical backfill yet, so Fuseki can't be dropped
    just because the flag is on."""
    main_module.settings.RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED = True
    envelope = _envelope(
        main_module,
        kind="collapse.mirror.entry",
        payload={"id": "c4", "collapse_id": "collapse_c4", "text": "some collapse text", "observer": "Juniper"},
    )
    await main_module.handle_triage_event(envelope)

    main_module.meta_tagger.bus.publish.assert_awaited_once()
    calls = main_module._falkor_client.calls
    event_calls = [params for cypher, params in calls if "MERGE (c:CollapseEvent" in cypher]
    assert event_calls and event_calls[0]["collapse_id"] == "collapse_c4"
    entity_calls = [params for cypher, params in calls if "MENTIONS_ENTITY" in cypher]
    assert entity_calls and "circe" in entity_calls[0]["names"]


@pytest.mark.asyncio
async def test_collapse_triage_falkor_write_failure_is_swallowed_at_warning_not_error(main_module, caplog) -> None:
    """Unlike the chat/social writer, a caught exception here is not data
    loss (Fuseki is still the real persistence path) -- logged at WARNING."""
    main_module.settings.RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED = True
    main_module._falkor_client = MagicMock()
    main_module._falkor_client.graph_query.side_effect = RuntimeError("falkor down")

    envelope = _envelope(
        main_module,
        kind="collapse.mirror.entry",
        payload={"id": "c5", "collapse_id": "collapse_c5", "text": "some collapse text", "observer": "Juniper"},
    )
    with caplog.at_level("WARNING"):
        await main_module.handle_triage_event(envelope)  # must not raise

    main_module.meta_tagger.bus.publish.assert_awaited_once()
    assert any("falkor_collapse_triage_write_failed" in r.message for r in caplog.records)
