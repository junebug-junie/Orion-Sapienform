from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(SERVICE_ROOT)]

from orion.graph.falkor_client import RecordingFalkorClient  # noqa: E402

from app.falkor_recall_writer import (  # noqa: E402
    extract_sentiment,
    filter_noise,
    write_chat_turn_tags_to_falkor,
    write_collapse_triage_tags_to_falkor,
)


def test_extract_sentiment_pulls_out_prefixed_tag() -> None:
    sentiment, remaining = extract_sentiment(["Circe", "sentiment:positive", "Atlas"])
    assert sentiment == "positive"
    assert remaining == ["Circe", "Atlas"]


def test_extract_sentiment_none_when_absent() -> None:
    sentiment, remaining = extract_sentiment(["Circe", "Atlas"])
    assert sentiment is None
    assert remaining == ["Circe", "Atlas"]


def test_filter_noise_rejects_digits_stopwords_and_relative_time() -> None:
    kept, rejected = filter_noise(
        ["Circe", "1", "today", "18 years ago", "about a month ago", "last week", "that day on", "dozen"]
    )
    assert kept == ["circe"]
    assert set(rejected) == {"1", "today", "18 years ago", "about a month ago", "last week", "that day on", "dozen"}


def test_filter_noise_normalizes_and_dedupes() -> None:
    kept, rejected = filter_noise(["Circe", "circe", " CIRCE "])
    assert kept == ["circe"]
    assert rejected == []


def test_filter_noise_keeps_real_entities() -> None:
    kept, rejected = filter_noise(["Juniper", "Orion", "Athena", "Atlas", "Circe"])
    assert kept == ["juniper", "orion", "athena", "atlas", "circe"]
    assert rejected == []


def test_write_chat_turn_tags_to_falkor_full_shape() -> None:
    client = RecordingFalkorClient()
    result = write_chat_turn_tags_to_falkor(
        client,
        turn_id="turn-1",
        source_kind="chat.history",
        session_id="sess-1",
        ts="2026-07-18T00:00:00+00:00",
        correlation_id="corr-1",
        tags=["Circe", "sentiment:positive", "today"],
        entities=["Circe", "Atlas"],
    )

    assert result["tags_written"] == 1
    assert result["tags_rejected"] == ["today"]
    assert result["entities_written"] == 2
    assert result["sentiment"] == "positive"
    assert result["session_linked"] is True

    cyphers = [c for c, _ in client.calls]
    assert any("MERGE (t:ChatTurn" in c for c in cyphers)
    assert any("MERGE (s:ChatSession" in c and "HAS_TURN" in c for c in cyphers)
    assert any("HAS_TAG" in c for c in cyphers)
    assert any("MENTIONS_ENTITY" in c for c in cyphers)

    turn_call = next(c for c, p in client.calls if "MERGE (t:ChatTurn" in c)
    turn_params = next(p for c, p in client.calls if "MERGE (t:ChatTurn" in c)
    assert turn_params["turn_id"] == "turn-1"
    assert turn_params["ts"] == "2026-07-18T00:00:00+00:00"
    assert turn_params["correlation_id"] == "corr-1"
    assert turn_params["sentiment"] == "positive"
    assert "sentiment" in turn_call  # SET clause references it
    assert turn_params["source_kind"] == "chat.history"


def test_write_chat_turn_tags_to_falkor_no_session_id_skips_session_merge() -> None:
    client = RecordingFalkorClient()
    write_chat_turn_tags_to_falkor(
        client,
        turn_id="turn-2",
        source_kind="chat.history",
        session_id=None,
        ts="2026-07-18T00:00:00+00:00",
        correlation_id=None,
        tags=["Circe"],
        entities=[],
    )
    cyphers = [c for c, _ in client.calls]
    assert not any("ChatSession" in c for c in cyphers)


def test_write_chat_turn_tags_to_falkor_no_tags_or_entities_still_writes_anchor() -> None:
    client = RecordingFalkorClient()
    result = write_chat_turn_tags_to_falkor(
        client,
        turn_id="turn-3",
        source_kind="social.turn.stored.v1",
        session_id=None,
        ts="2026-07-18T00:00:00+00:00",
        correlation_id=None,
        tags=[],
        entities=[],
    )
    # ts and source_kind are both required (non-optional) parameters, so the
    # SET clause always has at least these two -- there is no
    # bare-anchor-with-no-SET path.
    assert len(client.calls) == 1
    assert client.calls[0][0] == "MERGE (t:ChatTurn {turn_id: $turn_id}) SET t.source_kind = $source_kind, t.ts = $ts"
    assert client.calls[0][1]["source_kind"] == "social.turn.stored.v1"
    assert result["tags_written"] == 0
    assert result["entities_written"] == 0


def test_relative_time_regex_catches_bare_and_trailing_on_variants() -> None:
    kept, rejected = filter_noise(["that day", "that day on", "last week"])
    assert kept == []
    assert set(rejected) == {"that day", "that day on", "last week"}


def test_filter_noise_merges_diacritic_variants() -> None:
    kept, rejected = filter_noise(["Orion", "Oríon", "orión"])
    assert kept == ["orion"]
    assert rejected == []


def test_filter_noise_preserves_non_latin_script() -> None:
    """Regression: an earlier version of _normalize_identity_key used
    NFKD + encode('ascii','ignore'), which drops ANY non-Latin-script string
    to '' (verified: unicodedata.normalize('NFKD','北京').encode('ascii',
    'ignore') == b''), causing filter_noise to reject it as noise. Stripping
    only combining marks (category Mn) instead of ascii-encoding fixes this.
    Chinese/Cyrillic pass through case-folded and otherwise unchanged (no
    combining-mark decomposition applies); Greek's tonos accent DOES
    decompose under NFKD the same way a Latin diacritic does, so "Ελλάδα"
    losing its accent to "ελλαδα" is consistent stripping, not a bug --
    what matters is that the string survives as real content, not ''."""
    kept, rejected = filter_noise(["北京", "Москва", "Ελλάδα"])
    assert kept == ["北京", "москва", "ελλαδα"]
    assert rejected == []


def test_write_chat_turn_tags_to_falkor_diacritic_dedup_across_call() -> None:
    client = RecordingFalkorClient()
    write_chat_turn_tags_to_falkor(
        client,
        turn_id="turn-5",
        source_kind="chat.history",
        session_id=None,
        ts="2026-07-18T00:00:00+00:00",
        correlation_id=None,
        tags=[],
        entities=["Orion", "Oríon"],
    )
    entity_call_params = next(p for c, p in client.calls if "MENTIONS_ENTITY" in c)
    assert entity_call_params["names"] == ["orion"]


def test_write_chat_turn_tags_to_falkor_entity_dedup_across_call() -> None:
    client = RecordingFalkorClient()
    write_chat_turn_tags_to_falkor(
        client,
        turn_id="turn-4",
        source_kind="chat.history",
        session_id=None,
        ts="2026-07-18T00:00:00+00:00",
        correlation_id=None,
        tags=[],
        entities=["Circe", "circe", "CIRCE"],
    )
    entity_call_params = next(p for c, p in client.calls if "MENTIONS_ENTITY" in c)
    assert entity_call_params["names"] == ["circe"]


def test_write_collapse_triage_tags_to_falkor_full_shape() -> None:
    client = RecordingFalkorClient()
    result = write_collapse_triage_tags_to_falkor(
        client,
        collapse_id="collapse_abc123",
        correlation_id="corr-1",
        ts="2026-07-22T00:00:00+00:00",
        tags=["gnostic", "sentiment:positive", "today"],
        entities=["gnostic", "transcendent"],
    )

    assert result["tags_written"] == 1
    assert result["tags_rejected"] == ["today"]
    assert result["entities_written"] == 2
    assert result["sentiment"] == "positive"

    cyphers = [c for c, _ in client.calls]
    assert any("MERGE (c:CollapseEvent" in c for c in cyphers)
    assert any("HAS_TAG" in c for c in cyphers)
    assert any("MENTIONS_ENTITY" in c for c in cyphers)
    assert not any("ChatSession" in c or "ChatTurn" in c for c in cyphers)

    event_call = next(c for c, p in client.calls if "MERGE (c:CollapseEvent" in c)
    event_params = next(p for c, p in client.calls if "MERGE (c:CollapseEvent" in c)
    assert event_params["collapse_id"] == "collapse_abc123"
    assert event_params["ts"] == "2026-07-22T00:00:00+00:00"
    assert event_params["correlation_id"] == "corr-1"
    assert event_params["sentiment"] == "positive"
    assert "sentiment" in event_call


def test_write_collapse_triage_tags_to_falkor_shares_entity_namespace_with_chat_turns() -> None:
    """A collapse-mirror mention of an entity must resolve to the SAME
    :Entity node a chat turn would create -- not a separate namespace."""
    client = RecordingFalkorClient()
    write_collapse_triage_tags_to_falkor(
        client,
        collapse_id="collapse_xyz",
        correlation_id=None,
        ts="2026-07-22T00:00:00+00:00",
        tags=[],
        entities=["Juniper"],
    )
    entity_call_params = next(p for c, p in client.calls if "MENTIONS_ENTITY" in c)
    assert entity_call_params["names"] == ["juniper"]


def test_write_collapse_triage_tags_to_falkor_no_tags_or_entities_still_writes_anchor() -> None:
    client = RecordingFalkorClient()
    result = write_collapse_triage_tags_to_falkor(
        client,
        collapse_id="collapse_bare",
        correlation_id=None,
        ts="2026-07-22T00:00:00+00:00",
        tags=[],
        entities=[],
    )
    assert len(client.calls) == 1
    assert client.calls[0][0] == "MERGE (c:CollapseEvent {collapse_id: $collapse_id}) SET c.ts = $ts"
    assert result["tags_written"] == 0
    assert result["entities_written"] == 0
