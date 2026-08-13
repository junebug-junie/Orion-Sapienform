"""Contract tests for persisting doc_semantic_drift events.

This is the first real consumer of ``orion:substrate:doc_semantic_drift``.
The channel was ``consumer_services: []`` from 2026-08-11, and because
``OrionBusAsync.publish()`` is Redis pub/sub rather than a stream, that meant
every event was dropped on publish -- roughly two days of real doc-drift
scores exist nowhere. These tests pin the wiring that stops that, and the
column shape the accumulated samples will be read back through.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SERVICE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.models import DocSemanticDriftSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402

from orion.schemas.doc_semantic_drift import DocSemanticDriftV1  # noqa: E402


def _event(**overrides) -> DocSemanticDriftV1:
    payload = dict(
        observed_at=datetime.now(timezone.utc),
        sha="a" * 40,
        path="docs/real.md",
        commit_prefix="docs",
        diff_scoped_embedding_diff=0.3558,
        chunk_count_removed=1,
        chunk_count_added=5,
        hunk_removed_len_chars=120,
        hunk_added_len_chars=4200,
    )
    payload.update(overrides)
    return DocSemanticDriftV1(**payload)


# --------------------------------------------------------------------------
# Routing / subscription wiring
# --------------------------------------------------------------------------


def test_message_kind_routes_to_the_sql_model() -> None:
    assert DEFAULT_ROUTE_MAP["substrate.doc_semantic_drift.v1"] == "DocSemanticDriftSQL"


def test_route_map_survives_an_env_override_of_other_routes() -> None:
    """`route_map` merges the env JSON over the code defaults, so a live
    .env carrying an older, shorter override still picks this route up. This
    is the half of the config that is forgiving."""
    settings = Settings(SQL_WRITER_ROUTE_MAP_JSON='{"some.other.kind.v1":"SomethingElseSQL"}')
    assert settings.route_map["substrate.doc_semantic_drift.v1"] == "DocSemanticDriftSQL"
    assert settings.route_map["some.other.kind.v1"] == "SomethingElseSQL"


def test_channel_is_in_the_default_subscribe_list() -> None:
    """The unforgiving half: `effective_subscribe_channels` REPLACES the
    code default with whatever the env sets -- it does not merge, unlike
    `route_map` above. A live .env that predates this patch therefore leaves
    the writer permanently unsubscribed, and the row silently never appears.
    Confirmed live on 2026-08-12 with dev_economics, which is why the .env
    is edited by hand in the same patch."""
    assert "orion:substrate:doc_semantic_drift" in Settings().effective_subscribe_channels


def test_env_subscribe_list_replaces_rather_than_merges() -> None:
    """Pins the asymmetry itself, so a future refactor that "helpfully"
    makes this merge doesn't silently invalidate the .env discipline above
    -- and so the next person reading these tests learns the rule."""
    # A real list, not a JSON string: pydantic-settings only JSON-decodes
    # values arriving from the env, not direct constructor kwargs.
    settings = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:only:this:one"])
    assert settings.effective_subscribe_channels.count("orion:only:this:one") == 1
    assert "orion:substrate:doc_semantic_drift" not in settings.effective_subscribe_channels


# --------------------------------------------------------------------------
# Column shape
# --------------------------------------------------------------------------


def test_every_payload_field_has_a_column_or_is_deliberately_dropped() -> None:
    """Guards against a schema field being added upstream and silently not
    persisted -- the sample set would then be missing a dimension nobody
    notices until they try to use it."""
    columns = set(DocSemanticDriftSQL.__table__.columns.keys())
    payload_fields = set(DocSemanticDriftV1.model_fields.keys())
    # schema_version is a constant discriminator, not data worth a column.
    deliberately_dropped = {"schema_version"}
    assert payload_fields - deliberately_dropped <= columns


def test_score_column_is_nullable_because_none_is_a_real_state() -> None:
    """After the 2026-08-13 skip patch a NULL here means exactly one thing:
    a real embedding failure. Structurally-unscoreable changes are no longer
    published at all. A NOT NULL constraint would drop precisely the rows an
    operator most needs to see."""
    assert DocSemanticDriftSQL.__table__.columns["diff_scoped_embedding_diff"].nullable is True


def test_chunk_counts_are_indexed_for_the_comparability_filter() -> None:
    """`chunk_count_removed == 1 AND chunk_count_added == 1` is the first
    query anyone deriving a threshold will run -- it is the only subset
    comparable to the original 0.3996 single-window calibration."""
    assert DocSemanticDriftSQL.__table__.columns["chunk_count_removed"].index is True
    assert DocSemanticDriftSQL.__table__.columns["chunk_count_added"].index is True


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


def test_event_id_is_derived_from_sha_and_path() -> None:
    assert _event().event_id == f"doc-semantic-drift:{'a' * 40}:docs/real.md"


def test_rescoring_the_same_change_reuses_the_same_key() -> None:
    """The reason this is derived rather than a uuid4. The producer re-scores
    a whole range after a failed publish, so a re-delivered event must upsert
    onto the same row -- duplicates would silently inflate the sample count
    and corrupt the very distribution this table is being kept for."""
    first = _event(diff_scoped_embedding_diff=0.31)
    second = _event(diff_scoped_embedding_diff=0.87)
    assert first.event_id == second.event_id


def test_distinct_files_in_one_commit_get_distinct_keys() -> None:
    """One commit routinely touches several docs -- they must not collapse
    onto a single row."""
    assert _event(path="docs/a.md").event_id != _event(path="docs/b.md").event_id


def test_same_file_at_different_commits_gets_distinct_keys() -> None:
    """A doc edited across two commits is two real observations, not one."""
    assert _event(sha="a" * 40).event_id != _event(sha="b" * 40).event_id


def test_explicit_event_id_is_not_overwritten() -> None:
    """The validator fills a blank; it must not clobber a real value from
    the wire, or a replayed historical event would be rewritten."""
    assert _event(event_id="doc-semantic-drift:legacy:x.md").event_id == (
        "doc-semantic-drift:legacy:x.md"
    )


def test_event_id_is_the_primary_key() -> None:
    pk = [c.name for c in DocSemanticDriftSQL.__table__.primary_key.columns]
    assert pk == ["event_id"]
