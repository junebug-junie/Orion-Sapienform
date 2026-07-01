"""
Regression test for the RDF write payload shape bug in `_write_to_sinks`.

Before the fix, the RDF branch built a Python list of (subject, predicate,
object) tuples and fed it into `RdfWriteRequest(graph=..., triples=<list>)`.
That never matched the real consumer contract at `orion.schemas.rdf.RdfWriteRequest`,
where `triples` must be a pre-serialized string and `id`/`source` are required.
This test imports the REAL schema (not any local fallback stub) so it fails
against the old shape and passes against the fixed one.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

# Real consumer-side contract -- must NOT be the fallback stub in app.main.
from orion.schemas.rdf import RdfWriteRequest  # noqa: E402
from app.main import _build_event_triples  # noqa: E402


def _make_event(**overrides):
    defaults = dict(
        event_id="evt-123",
        event_type="presence",
        narrative="A person walked into the kitchen carrying a bag of groceries.",
        entities=["person", "kitchen", "groceries"],
        tags=["household"],
        confidence=0.9,
        salience=0.5,
        evidence_refs=["artifact-1"],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_event_triples_returns_nonempty_string_with_narrative_and_entity():
    evt = _make_event()

    nt_content = _build_event_triples(evt)

    assert isinstance(nt_content, str)
    assert nt_content.strip() != ""
    assert "walked into the kitchen" in nt_content
    assert "groceries" in nt_content


def test_rdf_write_request_accepts_new_shape_against_real_schema():
    evt = _make_event()
    nt_content = _build_event_triples(evt)

    req = RdfWriteRequest(
        id=evt.event_id,
        source="vision-scribe",
        graph="orion:vision",
        triples=nt_content,
    )

    assert isinstance(req.triples, str)
    assert req.triples == nt_content
    assert req.id == evt.event_id
    assert req.source == "vision-scribe"
    assert req.graph == "orion:vision"


def test_old_list_of_tuples_shape_is_rejected_by_real_schema():
    with pytest.raises(Exception) as exc_info:
        RdfWriteRequest(graph="vision", triples=[("a", "b", "c")])

    # Confirm it's specifically a pydantic validation failure, not something
    # unrelated -- guards against silently regressing back to this shape.
    assert "ValidationError" in type(exc_info.value).__name__


# ---------------------------------------------------------------------------
# Regression test for the SQL write payload shape bug in `_write_to_sinks`.
#
# Before the fix, the SQL branch built `SqlWriteRequest(table="vision_events",
# data={...})` and published it under kind `"sql.write.request"` to
# `orion:collapse:sql-write`. That never matched the real consumer contract:
# `orion.schemas.sql.schemas.SqlWriteRequest` requires `table_name`, not
# `table` (so construction raised and was silently swallowed), and even if it
# hadn't, `orion-sql-writer` has no route for `sql.write.request` -- it
# routes purely by envelope kind through `DEFAULT_ROUTE_MAP` into `MODEL_MAP`.
# The fix publishes the real `VisionEventBundleItem` instance directly under
# kind `"vision.event.v1"`, matching how `orion-sql-writer` validates
# `env.payload` against `VisionEventBundleItem` for the `VisionEventSQL` route.
# ---------------------------------------------------------------------------

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.vision import VisionEventBundleItem, VisionEventPayload  # noqa: E402

from app.main import ScribeService  # noqa: E402
from app.settings import Settings as ScribeSettings  # noqa: E402


def _make_source_envelope() -> BaseEnvelope:
    return BaseEnvelope(
        kind="vision.event.bundle",
        source=ServiceRef(name="orion-vision-council", version="0.1.0"),
        payload={},
    )


@pytest.mark.asyncio
async def test_write_to_sinks_publishes_vision_event_bundle_item_for_sql_write():
    service = ScribeService()
    settings = ScribeSettings()

    calls: list[tuple] = []

    async def fake_send_write(channel, kind, payload, source_env):
        calls.append((channel, kind, payload, source_env))

    service._send_write = fake_send_write

    evt = VisionEventBundleItem(**_make_event().__dict__)
    payload = VisionEventPayload(events=[evt])
    source_env = _make_source_envelope()

    await service._write_to_sinks(payload, source_env)

    sql_calls = [c for c in calls if c[0] == settings.CHANNEL_SQL_WRITE]
    assert len(sql_calls) == 1
    channel, kind, sent_payload, sent_source_env = sql_calls[0]

    assert channel == settings.CHANNEL_SQL_WRITE
    assert kind == "vision.event.v1"
    assert sent_payload is evt
    assert not isinstance(sent_payload, dict)
    assert not hasattr(sent_payload, "table")
    assert not hasattr(sent_payload, "data")
    assert sent_source_env is source_env
