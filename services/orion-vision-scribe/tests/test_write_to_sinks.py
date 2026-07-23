"""
Regression tests for `_write_to_sinks`.

The RDF write side was removed 2026-07-23 (live-verified pure redundancy
with Postgres `vision_events` -- see README.md's "RDF write path: removed"
section). SQL is now the sole sink.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.vision import VisionEventBundleItem, VisionEventPayload  # noqa: E402

from app.main import ScribeService  # noqa: E402
from app.settings import Settings as ScribeSettings  # noqa: E402


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


def _make_source_envelope() -> BaseEnvelope:
    return BaseEnvelope(
        kind="vision.event.bundle",
        source=ServiceRef(name="orion-vision-council", version="0.1.0"),
        payload={},
    )


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


@pytest.mark.asyncio
async def test_write_to_sinks_makes_exactly_one_publish_call_no_rdf():
    """Regression for the 2026-07-23 RDF-write removal: _write_to_sinks must
    publish to SQL only -- not to CHANNEL_RDF_ENQUEUE or any 'rdf.write.request'
    kind. Guards against the RDF branch silently reappearing."""
    service = ScribeService()

    calls: list[tuple] = []

    async def fake_send_write(channel, kind, payload, source_env):
        calls.append((channel, kind, payload, source_env))

    service._send_write = fake_send_write

    evt = VisionEventBundleItem(**_make_event().__dict__)
    payload = VisionEventPayload(events=[evt])
    source_env = _make_source_envelope()

    await service._write_to_sinks(payload, source_env)

    assert len(calls) == 1
    assert calls[0][1] == "vision.event.v1"
    assert not hasattr(ScribeSettings(), "CHANNEL_RDF_ENQUEUE")


@pytest.mark.asyncio
async def test_write_to_sinks_reports_ok_when_sql_write_succeeds():
    """ack.ok must depend on sql_ok alone now -- previously required
    sql_ok AND rdf_ok, which would have silently regressed to always-partial
    once the RDF branch was removed if this line wasn't updated too."""
    service = ScribeService()

    async def fake_send_write(channel, kind, payload, source_env):
        pass

    service._send_write = fake_send_write

    evt = VisionEventBundleItem(**_make_event().__dict__)
    payload = VisionEventPayload(events=[evt])
    source_env = _make_source_envelope()

    ack = await service._write_to_sinks(payload, source_env)

    assert ack.ok is True
