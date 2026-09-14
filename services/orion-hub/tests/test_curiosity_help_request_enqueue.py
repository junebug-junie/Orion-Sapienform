"""Post-run HelpRequest enqueue — Acceptance 1 + publish count.

Fake reader + fake bus only. Flag off or zero HelpRequests → zero publishes.
"""

from __future__ import annotations

import asyncio
from typing import Any

from orion.curiosity.peer_briefs import publish_help_requests_for_run
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.curiosity_peer import HELP_REQUEST_CHANNEL, HELP_REQUEST_KIND


SOURCE = ServiceRef(name="orion-hub", version="0.1.0", node="athena")
RUN_ID = "abcd1234abcd"


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish(self, channel: str, envelope: Any) -> None:
        self.published.append((channel, envelope))


class _FakeReader:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = list(rows or [])
        self.queries: list[str] = []

    def query(self, cypher: str) -> list[dict[str, Any]]:
        self.queries.append(cypher)
        return list(self.rows)


def _help_row(**over: Any) -> dict[str, Any]:
    base = {
        "help_id": "help-1",
        "run_id": RUN_ID,
        "prior_id": None,
        "mode": "world_curiosity",
        "question": "Why is RO_QUERY the only Hub path?",
        "tried_summary": "Looked at worldview.py header",
        "success_criteria": "A file:line pointer and one open question",
    }
    base.update(over)
    return base


def test_no_help_requests_publishes_nothing() -> None:
    bus = _FakeBus()
    reader = _FakeReader(rows=[])
    n = asyncio.run(
        publish_help_requests_for_run(
            enabled=True,
            run_id=RUN_ID,
            reader=reader,
            bus=bus,
            source_ref=SOURCE,
        )
    )
    assert n == 0
    assert bus.published == []
    assert reader.queries  # still queried when enabled


def test_one_help_request_publishes_one_payload() -> None:
    bus = _FakeBus()
    reader = _FakeReader(rows=[_help_row()])
    n = asyncio.run(
        publish_help_requests_for_run(
            enabled=True,
            run_id=RUN_ID,
            reader=reader,
            bus=bus,
            source_ref=SOURCE,
        )
    )
    assert n == 1
    assert len(bus.published) == 1
    channel, envelope = bus.published[0]
    assert channel == HELP_REQUEST_CHANNEL
    assert envelope.kind == HELP_REQUEST_KIND
    payload = envelope.payload
    assert payload["help_id"] == "help-1"
    assert payload["run_id"] == RUN_ID
    assert payload["question"].startswith("Why is RO_QUERY")
    assert "HelpRequest" in reader.queries[0]
    assert RUN_ID in reader.queries[0]


def test_flag_off_publishes_nothing_even_if_nodes_exist() -> None:
    bus = _FakeBus()
    reader = _FakeReader(rows=[_help_row()])
    n = asyncio.run(
        publish_help_requests_for_run(
            enabled=False,
            run_id=RUN_ID,
            reader=reader,
            bus=bus,
            source_ref=SOURCE,
        )
    )
    assert n == 0
    assert bus.published == []
    assert reader.queries == []  # must not even read when flag off
