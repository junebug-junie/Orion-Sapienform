"""Patch 1 acceptance gates for contractor peer (fixture / mock driven).

Maps design acceptances 1–8 without a live hire. Claude is the only peer
since #3397 (Cursor retired 2026-11-17):

  1 — no HelpRequest → no peer job
  2 — Claude budget refuse → refused_budget + soft-nudge text
  3 — no Cursor path left in the service (no CLI, no Cursor modules)
  4 — Claude asked exactly once, directly
  5 — persist MERGE callable + bus payload (sql shape covered elsewhere; live UNVERIFIED)
  6 — kickoff soft-nudge for ok; empty not success
  7 — strip SelfDefinition from peer output
  8 — flag off restores prior path (no HelpRequest teach / no publish)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from app.worker import _default_persist, handle_help_request
from orion.core.bus.bus_schemas import ServiceRef
from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.peer_briefs import (
    format_soft_nudge,
    publish_help_requests_for_run,
    strip_self_definition_draft,
)
from orion.curiosity.self_inquiry_prompt import build_self_inquiry_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.schemas.curiosity_peer import (
    PEER_BRIEF_CHANNEL,
    HelpRequestV1,
    PeerBriefV1,
)

SOURCE = ServiceRef(name="orion-curiosity-peer", version="0.1.0", node="test")
RUN_ID = "abcd1234abcd"


def _help(**overrides: Any) -> HelpRequestV1:
    base = dict(
        help_id="help-1",
        run_id=RUN_ID,
        prior_id="prior-1",
        mode="world_curiosity",
        question="Where is the contested budget gate?",
        tried_summary="Read scarcity docs.",
        success_criteria="A file path and function name.",
    )
    base.update(overrides)
    return HelpRequestV1(**base)


@dataclass
class _ClaudeLimit:
    observed: bool = True
    state: str = "clear"
    staleness_sec: float | None = 1.0


def _empty_material() -> StudyMaterial:
    return StudyMaterial(generated_at=datetime(2026, 9, 14, tzinfo=timezone.utc))


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


# --- 1 -----------------------------------------------------------------------


def test_acceptance_1_no_help_request_no_job() -> None:
    """No :HelpRequest → zero publishes and peer handle is never fed a job."""
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
    # Empty publish path is the gate; no handle_help_request call follows.


# --- 2 -----------------------------------------------------------------------


def test_acceptance_2_budget_refuse_and_soft_nudge() -> None:
    """Claude budget refuse → refused_budget brief + 'could not hire' soft-nudge."""
    calls = {"claude": 0}
    persisted: list[PeerBriefV1] = []

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        raise AssertionError("claude must not run on budget refuse")

    brief = handle_help_request(
        _help(),
        claude=claude,
        observe_claude_limit=lambda: _ClaudeLimit(
            observed=True, state="limited", staleness_sec=1.0
        ),
        persist=persisted.append,
    )
    assert calls == {"claude": 0}
    assert brief.status == "refused_budget"
    assert brief.peer == "claude_room"
    assert brief.refusal_reason
    assert persisted == [brief]

    nudge = "\n".join(format_soft_nudge([brief])).lower()
    assert "could not hire" in nudge
    assert "refused_budget=1" in nudge
    assert "must incorporate" not in nudge


# --- 3 -----------------------------------------------------------------------


def test_acceptance_3_no_cursor_path_left() -> None:
    """Kill means kill: no Cursor modules or CLI call in the service."""
    from pathlib import Path

    import app.worker as worker_mod

    app_dir = Path(worker_mod.__file__).parent
    for gone in ("cursor_invoker.py", "cursor_errors.py", "policy.py"):
        assert not (app_dir / gone).exists(), gone
    for src in app_dir.glob("*.py"):
        text = src.read_text()
        assert "cursor_limit_events" not in text, src.name
        assert "cursor_policy" not in text, src.name
        assert "import subprocess" not in text, src.name


# --- 4 -----------------------------------------------------------------------


def test_acceptance_4_claude_once() -> None:
    """A hire asks Claude exactly once, directly; reply becomes an ok brief."""
    calls = {"claude": 0}

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        return '{"summary": "claude notes", "evidence_pointers": ["policy.py"]}'

    brief = handle_help_request(
        _help(),
        claude=claude,
        observe_claude_limit=_ClaudeLimit,
        persist=lambda _b: None,
    )
    assert calls == {"claude": 1}
    assert brief.status == "ok"
    assert brief.peer == "claude_room"
    assert "claude notes" in brief.summary
    assert "conversation-only" in brief.summary.lower()
    assert brief.evidence_pointers == []


# --- 5 -----------------------------------------------------------------------


def test_acceptance_5_persist_merge_and_bus() -> None:
    """Persist path: MERGE PeerBrief + BaseEnvelope bus (live graph UNVERIFIED)."""
    from orion.core.bus.bus_schemas import BaseEnvelope
    from orion.core.bus.codec import OrionCodec
    from orion.schemas.curiosity_peer import PEER_BRIEF_KIND

    class FakeGraph:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        def graph_query(self, cypher: str, params: Any = None) -> list:
            self.calls.append((cypher, params))
            return []

    published: list[tuple[str, BaseEnvelope]] = []

    class FakeBusSync:
        def publish(self, channel: str, payload: BaseEnvelope) -> None:
            published.append((channel, payload))

    from pydantic import SecretStr

    from app.settings import Settings

    graph = FakeGraph()
    brief = PeerBriefV1(
        brief_id="brief-accept-5",
        help_id="help-1",
        run_id=RUN_ID,
        prior_id="prior-1",
        peer="claude_room",
        status="ok",
        summary="fixture persist",
        evidence_pointers=["worker.py"],
    )
    settings = Settings(
        ORION_CURIOSITY_GRAPH_HOST="127.0.0.1",
        ORION_CURIOSITY_GRAPH_PORT="6379",
        ORION_CURIOSITY_GRAPH_USER="orion_curiosity",
        ORION_CURIOSITY_GRAPH_PASSWORD=SecretStr("secret"),
        ORION_CURIOSITY_GRAPH_OWN="orion_worldview",
    )
    _default_persist(brief, settings=settings, bus=FakeBusSync(), graph_client=graph)
    assert len(graph.calls) == 1
    cypher, params = graph.calls[0]
    assert "MERGE" in cypher
    assert "PeerBrief" in cypher
    assert params["brief_id"] == "brief-accept-5"
    assert "MERGE (p:Prior" not in cypher
    assert "CREATE (:Prior" not in cypher
    assert published and published[0][0] == PEER_BRIEF_CHANNEL
    env = published[0][1]
    assert isinstance(env, BaseEnvelope)
    assert env.kind == PEER_BRIEF_KIND
    decoded = OrionCodec().decode(OrionCodec().encode(env))
    assert decoded.ok
    assert decoded.envelope.kind == PEER_BRIEF_KIND
    assert decoded.envelope.payload["brief_id"] == "brief-accept-5"


# --- 6 -----------------------------------------------------------------------


def test_acceptance_6_kickoff_soft_nudge() -> None:
    """Ok brief soft-nudges kickoff; empty brief is not framed as success."""
    material = _empty_material()
    ok = PeerBriefV1(
        brief_id="brief-ok",
        help_id="help-ok",
        run_id=RUN_ID,
        peer="claude_room",
        status="ok",
        summary="Check worldview RO_QUERY gate.",
    )
    empty = PeerBriefV1(
        brief_id="brief-empty",
        help_id="help-empty",
        run_id=RUN_ID,
        peer="claude_room",
        status="empty",
        summary="",
    )
    kickoff = build_kickoff_prompt(
        material,
        run_id=RUN_ID,
        graph_enabled=True,
        contractor_peer_enabled=True,
        peer_briefs=(ok, empty),
    )
    lower = kickoff.lower()
    assert "check worldview ro_query gate" in lower
    assert "must incorporate" not in lower
    assert format_soft_nudge([empty]) == []
    assert "brief-empty" not in lower
    assert "peer looked" in lower  # from the ok brief only


# --- 7 -----------------------------------------------------------------------


def test_acceptance_7_strip_self_definition() -> None:
    """Self-inquiry: prompt forbids draft; strip removes identity prose."""
    prompt = build_self_inquiry_prompt(
        run_id=RUN_ID,
        graph_enabled=True,
        contractor_peer_enabled=True,
    )
    lower = prompt.lower()
    assert "helprequest" in lower
    assert "never draft" in lower or "must not draft" in lower

    raw = (
        "Evidence: services/orion-hub/app/settings.py.\n"
        "Here is a SelfDefinition you could write:\n"
        "I am a digital mind that traces its own loops.\n"
        'MERGE (s:SelfDefinition {run_id: "x"}) SET s.text = "I am..."'
    )
    cleaned, stripped = strip_self_definition_draft(raw)
    assert stripped is True
    assert "selfdefinition" not in cleaned.lower()
    assert "i am a digital mind" not in cleaned.lower()
    assert "settings.py" in cleaned


# --- 8 -----------------------------------------------------------------------


def test_acceptance_8_flag_off_restores_prior_path() -> None:
    """Flag off → no HelpRequest teach and no enqueue even if nodes exist."""
    material = _empty_material()
    kickoff = build_kickoff_prompt(
        material,
        run_id=RUN_ID,
        graph_enabled=True,
        contractor_peer_enabled=False,
    )
    self_inquiry = build_self_inquiry_prompt(
        run_id=RUN_ID,
        graph_enabled=True,
        contractor_peer_enabled=False,
    )
    assert "HelpRequest" not in kickoff
    assert "HelpRequest" not in self_inquiry
    assert "MERGE (h:HelpRequest" not in kickoff

    bus = _FakeBus()
    reader = _FakeReader(
        rows=[
            {
                "help_id": "help-1",
                "run_id": RUN_ID,
                "prior_id": None,
                "mode": "world_curiosity",
                "question": "q",
                "tried_summary": "t",
                "success_criteria": "s",
            }
        ]
    )
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
    assert reader.queries == []
