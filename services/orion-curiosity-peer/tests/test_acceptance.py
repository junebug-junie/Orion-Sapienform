"""Patch 1 acceptance gates for contractor peer (fixture / mock driven).

Maps design acceptances 1–8 without a live Cursor hire:

  1 — no HelpRequest → no peer job
  2 — budget refuse → refused_budget + soft-nudge text
  3 — Cursor CLI argv read-only; no --force/--yolo; --mode ask required
  4 — Claude fallback exactly once on token unavailable
  5 — persist MERGE callable + bus payload (sql shape covered elsewhere; live UNVERIFIED)
  6 — kickoff soft-nudge for ok; empty not success
  7 — strip SelfDefinition from peer output
  8 — flag off restores prior path (no HelpRequest teach / no publish)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from app.cursor_errors import TokenUnavailable
from app.policy import assert_read_only_cli_argv, build_cursor_agent_argv
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
from orion.dev_economics.cursor_limit_events import CursorLimitObservation
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


def _clear_budget() -> CursorLimitObservation:
    return CursorLimitObservation(observed=True, state="clear", staleness_sec=1.0)


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
    """Budget refuse → refused_budget brief + 'could not hire' soft-nudge."""
    calls = {"cursor": 0, "claude": 0}
    persisted: list[PeerBriefV1] = []

    def cursor(*_a: Any, **_k: Any) -> PeerBriefV1:
        calls["cursor"] += 1
        raise AssertionError("cursor must not run on budget refuse")

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        raise AssertionError("claude must not run on budget refuse")

    brief = handle_help_request(
        _help(),
        cursor=cursor,
        claude=claude,
        observe_limit=lambda: CursorLimitObservation(
            observed=False, state="unknown", staleness_sec=None
        ),
        persist=persisted.append,
    )
    assert calls == {"cursor": 0, "claude": 0}
    assert brief.status == "refused_budget"
    assert brief.refusal_reason
    assert persisted == [brief]

    nudge = "\n".join(format_soft_nudge([brief])).lower()
    assert "could not hire" in nudge
    assert "refused_budget=1" in nudge
    assert "must incorporate" not in nudge


# --- 3 -----------------------------------------------------------------------


def test_acceptance_3_cli_argv_read_only_no_force() -> None:
    """Invoker policy: ask/print argv only; force/yolo/plan refused."""
    ok = build_cursor_agent_argv(
        agent_bin="cursor-agent",
        prompt="investigate",
        workspace="/repo",
        model="composer-2.5",
    )
    assert_read_only_cli_argv(ok)
    assert ok[ok.index("--mode") + 1] == "ask"
    for bad in ("--force", "--yolo", "--approve-mcps"):
        with pytest.raises(ValueError):
            assert_read_only_cli_argv([*ok[:-1], bad, ok[-1]])

    plan = [
        "cursor-agent",
        "-p",
        "--mode",
        "plan",
        "--workspace",
        "/repo",
        "--trust",
        "x",
    ]
    with pytest.raises(ValueError) as exc:
        assert_read_only_cli_argv(plan)
    assert "ask" in str(exc.value).lower()


# --- 4 -----------------------------------------------------------------------


def test_acceptance_4_fallback_once() -> None:
    """Token unavailable → exactly one Claude attempt; success skips second."""
    from dataclasses import dataclass

    calls = {"cursor": 0, "claude": 0}

    @dataclass
    class _ClearClaude:
        observed: bool = True
        state: str = "clear"
        staleness_sec: float | None = 1.0

    def cursor(*_a: Any, **_k: Any) -> PeerBriefV1:
        calls["cursor"] += 1
        raise TokenUnavailable("cursor tokens dry")

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        return '{"summary": "fallback notes", "evidence_pointers": ["policy.py"]}'

    brief = handle_help_request(
        _help(),
        cursor=cursor,
        claude=claude,
        observe_limit=_clear_budget,
        observe_claude_limit=lambda: _ClearClaude(),
        persist=lambda _b: None,
    )
    assert calls == {"cursor": 1, "claude": 1}
    assert brief.status == "ok"
    assert brief.peer == "claude_room"
    assert "fallback notes" in brief.summary
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
        peer="cursor_auto",
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
        peer="cursor_auto",
        status="ok",
        summary="Check worldview RO_QUERY gate.",
    )
    empty = PeerBriefV1(
        brief_id="brief-empty",
        help_id="help-empty",
        run_id=RUN_ID,
        peer="cursor_auto",
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
