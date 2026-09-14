"""Claude fallback exactly once + dual-failure / budget refuse briefs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import SecretStr

from app.claude_fallback import CONTRACTOR_PEER_MARKER, run_claude_fallback
from app.cursor_errors import TokenUnavailable, classify_cursor_failure
from app.settings import Settings
from app.worker import (
    _default_persist,
    _map_claude_result,
    _run_coro_threadsafe,
    apply_peer_brief_consumed,
    handle_help_request,
    make_bus_claude_fallback,
)
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.bus.codec import OrionCodec
from orion.dev_economics.cursor_limit_events import CursorLimitObservation
from orion.schemas.curiosity_peer import (
    PEER_BRIEF_KIND,
    HelpRequestV1,
    PeerBriefConsumedV1,
    PeerBriefV1,
)
from orion.schemas.room_claude import ExternalRoomResponderV1, RoomClaudeUtteranceV1


def _help(**overrides: Any) -> HelpRequestV1:
    base = dict(
        help_id="help-1",
        run_id="run-1",
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


@dataclass
class _ClearClaudeLimit:
    observed: bool = True
    state: str = "clear"
    staleness_sec: float | None = 1.0


def _clear_claude() -> _ClearClaudeLimit:
    return _ClearClaudeLimit()


def test_classify_token_unavailable() -> None:
    assert classify_cursor_failure(TokenUnavailable("dry")) == "token_unavailable"
    assert classify_cursor_failure(RuntimeError("boom")) == "other"
    assert (
        classify_cursor_failure(RuntimeError("401 Unauthorized: invalid api key"))
        == "token_unavailable"
    )
    # Bare "token" must not false-positive (narrowed markers).
    assert classify_cursor_failure(RuntimeError("tokenizing the input")) == "other"
    # Desktop CLI login markers.
    assert (
        classify_cursor_failure(RuntimeError("not logged in — please run agent login"))
        == "token_unavailable"
    )
    assert (
        classify_cursor_failure(RuntimeError("Please run `agent login` to continue"))
        == "token_unavailable"
    )


def test_cursor_token_failure_tries_claude_once() -> None:
    calls = {"cursor": 0, "claude": 0}
    persisted: list[PeerBriefV1] = []

    def cursor(*_a: Any, **_k: Any) -> PeerBriefV1:
        calls["cursor"] += 1
        raise TokenUnavailable("cursor tokens dry")

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        return '{"summary": "ok summary", "evidence_pointers": ["policy.py"]}'

    brief = handle_help_request(
        _help(),
        cursor=cursor,
        claude=claude,
        observe_limit=lambda: _clear_budget(),
        observe_claude_limit=_clear_claude,
        persist=persisted.append,
    )
    assert calls == {"cursor": 1, "claude": 1}
    assert brief.status == "ok"
    assert brief.peer == "claude_room"
    assert "ok summary" in brief.summary
    assert "conversation-only" in brief.summary.lower()
    assert brief.evidence_pointers == []
    assert persisted == [brief]


def test_claude_budget_refuse_skips_claude_spend() -> None:
    calls = {"cursor": 0, "claude": 0}
    persisted: list[PeerBriefV1] = []

    def cursor(*_a: Any, **_k: Any) -> PeerBriefV1:
        calls["cursor"] += 1
        raise TokenUnavailable("dry")

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        raise AssertionError("claude must not spend")

    brief = handle_help_request(
        _help(),
        cursor=cursor,
        claude=claude,
        observe_limit=lambda: _clear_budget(),
        observe_claude_limit=lambda: _ClearClaudeLimit(
            observed=False, state="unknown", staleness_sec=None
        ),
        persist=persisted.append,
    )
    assert calls == {"cursor": 1, "claude": 0}
    assert brief.status == "refused_budget"
    assert brief.peer == "claude_room"
    assert brief.refusal_reason and "claude_budget_unobserved" in brief.refusal_reason
    assert persisted == [brief]


def test_both_fail_yields_failed_brief() -> None:
    calls = {"cursor": 0, "claude": 0}
    persisted: list[PeerBriefV1] = []

    def cursor(*_a: Any, **_k: Any) -> PeerBriefV1:
        calls["cursor"] += 1
        raise TokenUnavailable("no tokens")

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        raise RuntimeError("claude room timed out")

    brief = handle_help_request(
        _help(),
        cursor=cursor,
        claude=claude,
        observe_limit=lambda: _clear_budget(),
        observe_claude_limit=_clear_claude,
        persist=persisted.append,
    )
    assert calls == {"cursor": 1, "claude": 1}
    assert brief.status == "failed"
    assert brief.refusal_reason
    assert "claude" in brief.refusal_reason.lower() or "timed out" in brief.refusal_reason.lower()
    assert persisted == [brief]


def test_other_cursor_failure_skips_claude() -> None:
    calls = {"cursor": 0, "claude": 0}

    def cursor(*_a: Any, **_k: Any) -> PeerBriefV1:
        calls["cursor"] += 1
        raise RuntimeError("agent crashed")

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        return "should not run"

    brief = handle_help_request(
        _help(),
        cursor=cursor,
        claude=claude,
        observe_limit=lambda: _clear_budget(),
        persist=lambda _b: None,
    )
    assert calls == {"cursor": 1, "claude": 0}
    assert brief.status == "failed"
    assert brief.peer == "cursor_auto"


def test_budget_refuse_skips_peers() -> None:
    calls = {"cursor": 0, "claude": 0}
    persisted: list[PeerBriefV1] = []

    def cursor(*_a: Any, **_k: Any) -> PeerBriefV1:
        calls["cursor"] += 1
        raise AssertionError("cursor must not run")

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        raise AssertionError("claude must not run")

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
    assert brief.refusal_reason == "budget_unobserved"
    assert persisted == [brief]


def test_cursor_success_skips_claude() -> None:
    calls = {"claude": 0}

    def cursor(*_a: Any, **_k: Any) -> PeerBriefV1:
        return PeerBriefV1(
            brief_id="brief-c",
            help_id="help-1",
            run_id="run-1",
            prior_id="prior-1",
            peer="cursor_auto",
            status="ok",
            summary="from cursor",
        )

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        return "nope"

    brief = handle_help_request(
        _help(),
        cursor=cursor,
        claude=claude,
        observe_limit=lambda: _clear_budget(),
        persist=lambda _b: None,
    )
    assert calls["claude"] == 0
    assert brief.peer == "cursor_auto"
    assert brief.status == "ok"


def test_map_claude_result_strips_evidence_pointers() -> None:
    raw = PeerBriefV1(
        brief_id="b1",
        help_id="help-1",
        run_id="run-1",
        peer="claude_room",
        status="ok",
        summary="notes",
        evidence_pointers=["worker.py", "policy.py"],
    )
    mapped = _map_claude_result(_help(), raw)
    assert mapped.evidence_pointers == []
    assert "conversation-only" in mapped.summary.lower()


def test_run_claude_fallback_publishes_and_waits_once() -> None:
    published: list[Any] = []
    waits = {"n": 0}
    order: list[str] = []

    def publish(req: Any) -> None:
        order.append("publish")
        published.append(req)

    def wait(
        request_id: str,
        *,
        timeout_sec: float,
        publish_request: Any = None,
    ) -> str:
        waits["n"] += 1
        order.append("subscribe")
        assert timeout_sec > 0
        if publish_request is not None:
            publish_request()
        assert request_id == published[0].request_id
        return '{"summary": "room notes", "evidence_pointers": []}'

    brief = run_claude_fallback(
        _help(),
        publish_request=publish,
        wait_utterance=wait,
        timeout_sec=12.0,
    )
    assert len(published) == 1
    req = published[0]
    assert req.trigger == "auto"
    assert req.invited_by in ("orion", "Orion", "system", "orion-system")
    assert CONTRACTOR_PEER_MARKER in req.prompt
    assert waits["n"] == 1
    assert order == ["subscribe", "publish"]
    assert brief.peer == "claude_room"
    assert brief.status == "ok"
    assert "room notes" in brief.summary


def test_run_claude_fallback_failure_raises() -> None:
    def publish(_req: Any) -> None:
        return None

    def wait(
        _request_id: str,
        *,
        timeout_sec: float,
        publish_request: Any = None,
    ) -> str:
        if publish_request is not None:
            publish_request()
        raise TimeoutError("no utterance")

    with pytest.raises(TimeoutError):
        run_claude_fallback(
            _help(),
            publish_request=publish,
            wait_utterance=wait,
        )


def test_bus_claude_fallback_unwraps_titanium_payload_envelope() -> None:
    """Room companion publishes Titanium envelopes with utterance under `payload`."""
    order: list[str] = []
    settings = Settings()
    brief_body = '{"summary": "from enveloped utterance", "evidence_pointers": []}'

    class _SubCtx:
        async def __aenter__(self) -> "_SubCtx":
            order.append("subscribe")
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

    class FakeBus:
        def __init__(self) -> None:
            self._request_id: str | None = None

        async def publish(self, channel: str, payload: dict[str, Any]) -> None:
            order.append("publish")
            assert channel == settings.CHANNEL_ROOM_CLAUDE_REQUEST
            self._request_id = str(payload["request_id"])

        def subscribe(self, channel: str) -> _SubCtx:
            assert channel == settings.CHANNEL_ROOM_CLAUDE_UTTERANCE
            return _SubCtx()

        async def iter_messages(self, _pubsub: Any):
            assert self._request_id is not None
            utt = RoomClaudeUtteranceV1(
                request_id=self._request_id,
                room_id="curiosity-contractor-peer",
                responder=ExternalRoomResponderV1(
                    participant_id="claude",
                    participant_name="Claude",
                ),
                text=brief_body,
                model="claude-sonnet",
                cost_usd=0.01,
                ok=True,
            )
            envelope = {"payload": utt.model_dump(mode="json")}
            yield {"type": "message", "data": json.dumps(envelope)}

    bus = FakeBus()
    claude = make_bus_claude_fallback(settings=settings, bus=bus, timeout_sec=2.0)
    brief = claude(_help(), context_pack="")
    assert order[0] == "subscribe"
    assert "publish" in order
    assert order.index("subscribe") < order.index("publish")
    assert brief.peer == "claude_room"
    assert brief.status == "ok"
    assert "from enveloped utterance" in brief.summary


def test_run_coro_threadsafe_uses_captured_loop() -> None:
    """Worker-thread bus ops must not asyncio.run on a foreign OrionBusAsync loop."""
    seen: list[str] = []

    async def _main() -> None:
        loop = asyncio.get_running_loop()

        async def _work() -> str:
            seen.append("on-loop")
            return "ok"

        def _from_thread() -> str:
            return _run_coro_threadsafe(_work(), loop, timeout=2.0)

        out = await asyncio.to_thread(_from_thread)
        assert out == "ok"

    asyncio.run(_main())
    assert seen == ["on-loop"]


def test_default_persist_publishes_base_envelope_codec_roundtrip() -> None:
    class FakeGraph:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        def graph_query(self, cypher: str, params: Any = None) -> list:
            self.calls.append((cypher, params))
            return []

    published: list[tuple[str, BaseEnvelope]] = []

    class FakeBus:
        def publish(self, channel: str, payload: BaseEnvelope) -> None:
            published.append((channel, payload))

    graph = FakeGraph()
    brief = PeerBriefV1(
        brief_id="brief-merge-1",
        help_id="help-1",
        run_id="run-1",
        prior_id="prior-1",
        peer="claude_room",
        status="ok",
        summary="merge me",
        evidence_pointers=[],
    )
    settings = Settings(
        ORION_CURIOSITY_GRAPH_HOST="127.0.0.1",
        ORION_CURIOSITY_GRAPH_PORT="6379",
        ORION_CURIOSITY_GRAPH_USER="orion_curiosity",
        ORION_CURIOSITY_GRAPH_PASSWORD=SecretStr("secret"),
        ORION_CURIOSITY_GRAPH_OWN="orion_worldview",
    )
    _default_persist(brief, settings=settings, bus=FakeBus(), graph_client=graph)
    assert len(graph.calls) == 1
    cypher, params = graph.calls[0]
    assert "MERGE" in cypher
    assert "PeerBrief" in cypher
    assert params["brief_id"] == "brief-merge-1"
    assert published and isinstance(published[0][1], BaseEnvelope)
    env = published[0][1]
    assert env.kind == PEER_BRIEF_KIND
    decoded = OrionCodec().decode(OrionCodec().encode(env))
    assert decoded.ok
    assert decoded.envelope.kind == PEER_BRIEF_KIND
    assert decoded.envelope.payload["brief_id"] == "brief-merge-1"


def test_default_persist_skips_graph_when_unconfigured(caplog: pytest.LogCaptureFixture) -> None:
    published: list[tuple[str, BaseEnvelope]] = []

    class FakeBus:
        def publish(self, channel: str, payload: BaseEnvelope) -> None:
            published.append((channel, payload))

    brief = PeerBriefV1(
        brief_id="brief-bus-only",
        help_id="help-1",
        run_id="run-1",
        peer="cursor_auto",
        status="ok",
        summary="bus only",
    )
    with caplog.at_level("WARNING"):
        _default_persist(brief, settings=Settings(), bus=FakeBus())
    assert published and published[0][1].payload["brief_id"] == "brief-bus-only"
    assert any("graph_unconfigured" in r.message for r in caplog.records)


def test_apply_peer_brief_consumed_marks_graph() -> None:
    class FakeGraph:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        def graph_query(self, cypher: str, params: Any = None) -> list:
            self.calls.append((cypher, params))
            return []

    graph = FakeGraph()
    payload = PeerBriefConsumedV1(brief_ids=["brief-a", "brief-b"]).model_dump(
        mode="json"
    )
    ids = apply_peer_brief_consumed(
        {"payload": payload},
        settings=Settings(),
        graph_client=graph,
    )
    assert ids == ["brief-a", "brief-b"]
    assert graph.calls
    cypher, params = graph.calls[0]
    assert "SET b.consumed = true" in cypher
    assert params["brief_ids"] == ["brief-a", "brief-b"]
