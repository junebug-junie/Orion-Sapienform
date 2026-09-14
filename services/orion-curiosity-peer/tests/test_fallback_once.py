"""Claude fallback exactly once + dual-failure / budget refuse briefs."""

from __future__ import annotations

from typing import Any

import pytest

from app.claude_fallback import CONTRACTOR_PEER_MARKER, run_claude_fallback
from app.cursor_errors import TokenUnavailable, classify_cursor_failure
from app.worker import handle_help_request
from orion.dev_economics.cursor_limit_events import CursorLimitObservation
from orion.schemas.curiosity_peer import HelpRequestV1, PeerBriefV1


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


def test_classify_token_unavailable() -> None:
    assert classify_cursor_failure(TokenUnavailable("dry")) == "token_unavailable"
    assert classify_cursor_failure(RuntimeError("boom")) == "other"
    assert (
        classify_cursor_failure(RuntimeError("401 Unauthorized: invalid api key"))
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
        persist=persisted.append,
    )
    assert calls == {"cursor": 1, "claude": 1}
    assert brief.status == "ok"
    assert brief.peer == "claude_room"
    assert "ok summary" in brief.summary
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


def test_run_claude_fallback_publishes_and_waits_once() -> None:
    published: list[Any] = []
    waits = {"n": 0}

    def publish(req: Any) -> None:
        published.append(req)

    def wait(request_id: str, *, timeout_sec: float) -> str:
        waits["n"] += 1
        assert request_id == published[0].request_id
        assert timeout_sec > 0
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
    assert brief.peer == "claude_room"
    assert brief.status == "ok"
    assert "room notes" in brief.summary


def test_run_claude_fallback_failure_raises() -> None:
    def publish(_req: Any) -> None:
        return None

    def wait(_request_id: str, *, timeout_sec: float) -> str:
        raise TimeoutError("no utterance")

    with pytest.raises(TimeoutError):
        run_claude_fallback(
            _help(),
            publish_request=publish,
            wait_utterance=wait,
        )
