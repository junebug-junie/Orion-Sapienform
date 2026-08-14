"""Session lifecycle and utterance shape for a room turn.

These test the LIFECYCLE (mint -> resume -> recover), not just the arithmetic
of any single call: a first turn that mints and a second that resumes is the
whole reason Claude is a participant rather than a stateless helper, and
hand-injecting a session id would test neither.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.claude_session import ClaudeTurnResult
from app.main import build_subprocess_env, parse_request, room_key, run_turn
from app.room_prompt import build_turn_prompt, filtered_summary
from app.session_store import get_session, get_or_create_session
from app.settings import Settings
from orion.schemas.room_claude import RoomClaudeRequestV1, RoomTranscriptEntryV1


@pytest.fixture()
def settings(tmp_path) -> Settings:
    # Workspace is redirected into tmp_path because run_turn creates it for
    # real (the container's /data is a volume, so the dir cannot exist at
    # image build time). Without this the suite would try to mkdir /data on
    # whatever machine runs it.
    return Settings(
        ROOM_COMPANION_SESSION_STATE_PATH=str(tmp_path / "sessions.json"),
        ROOM_COMPANION_CLAUDE_CONFIG_DIR="/root/.claude",
        ROOM_COMPANION_WORKSPACE=str(tmp_path / "workspace"),
    )


def _request(**kw) -> RoomClaudeRequestV1:
    base = dict(room_id="hub-direct", invited_by="Juniper", prompt="hey Claude")
    base.update(kw)
    return RoomClaudeRequestV1(**base)


def _ok(text="hello", session="sess-1", cost=0.004) -> ClaudeTurnResult:
    return ClaudeTurnResult(
        ok=True, text=text, claude_session_id=session, model="claude-sonnet-5",
        cost_usd=cost, duration_ms=1200, exit_code=0,
    )


def test_first_turn_mints_a_session_and_second_turn_resumes_it(settings):
    """The lifecycle, not the arithmetic."""
    calls = []

    def _capture(prompt, **kw):
        calls.append(kw)
        return _ok()

    with patch("app.main.run_room_turn", side_effect=_capture):
        run_turn(settings, _request())
        run_turn(settings, _request())

    assert calls[0]["resume"] is False, "first turn must mint, not resume"
    assert calls[1]["resume"] is True, "second turn must resume the same session"
    assert calls[0]["session_id"] == calls[1]["session_id"]


def test_system_prompt_only_on_first_turn(settings):
    """Re-appending the persona every turn would be wasted tokens: the CLI
    already holds it under --resume."""
    calls = []

    with patch("app.main.run_room_turn", side_effect=lambda p, **kw: calls.append(kw) or _ok()):
        run_turn(settings, _request())
        run_turn(settings, _request())

    assert calls[0]["append_system_prompt"], "first turn needs the room framing"
    assert calls[1]["append_system_prompt"] is None


def test_transcript_only_sent_on_first_turn(settings):
    """After the session exists, Claude has the conversation on its own side.
    Re-sending it would cost tokens and hand Claude two copies to reconcile."""
    prompts = []
    transcript = [
        RoomTranscriptEntryV1(speaker_id="orion", speaker_name="Oríon", text="I'd like a peer."),
    ]

    with patch("app.main.run_room_turn", side_effect=lambda p, **kw: prompts.append(p) or _ok()):
        run_turn(settings, _request(transcript=transcript))
        run_turn(settings, _request(transcript=transcript))

    assert "I'd like a peer." in prompts[0]
    assert "I'd like a peer." not in prompts[1]
    assert prompts[1] == "Juniper: hey Claude", "later turns carry the speaker and nothing else"


def test_lost_session_is_forgotten_so_the_room_can_recover(settings):
    """A --resume against a session the CLI no longer has fails identically
    forever. Without forgetting it, the room is permanently wedged."""
    key = room_key(_request())
    session_id, _ = get_or_create_session(settings.ROOM_COMPANION_SESSION_STATE_PATH, key)

    failure = ClaudeTurnResult(
        ok=False, text="", claude_session_id=None, model=None, cost_usd=0.0,
        duration_ms=10, exit_code=1, error="No conversation found with session ID: abc",
    )
    with patch("app.main.run_room_turn", return_value=failure):
        utterance = run_turn(settings, _request())

    assert not utterance.ok
    assert get_session(settings.ROOM_COMPANION_SESSION_STATE_PATH, key) is None, (
        "a lost session must be dropped so the next turn can mint a fresh one"
    )


def test_ordinary_failure_does_not_drop_the_session(settings):
    """A timeout is not a lost session -- discarding continuity on every
    transient error would quietly amnesia the room."""
    key = room_key(_request())
    get_or_create_session(settings.ROOM_COMPANION_SESSION_STATE_PATH, key)

    failure = ClaudeTurnResult(
        ok=False, text="", claude_session_id=None, model=None, cost_usd=0.0,
        duration_ms=10, exit_code=-1, error="timeout after 180s",
    )
    with patch("app.main.run_room_turn", return_value=failure):
        run_turn(settings, _request())

    assert get_session(settings.ROOM_COMPANION_SESSION_STATE_PATH, key) is not None


def test_failed_turn_still_publishes_an_utterance(settings):
    """Silence is indistinguishable from Claude choosing not to speak, which
    is what makes an outage invisible. Failures must be audible."""
    failure = ClaudeTurnResult(
        ok=False, text="", claude_session_id=None, model=None, cost_usd=0.0,
        duration_ms=10, exit_code=0, error="401 OAuth access token is invalid",
    )
    with patch("app.main.run_room_turn", return_value=failure):
        utterance = run_turn(settings, _request())

    assert utterance.ok is False
    assert "401" in (utterance.error or "")
    assert utterance.responder.participant_name == "Claude"


def test_utterance_carries_cost_model_and_responder_identity(settings):
    with patch("app.main.run_room_turn", return_value=_ok(cost=0.0123)):
        utterance = run_turn(settings, _request(correlation_id="corr-7"))

    assert utterance.cost_usd == pytest.approx(0.0123)
    assert utterance.model == "claude-sonnet-5"
    assert utterance.correlation_id == "corr-7"
    assert utterance.responder.participant_id == "claude"
    assert utterance.responder.participant_kind == "peer_ai"


def test_parse_request_accepts_bare_and_enveloped_payloads():
    payload = _request().model_dump(mode="json")
    assert parse_request(payload) is not None
    assert parse_request({"kind": "room.claude.request.v1", "payload": payload}) is not None
    assert parse_request({"nonsense": True}) is None
    assert parse_request("not a dict") is None


def test_room_key_separates_platforms():
    assert room_key(_request(platform="hub", room_id="r")) == "hub:r"
    assert room_key(_request(platform="aitown", room_id="r")) == "aitown:r"


def test_first_turn_prompt_includes_roster_and_speaker():
    prompt = build_turn_prompt(
        prompt="what do you think?",
        invited_by_name="Juniper",
        transcript=[],
        social_memory_summary={"room": {"active_participants": ["Juniper", "Oríon"]}},
        first_turn=True,
    )
    assert "Juniper, Oríon" in prompt
    assert prompt.endswith("Juniper: what do you think?")


def test_filtered_summary_drops_keys_outside_the_allowlist():
    """Third-party-bound payload: 'everything we happened to have' is the
    wrong default."""
    summary = {
        "room": {"active_participants": ["Juniper"]},
        "stance": {"summary": "warm"},
        "self_state": {"drives": {"curiosity": 0.9}},
        "recall_debug": {"documents": ["private journal entry"]},
    }
    filtered = filtered_summary(summary)
    assert set(filtered) == {"room", "stance"}
    assert "self_state" not in filtered
    assert "recall_debug" not in filtered


def test_build_subprocess_env_sets_config_dir(settings):
    env = build_subprocess_env(settings)
    assert env["CLAUDE_CONFIG_DIR"] == "/root/.claude"
