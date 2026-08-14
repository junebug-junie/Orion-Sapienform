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
from app.room_prompt import SYSTEM_PROMPT, build_turn_prompt, filtered_summary
from app.session_store import get_session, peek_or_mint_session, remember_session
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
    assert calls[1]["resume"] is True, "second turn must resume"
    # Turn 2 resumes the id the CLI REPORTED on turn 1, not the one that was
    # optimistically minted for it -- that is the point of recording the
    # reported id rather than trusting --session-id was honored.
    assert calls[1]["session_id"] == _ok().claude_session_id


def test_system_prompt_is_passed_on_every_turn_including_resumes(settings):
    """The room framing must be re-sent on resumed turns.

    This test previously asserted the OPPOSITE -- that the flag belongs only
    on the first turn because "the CLI already holds it under --resume". That
    rationale was false, and the test locked in a bug that made the feature
    stop being the feature after one turn: verified live, a resumed turn
    without the flag reverts to Claude Code's default CLI persona.

    Asserts the flag on turn 2 specifically, not merely that it is non-None
    somewhere, because turn 2 is where the regression lived.
    """
    calls = []

    with patch("app.main.run_room_turn", side_effect=lambda p, **kw: calls.append(kw) or _ok()):
        run_turn(settings, _request())
        run_turn(settings, _request())

    assert calls[1]["resume"] is True, "second turn must be the resumed one"
    assert calls[0]["append_system_prompt"] == SYSTEM_PROMPT
    assert calls[1]["append_system_prompt"] == SYSTEM_PROMPT, (
        "a resumed turn without the room framing reverts to the default "
        "assistant persona -- see the docstring"
    )


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
    remember_session(settings.ROOM_COMPANION_SESSION_STATE_PATH, key, "sess-1")

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
    remember_session(settings.ROOM_COMPANION_SESSION_STATE_PATH, key, "sess-1")

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


def test_session_is_not_persisted_until_a_turn_succeeds(settings):
    """Writing the minted uuid at mint time recorded a session the CLI had
    never created, so a failed first turn left the map pointing at nothing and
    the next turn burned a doomed --resume before recovering."""
    key = room_key(_request())
    failure = ClaudeTurnResult(
        ok=False, text="", claude_session_id=None, model=None, cost_usd=0.0,
        duration_ms=10, exit_code=1, error="timeout after 180s",
    )
    with patch("app.main.run_room_turn", return_value=failure):
        run_turn(settings, _request())

    assert get_session(settings.ROOM_COMPANION_SESSION_STATE_PATH, key) is None


def test_session_recorded_is_the_one_the_cli_reported(settings):
    """--session-id is honored today, but nothing would notice if it stopped
    being -- and a drifted map resumes the wrong conversation."""
    key = room_key(_request())
    with patch("app.main.run_room_turn", return_value=_ok(session="cli-chosen-id")):
        run_turn(settings, _request())

    assert get_session(settings.ROOM_COMPANION_SESSION_STATE_PATH, key) == "cli-chosen-id"


def test_social_memory_summary_is_filtered_before_it_reaches_claude(settings):
    """The privacy boundary must be applied in the real call path, not merely
    available as a helper -- an uncalled safeguard is decoration."""
    prompts = []
    summary = {
        "room": {"active_participants": ["Juniper", "Oríon"]},
        "self_state": {"drives": {"curiosity": 0.9}},
        "recall_debug": {"documents": ["a private journal entry"]},
    }
    with patch("app.main.run_room_turn", side_effect=lambda p, **kw: prompts.append(p) or _ok()):
        run_turn(settings, _request(social_memory_summary=summary))

    assert "Juniper, Oríon" in prompts[0]
    assert "curiosity" not in prompts[0]
    assert "journal" not in prompts[0]


def test_missing_session_matcher_does_not_fire_on_unrelated_errors(settings):
    """The 'session'+'not found' clause is broad; a false positive silently
    amnesias a live room."""
    from app.main import _looks_like_missing_session

    assert _looks_like_missing_session("No conversation found with session ID: abc")
    assert _looks_like_missing_session("session abc not found")
    assert not _looks_like_missing_session("timeout after 180s")
    assert not _looks_like_missing_session("401 OAuth access token is invalid")
    assert not _looks_like_missing_session("model not found")
    assert not _looks_like_missing_session(None)


def test_corrupt_state_file_is_quarantined_not_silently_overwritten(settings, tmp_path):
    """Returning {} on a parse error and then rewriting would discard EVERY
    room's mapping, not just the bad entry."""
    from app.session_store import _read_all, remember_session

    path = tmp_path / "sessions.json"
    path.write_bytes(b"\xff\xfe not valid utf-8 or json")

    assert _read_all(path) == {}
    assert path.with_suffix(".json.corrupt").exists(), "bad content must be kept for inspection"

    remember_session(path, "hub:r", "fresh")
    assert get_session(path, "hub:r") == "fresh"
