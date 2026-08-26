"""2026-08-25: the chat_turn_pre/chat_turn_post triggers and their join key.

Hub now brackets each Orion-mode chat turn with two captures
(services/orion-hub/scripts/chat_turn_affect.py). For that pair to be worth
anything downstream, two things must hold on the published event:

* `trigger` must survive as the real label, not be silently flattened into
  "manual" -- the old `"ambient" if value == "ambient" else "manual"` clamp
  did exactly that to anything it did not recognize, which was harmless with
  two labels and actively misleading with four (a capture Orion triggered
  itself would be indistinguishable from Juniper pressing "Check now").
* `chat_correlation_id` must reach the event WITHOUT disturbing the existing
  `correlation_id`, which means something else and must keep meaning it.
"""

from __future__ import annotations

import uuid

import pytest

from app.main import _VALID_TRIGGERS, CaptureAndAssessRequest, _normalize_trigger, service
from orion.schemas.affectgpt import AffectGptAssessRequestPayload, AffectGptAssessResultPayload


@pytest.mark.parametrize("trigger", sorted(_VALID_TRIGGERS))
def test_every_valid_trigger_survives_normalization(trigger):
    assert _normalize_trigger(trigger) == trigger


@pytest.mark.parametrize("bad", ["chat_turn", "CHAT_TURN_PRE", "", None, 7, ["ambient"], {"a": 1}])
def test_unknown_trigger_clamps_to_manual(bad):
    """Still fail-safe: JuniperMultimodalAffectV1.trigger is a Literal with
    no handler around it deep inside _wrap_event, so an unrecognized value
    must not reach it and raise."""
    assert _normalize_trigger(bad) == "manual"


def test_new_triggers_are_accepted_at_the_api_boundary():
    req = CaptureAndAssessRequest(trigger="chat_turn_post", chat_correlation_id="turn-9")
    assert req.trigger == "chat_turn_post"
    assert req.chat_correlation_id == "turn-9"


def test_chat_correlation_id_defaults_to_none_for_existing_callers():
    """Hub's manual button and ambient loop send no such field; they must
    keep validating exactly as before."""
    assert CaptureAndAssessRequest(trigger="ambient").chat_correlation_id is None
    assert CaptureAndAssessRequest().trigger == "manual"


def test_wrap_event_carries_both_join_axes_independently():
    capture_corr = uuid.uuid4()
    event = service._wrap_event(
        AffectGptAssessResultPayload(ok=True, raw_response="looks focused"),
        AffectGptAssessRequestPayload(video_path="/s/clip.mp4", audio_path="/s/clip.wav"),
        trigger="chat_turn_pre",
        corr_id=capture_corr,
        chat_correlation_id="turn-42",
    )
    assert event.trigger == "chat_turn_pre"
    # Two distinct ids, not one reused for both purposes.
    assert event.correlation_id == str(capture_corr)
    assert event.chat_correlation_id == "turn-42"
    assert event.correlation_id != event.chat_correlation_id


def test_wrap_event_leaves_chat_correlation_id_unset_for_manual():
    event = service._wrap_event(
        AffectGptAssessResultPayload(ok=True),
        AffectGptAssessRequestPayload(video_path="/s/c.mp4", audio_path="/s/c.wav"),
        trigger="manual",
        corr_id=uuid.uuid4(),
    )
    assert event.trigger == "manual"
    assert event.chat_correlation_id is None


def test_chat_correlation_id_survives_a_failed_capture():
    """A capture that fails still publishes an event (that behaviour is
    load-bearing -- see trigger_assessment's own comment). If the join key
    were dropped on the failure path, a turn whose pre-capture failed would
    be unjoinable to its own post-capture, which is exactly when knowing
    'the pre leg failed' matters most."""
    event = service._wrap_event(
        AffectGptAssessResultPayload(ok=False, error="capture failed", error_code="capture_failed"),
        AffectGptAssessRequestPayload(video_path="", audio_path=""),
        trigger="chat_turn_post",
        corr_id=uuid.uuid4(),
        chat_correlation_id="turn-43",
    )
    assert event.ok is False
    assert event.trigger == "chat_turn_post"
    assert event.chat_correlation_id == "turn-43"
