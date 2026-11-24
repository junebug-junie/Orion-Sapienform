"""The chat stance's reverie glimpse: the newest question a quiet-stretch reverie left (reveries that ask,
2026-11-24), read from felt_state_reader's ``latest_reverie_question`` lane, labelled as Orion's own."""
from __future__ import annotations

import json

from app.chat_stance import REVERIE_GLIMPSE_PREFIX, _project_reverie_glimpse

_QUESTION = "Why do turns I call unlikely to run long run long 37% of the time?"
_RENDERED = ("Something I was turning over on my own (reverie, 11-24), "
             "not something Juniper and I discussed: " + REVERIE_GLIMPSE_PREFIX + _QUESTION)


def _payload(**overrides):
    payload = {"question_id": "q-1", "question": _QUESTION, "seed_kind": "lost_bet",
               "routed_to": "focus:f-1", "created_at": "2026-11-24T03:05:00+00:00"}
    payload.update(overrides)
    return payload


def test_none_when_ctx_key_absent():
    assert _project_reverie_glimpse({}) is None


def test_none_when_question_blank_or_missing():
    assert _project_reverie_glimpse({"latest_reverie_question": _payload(question="   ")}) is None
    assert _project_reverie_glimpse({"latest_reverie_question": {"created_at": "2026-11-24T03:05:00+00:00"}}) is None


def test_question_is_rendered_as_orions_own_with_its_date():
    assert _project_reverie_glimpse({"latest_reverie_question": _payload()}) == _RENDERED


def test_json_string_payload_is_accepted_and_garbage_fails_open():
    assert _project_reverie_glimpse({"latest_reverie_question": json.dumps(_payload())}) == _RENDERED
    assert _project_reverie_glimpse({"latest_reverie_question": "{not valid json"}) is None
    assert _project_reverie_glimpse({"latest_reverie_question": 12345}) is None


def test_retired_narration_key_is_ignored():
    ctx = {"latest_reverie_thought": {"interpretation": "The coalition is fixated.", "hollow": False}}
    assert _project_reverie_glimpse(ctx) is None


def test_only_the_question_text_crosses_into_the_prompt():
    result = _project_reverie_glimpse({"latest_reverie_question": _payload(seed_kind="absence", routed_to="focus:x")})
    assert "focus:x" not in result and "absence" not in result and "q-1" not in result


def test_glimpse_is_never_presented_as_juniper_or_as_shared():
    result = _project_reverie_glimpse({"latest_reverie_question": _payload(
        question="Juniper told me the disk is flashed; did it change the GPU limits?")})
    assert result.startswith("Something I was turning over on my own (reverie, ")
    prefix = result.split(": ", 1)[0]
    assert "Juniper told me" not in prefix
