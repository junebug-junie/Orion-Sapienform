"""Strata structured output, done in the gateway (app/strata_structured.py; supersedes the judge-only retry #3292).

Strata only prompts and validates; its 502 throws the text away. These tests pin: Strata's own
directive and strict parse are reproduced, the one exact repair (Python-literal dicts), one
corrective retry, and the same failure message callers already handle when that fails too.
"""
from __future__ import annotations

import json
from typing import Any, Callable, List

import httpx
import pytest

from app import strata_structured as ss
from app import upstream_cancel
from app.llm_backend import _execute_openai_chat
from app.models import ChatBody
from app.settings import settings


# ── pure parts ──────────────────────────────────────────────────────────────────────────────

def test_take_over_moves_the_format_into_strata_s_own_directive() -> None:
    payload = {"messages": [{"role": "system", "content": "be Orion"}, {"role": "user", "content": "hi"}],
               "response_format": {"type": "json_object"}, "max_tokens": 9}
    sent, taken = ss.take_over(payload)
    assert taken and "response_format" not in sent and sent["max_tokens"] == 9
    assert sent["messages"][0]["content"] == 'be Orion\n\n' + ss.directive({"type": "object"})
    assert payload["messages"][0]["content"] == "be Orion"  # the caller's payload is not changed


def test_take_over_inserts_a_system_message_and_uses_the_named_schema() -> None:
    schema = {"type": "object", "properties": {"mood": {"type": "string"}}}
    sent, taken = ss.take_over({"messages": [{"role": "user", "content": "hi"}], "response_format": {
        "type": "json_schema", "json_schema": {"name": "Mood", "schema": schema}}})
    assert taken and sent["messages"][0] == {"role": "system", "content": ss.directive(schema)}


@pytest.mark.parametrize("payload", [
    {"messages": [], "response_format": {"type": "text"}},
    {"messages": []},
    {"messages": [], "response_format": {"type": "json_object"}, "tools": [{"type": "function"}]},
])
def test_take_over_leaves_everything_else_to_strata(payload) -> None:
    assert ss.take_over(payload) == (payload, False)


def test_valid_reply_is_canonical_like_strata() -> None:
    v = ss.check('```json\n{"a": 1,  "b": "x"}\n```', "stop")
    assert v.ok and v.text == '{"a":1,"b":"x"}' and v.repaired is None


def test_python_literal_dict_is_repaired_exactly() -> None:
    # The 2026-11-13 stance_react failure: "Expecting property name ... line 1 column 2 (char 1)".
    v = ss.check("{'mood': 'calm', 'n': 2, 'ok': True, 'none': None, 'xs': ['a', \"b's\"]}", "stop")
    assert v.ok and v.repaired == "python_literal"
    assert json.loads(v.text) == {"mood": "calm", "n": 2, "ok": True, "none": None, "xs": ["a", "b's"]}


@pytest.mark.parametrize("text", [
    "{'a': __import__('os')}",      # a call: never evaluated
    "{'a': float('nan')}",
    "{1: 'non-string key'}",
    "['not', 'a', 'dict']",
])
def test_no_repair_beyond_plain_literals(text) -> None:
    v = ss.check(text, "stop")
    assert not v.ok and v.error.startswith("model did not return valid JSON")


def test_bare_quote_inside_a_string_is_not_guessed_at() -> None:
    v = ss.check('{"why": "Juniper said "hello" to me", "x": 1}', "stop")
    assert not v.ok and "Expecting ',' delimiter" in v.error and v.retryable


def test_duplicate_keys_and_nan_are_refused_like_strata() -> None:
    assert not ss.check('{"a": 1, "a": 2}', "stop").ok
    assert not ss.check('{"a": NaN}', "stop").ok
    assert not ss.check("{'a': 1, 'a': 2}", "stop").ok  # the repair keeps Strata's duplicate-key rule


def test_adjacent_strings_are_never_joined() -> None:
    # Python would read "x" "b" as "xb": the bare-quote failure turned into the wrong text.
    assert not ss.check('{"a": "x" "b"}', "stop").ok
    assert not ss.check("{'a': 'she said ' 'hi'}", "stop").ok


def test_a_reply_without_finish_reason_is_not_called_incomplete() -> None:
    assert ss.check('{"a": 1}', None).ok


def test_an_unfinished_reply_is_incomplete_and_not_retried() -> None:
    v = ss.check('{"a": "cut', "length")
    assert not v.ok and not v.retryable and "finish_reason=length" in v.error


def test_corrective_messages_show_the_model_its_reply_and_the_error() -> None:
    msgs = ss.corrective_messages([{"role": "user", "content": "q"}], "{'bad'", "model did not return valid JSON: x")
    assert msgs[1] == {"role": "assistant", "content": "{'bad'"}
    assert "x" in msgs[2]["content"] and '\\"' in msgs[2]["content"]


# ── the real request path, against a fake Strata ────────────────────────────────────────────

def _reply(content: str, finish: str = "stop") -> dict:
    return {"model": "Active-GGUF-Model", "choices": [{"message": {"content": content}, "finish_reason": finish}]}


def _strata(monkeypatch, replies: List[dict]) -> List[dict]:
    """A fake Strata that answers ``replies`` in order and records each request body."""
    seen: List[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=replies[len(seen) - 1])

    monkeypatch.setattr(upstream_cancel, "cancellable_transport", lambda: httpx.MockTransport(handler))
    return seen


@pytest.fixture(autouse=True)
def _no_lane_routing(monkeypatch):
    monkeypatch.setattr(settings, "llm_lane_routing_enabled", False)


def _body() -> ChatBody:
    return ChatBody(route="chat", messages=[{"role": "user", "content": "stance?"}],
                    options={"response_format": {"type": "json_object"}}, trace_id="corr-json")


def _run(backend: str = "strata") -> dict:
    return _execute_openai_chat(_body(), "m", "http://circe:8011", backend, route="chat", served_by="circe-worker-chat")


def test_a_valid_reply_is_one_call_and_strata_never_sees_response_format(monkeypatch) -> None:
    seen = _strata(monkeypatch, [_reply('{"stance": "warm"}')])
    result = _run()
    assert len(seen) == 1 and "response_format" not in seen[0]
    assert ss.directive({"type": "object"}) in seen[0]["messages"][0]["content"]
    assert result["text"] == '{"stance":"warm"}' and "error" not in result["raw"]


def test_a_python_literal_is_repaired_without_a_second_call(monkeypatch) -> None:
    seen = _strata(monkeypatch, [_reply("{'stance': 'warm'}")])
    result = _run()
    assert len(seen) == 1 and json.loads(result["text"]) == {"stance": "warm"}


def test_a_bad_reply_gets_one_corrective_retry(monkeypatch) -> None:
    bad = '{"why": "she said "yay" today"}'
    seen = _strata(monkeypatch, [_reply(bad), _reply('{"why": "she said \\"yay\\" today"}')])
    result = _run()
    assert len(seen) == 2
    retry = seen[1]["messages"]
    assert retry[-2] == {"role": "assistant", "content": bad} and "not one valid JSON object" in retry[-1]["content"]
    assert json.loads(result["text"]) == {"why": 'she said "yay" today'}


def test_two_bad_replies_fail_with_strata_s_own_message(monkeypatch) -> None:
    seen = _strata(monkeypatch, [_reply('{"a": "x" "b"}'), _reply('{"a": "x" "b"}')])
    result = _run()
    assert len(seen) == 2 and result["text"] == ""
    assert result["raw"]["error"] == "upstream_http_5xx"
    details = result["raw"]["details"]
    assert details["reason"] == "http_502" and details["message"].startswith("model did not return valid JSON: ")


def test_a_cut_off_reply_is_not_retried(monkeypatch) -> None:
    seen = _strata(monkeypatch, [_reply('{"a": "cut', finish="length")])
    result = _run()
    assert len(seen) == 1 and "finish_reason=length" in result["raw"]["details"]["message"]


def test_other_backends_keep_response_format_and_one_call(monkeypatch) -> None:
    seen = _strata(monkeypatch, [_reply("{'not': 'checked here'}")])
    result = _run(backend="llamacpp")
    assert len(seen) == 1 and seen[0]["response_format"] == {"type": "json_object"}
    assert result["text"] == "{'not': 'checked here'}"


# ── review findings (2026-11-13) ────────────────────────────────────────────────────────────

# Strata 0.2.7's own words, copied from /opt/strata/terminal/serve/structured.py prepare_format. A
# change to the gateway's copy that Strata did not make must fail here, not pass silently.
STRATA_DIRECTIVE_HEAD = (
    "OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. "
    "No Markdown, headings, code fences, commentary, or text outside the JSON. "
    "Use every required field, correct types, and only allowed fields. "
    "Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n"
)


def test_directive_is_strata_s_literal_text() -> None:
    assert ss.directive({"type": "object"}) == STRATA_DIRECTIVE_HEAD + '{"type":"object"}'


@pytest.mark.parametrize("text", [
    r"{'formula': '\alpha + \beta'}",   # Python: \a is BEL; JSON: invalid
    r"{'a': '\x41'}",                   # Python: "A"
    r"{'a': '\101'}",                   # octal
    r"{'a': 'x\/y'}",                   # JSON reads /, Python keeps the backslash
    "{'a': r'raw'}",                    # prefixed string
    "{'a': '''triple'''}",
])
def test_a_repair_never_changes_what_a_string_says(text) -> None:
    assert not ss.check(text, "stop").ok


def test_same_meaning_escapes_are_still_repaired() -> None:
    v = ss.check(r"{'a': 'line\nnext \"q\" é tab\t it\'s'}", "stop")
    assert v.ok and json.loads(v.text) == {"a": 'line\nnext "q" é tab\t it\'s'}


def test_take_over_appends_to_list_system_content() -> None:
    sent, taken = ss.take_over({"messages": [{"role": "system", "content": [{"type": "text", "text": "be"}]}],
                                "response_format": {"type": "json_object"}})
    assert taken and sent["messages"][0]["content"][-1] == {"type": "text", "text": ss.directive({"type": "object"})}


def test_a_schema_json_cannot_hold_is_left_to_strata() -> None:
    payload = {"messages": [], "response_format": {"type": "json_schema", "json_schema": {
        "name": "N", "schema": {"type": "object", "properties": {"x": {"maximum": float("nan")}}}}}}
    assert ss.take_over(payload) == (payload, False)


def test_an_empty_reply_is_retried() -> None:
    v = ss.check("", "stop")
    assert not v.ok and v.retryable


OVERFLOW_400 = {"error": {"message": "the request exceeds the available context size", "type": "invalid_request_error"}}


def _strata_raw(monkeypatch, responses: List[httpx.Response]) -> List[dict]:
    seen: List[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return responses[len(seen) - 1]

    monkeypatch.setattr(upstream_cancel, "cancellable_transport", lambda: httpx.MockTransport(handler))
    return seen


@pytest.mark.parametrize("second", [
    httpx.Response(400, json=OVERFLOW_400),
    httpx.Response(503, json={"error": {"message": "engine starting"}}),
])
def test_a_failed_retry_hands_back_the_original_json_failure(monkeypatch, second) -> None:
    seen = _strata_raw(monkeypatch, [httpx.Response(200, json=_reply('{"a": "x" "b"}')), second])
    result = _run()
    assert len(seen) == 2 and result["text"] == ""
    details = result["raw"]["details"]
    assert details["reason"] == "http_502" and details["message"].startswith("model did not return valid JSON")
    assert result["raw"]["error"] == "upstream_http_5xx"  # never context_overflow: no re-lease of the whole call


def test_no_retry_when_the_first_call_used_the_budget(monkeypatch) -> None:
    from app import llm_backend

    seen = _strata_raw(monkeypatch, [httpx.Response(200, json=_reply('{"a": "x" "b"}'))])
    monkeypatch.setattr(llm_backend, "_resolve_http_read_timeout_sec", lambda body: 0.0)
    result = _run()
    assert len(seen) == 1 and result["raw"]["details"]["reason"] == "http_502"
