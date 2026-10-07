from __future__ import annotations

import json
import logging

import pytest

import app.current_turn_llm_signals as signals_module
from app.current_turn_llm_signals import (
    build_current_turn_llm_prompt,
    parse_current_turn_llm_read,
    parse_current_turn_llm_signals,
    populate_current_turn_llm_signals,
    reset_current_turn_llm_signals_bus_for_tests,
)
from app.settings import settings


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_current_turn_llm_signals_bus_for_tests()
    yield
    reset_current_turn_llm_signals_bus_for_tests()


# --- parse_current_turn_llm_signals -----------------------------------------


def test_parse_empty_array_is_a_clean_empty_result() -> None:
    assert parse_current_turn_llm_signals("[]") == []


def test_parse_real_candidates() -> None:
    raw = '[{"phrase": "Sarah", "type": "person"}, {"phrase": "coffee shop", "type": "place"}]'
    parsed = parse_current_turn_llm_signals(raw)
    assert parsed == [
        {"phrase": "Sarah", "type": "person"},
        {"phrase": "coffee shop", "type": "place"},
    ]


def test_parse_tolerates_surrounding_prose() -> None:
    raw = 'Sure, here is the array:\n[{"phrase": "Sarah", "type": "person"}]\nDone.'
    parsed = parse_current_turn_llm_signals(raw)
    assert parsed == [{"phrase": "Sarah", "type": "person"}]


def test_parse_ignores_stray_bracket_before_the_real_array() -> None:
    # A naive greedy regex (first "[" to last "]" in the whole string) would
    # span from the footnote's "[1]" through the real array's closing "]",
    # producing invalid JSON. Bracket-depth scanning finds the real array.
    raw = 'See item [1] for details.\n[{"phrase": "Sarah", "type": "person"}]'
    parsed = parse_current_turn_llm_signals(raw)
    assert parsed == [{"phrase": "Sarah", "type": "person"}]


def test_parse_ignores_stray_bracket_after_the_real_array() -> None:
    raw = '[{"phrase": "Sarah", "type": "person"}]\nHope that helps! [end]'
    parsed = parse_current_turn_llm_signals(raw)
    assert parsed == [{"phrase": "Sarah", "type": "person"}]


def test_parse_unrecognized_type_falls_back_to_other() -> None:
    # Multi-word phrase, so the structural floor (below) doesn't apply here --
    # this test is purely about the type-fallback behavior.
    raw = '[{"phrase": "Something specific", "type": "not_a_real_type"}]'
    parsed = parse_current_turn_llm_signals(raw)
    assert parsed == [{"phrase": "Something specific", "type": "other"}]


def test_parse_malformed_json_returns_none_not_empty() -> None:
    # "None" (unparsable) must be distinguishable from "[]" (genuinely empty).
    assert parse_current_turn_llm_signals("not json at all, sorry") is None


def test_parse_empty_string_returns_none() -> None:
    assert parse_current_turn_llm_signals("") is None


def test_parse_non_array_json_returns_none() -> None:
    assert parse_current_turn_llm_signals('{"phrase": "Sarah"}') is None


def test_parse_drops_entries_without_a_usable_phrase() -> None:
    raw = (
        '[{"phrase": "", "type": "person"}, {"type": "person"}, {"phrase": "x"}, '
        '{"phrase": "Ok"}, {"phrase": "Sam", "type": "person"}]'
    )
    # "x" is length 1 after strip -- below the parser's own 2-char floor.
    # "Ok" (2 chars, no type -> defaults to "other") is now ALSO dropped: a bare
    # single word typed "other" is exactly the class of garbage
    # (test_single_bare_word_typed_other_or_concept_is_dropped below) this
    # parser is supposed to keep out, and "ok" was never on the detector's
    # STOP_PHRASES list downstream -- it would have sailed all the way through
    # as a real signal before this filter existed. "Sam" (person, single word)
    # survives -- see the structural-floor tests below for the full rationale.
    parsed = parse_current_turn_llm_signals(raw)
    assert parsed == [{"phrase": "Sam", "type": "person"}]


# --- structural floor: single bare word must be person/place ----------------


def test_single_bare_word_typed_other_or_concept_is_dropped() -> None:
    # The real garbage confirmed live 2026-08-21 (a same-turn LLM probe still
    # returning bare single words as "concept"/"other" candidates, one step
    # removed from the deleted regex detector's failure mode): "bus", "Glad",
    # "Compact", "Interesting" -- none are a person or place, none have a space.
    raw = (
        '[{"phrase": "bus", "type": "concept"}, {"phrase": "Glad", "type": "other"}, '
        '{"phrase": "Compact", "type": "activity"}, {"phrase": "Interesting", "type": "belief"}]'
    )
    assert parse_current_turn_llm_signals(raw) == []


def test_single_bare_word_person_or_place_survives() -> None:
    raw = '[{"phrase": "Sarah", "type": "person"}, {"phrase": "Paris", "type": "place"}]'
    assert parse_current_turn_llm_signals(raw) == [
        {"phrase": "Sarah", "type": "person"},
        {"phrase": "Paris", "type": "place"},
    ]


def test_single_bare_word_in_an_uncased_script_survives() -> None:
    # Regression (code review, 3rd pass): the capitalization check must use
    # `not phrase[:1].islower()`, NOT `phrase[:1].isupper()` -- isupper() is
    # False for every uncased script (CJK, Arabic, Hebrew, Thai, ...), which
    # would wrongly drop a real bare name in one of those scripts even though
    # islower() is equally False there (no case distinction exists at all).
    # The pre-this-diff code never had this bug -- it only checked type_hint.
    raw = json.dumps(
        [{"phrase": "東京", "type": "place"}, {"phrase": "محمد", "type": "person"}]
    )
    parsed = parse_current_turn_llm_signals(raw)
    assert parsed == [
        {"phrase": "東京", "type": "place"},
        {"phrase": "محمد", "type": "person"},
    ]


def test_bare_word_name_acceptance_is_logged_at_info_for_auditability(caplog) -> None:
    # Disclosed, not fixed (review, 2026-08-22): a capitalized interjection
    # mistyped person/place would still sail through this floor -- not yet
    # observed live, so no denylist was built on spec. This INFO log (louder
    # than the DEBUG drop-case log) is the instrumentation-first substitute:
    # every bare-word acceptance is greppable so a real occurrence of that
    # failure mode is auditable instead of invisible.
    with caplog.at_level(logging.INFO, logger="orion.cortex.current_turn_llm_signals"):
        parse_current_turn_llm_signals('[{"phrase": "Sarah", "type": "person"}]')
    assert any("current_turn_llm_signal_bare_word_name_accepted" in r.message for r in caplog.records)


def test_single_bare_lowercase_word_mistyped_person_or_place_is_still_dropped() -> None:
    # Confirmed live 2026-08-22, hours after the type-carve-out floor shipped:
    # "bus" got through again because the model classified it "place" that
    # time instead of "concept"/"other" -- the word itself never changed, only
    # the model's own (unreliable) classification did. A genuine name/place is
    # capitalized by ordinary convention ("Sarah", "Paris" above); requiring
    # that on top of the type check is a second, independent signal.
    raw = '[{"phrase": "bus", "type": "place"}, {"phrase": "glad", "type": "person"}]'
    assert parse_current_turn_llm_signals(raw) == []


def test_multi_word_phrase_survives_regardless_of_type() -> None:
    raw = '[{"phrase": "the reactor rollout plan", "type": "plan"}, {"phrase": "context compaction", "type": "concept"}]'
    assert parse_current_turn_llm_signals(raw) == [
        {"phrase": "the reactor rollout plan", "type": "plan"},
        {"phrase": "context compaction", "type": "concept"},
    ]


def test_multi_word_phrase_joined_by_non_breaking_space_still_counts_as_multi_word() -> None:
    # Regression (code review, 2nd pass): a literal `" " not in phrase` check
    # would misclassify this as a single bare token and drop it. `.split()`
    # recognizes U+00A0 (non-breaking space) as whitespace.
    phrase = "context\u00a0compaction"  # NBSP, not a normal space
    raw = json.dumps([{"phrase": phrase, "type": "concept"}])
    assert parse_current_turn_llm_signals(raw) == [{"phrase": phrase, "type": "concept"}]


# --- build_current_turn_llm_prompt ------------------------------------------


def test_prompt_includes_user_text_and_excludes_filler_instruction() -> None:
    prompt = build_current_turn_llm_prompt("Heck yeah!")
    assert "Heck yeah!" in prompt
    assert "interjection" in prompt.lower()
    assert '"wants_direct_answer"' in prompt
    assert '"items"' in prompt


# --- parse_current_turn_llm_read (disclosure-aware object shape) ------------


def test_parse_read_carries_direct_answer_and_natural_question() -> None:
    """corr beab81a3 (2026-09-28): Juniper said they'd be busy with work travel.
    The old probe returned [] -- vague life news had no place in a names/places/
    concrete-plans contract. The read must carry the follow-up a friend would ask."""
    raw = json.dumps(
        {
            "wants_direct_answer": False,
            "items": [{"phrase": "work travel", "type": "plan", "question": "Where are you headed?"}],
        }
    )
    assert parse_current_turn_llm_read(raw) == {
        "wants_direct_answer": False,
        "signals": [{"phrase": "work travel", "type": "plan", "natural_question": "Where are you headed?"}],
    }


def test_parse_read_empty_items_is_a_clean_empty_result() -> None:
    assert parse_current_turn_llm_read('{"wants_direct_answer": true, "items": []}') == {
        "wants_direct_answer": True,
        "signals": [],
    }


def test_parse_read_tolerates_prose_around_the_object() -> None:
    raw = 'Here you go:\n{"wants_direct_answer": false, "items": [{"phrase": "pottery class", "type": "activity", "question": "How was the first session?"}]}\n'
    read = parse_current_turn_llm_read(raw)
    assert read is not None
    assert read["signals"][0]["natural_question"] == "How was the first session?"


def test_parse_read_legacy_array_leaves_direct_answer_unknown() -> None:
    assert parse_current_turn_llm_read('[{"phrase": "Sarah", "type": "person"}]') == {
        "wants_direct_answer": None,
        "signals": [{"phrase": "Sarah", "type": "person"}],
    }


def test_parse_read_non_boolean_direct_answer_is_unknown_not_truthy() -> None:
    read = parse_current_turn_llm_read('{"wants_direct_answer": "yes", "items": []}')
    assert read == {"wants_direct_answer": None, "signals": []}


def test_parse_read_bare_word_floor_still_applies_inside_the_object() -> None:
    raw = json.dumps(
        {
            "wants_direct_answer": False,
            "items": [
                {"phrase": "lol", "type": "other", "question": "What's funny?"},
                {"phrase": "my sister's visit", "type": "person", "question": "How long is she staying?"},
            ],
        }
    )
    read = parse_current_turn_llm_read(raw)
    assert [s["phrase"] for s in read["signals"]] == ["my sister's visit"]


def test_parse_read_blank_question_is_dropped_and_long_question_is_bounded() -> None:
    raw = json.dumps(
        {
            "wants_direct_answer": False,
            "items": [
                {"phrase": "repainting the living room", "type": "plan", "question": "   "},
                {"phrase": "the dentist visit", "type": "activity", "question": "x" * 500},
            ],
        }
    )
    read = parse_current_turn_llm_read(raw)
    assert "natural_question" not in read["signals"][0]
    assert len(read["signals"][1]["natural_question"]) <= 160


def test_parse_read_object_without_items_key_falls_back_to_array_scan() -> None:
    """A bare array response's first `{...}` span is an item, not the read object;
    it must not be mistaken for one."""
    raw = '[{"phrase": "Sarah", "type": "person"}]'
    assert parse_current_turn_llm_read(raw)["wants_direct_answer"] is None


def test_parse_read_garbage_returns_none() -> None:
    assert parse_current_turn_llm_read("I refuse to output JSON today.") is None


# --- populate_current_turn_llm_signals --------------------------------------


@pytest.mark.asyncio
async def test_empty_user_message_skips_call_entirely(monkeypatch) -> None:
    called = []

    async def _fake_llm_call(bus, *, prompt):
        called.append(prompt)
        return "[]"

    monkeypatch.setattr(signals_module, "_llm_call", _fake_llm_call)
    signals_module.bind_current_turn_llm_signals_bus(object())

    ctx = {"user_message": ""}
    await populate_current_turn_llm_signals(ctx)
    assert ctx["current_turn_llm_signals"] == []
    assert called == []


@pytest.mark.asyncio
async def test_unbound_bus_fails_open_with_distinguishable_warning(caplog) -> None:
    ctx = {"user_message": "Heck yeah!"}
    with caplog.at_level(logging.WARNING, logger="orion.cortex.current_turn_llm_signals"):
        await populate_current_turn_llm_signals(ctx)
    assert ctx["current_turn_llm_signals"] == []
    assert any("bus_unbound" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_filler_message_with_empty_llm_result_yields_no_candidates(monkeypatch) -> None:
    async def _fake_llm_call(bus, *, prompt):
        return "[]"

    monkeypatch.setattr(signals_module, "_llm_call", _fake_llm_call)
    signals_module.bind_current_turn_llm_signals_bus(object())

    for message in ("Heck yeah!", "yeah", "yep", "wow"):
        ctx = {"user_message": message}
        await populate_current_turn_llm_signals(ctx)
        assert ctx["current_turn_llm_signals"] == []


@pytest.mark.asyncio
async def test_real_candidates_flow_through_to_ctx(monkeypatch) -> None:
    async def _fake_llm_call(bus, *, prompt):
        return '[{"phrase": "Sarah", "type": "person"}, {"phrase": "coffee shop", "type": "place"}]'

    monkeypatch.setattr(signals_module, "_llm_call", _fake_llm_call)
    signals_module.bind_current_turn_llm_signals_bus(object())

    ctx = {"user_message": "I'm meeting Sarah at the coffee shop tomorrow"}
    await populate_current_turn_llm_signals(ctx)
    assert ctx["current_turn_llm_signals"] == [
        {"phrase": "Sarah", "type": "person"},
        {"phrase": "coffee shop", "type": "place"},
    ]


@pytest.mark.asyncio
async def test_rpc_failure_fails_open_with_distinguishable_warning(monkeypatch, caplog) -> None:
    async def _boom(bus, *, prompt):
        raise TimeoutError("rpc timed out")

    monkeypatch.setattr(signals_module, "_llm_call", _boom)
    signals_module.bind_current_turn_llm_signals_bus(object())

    ctx = {"user_message": "I'm meeting Sarah tomorrow"}
    with caplog.at_level(logging.WARNING, logger="orion.cortex.current_turn_llm_signals"):
        await populate_current_turn_llm_signals(ctx)
    assert ctx["current_turn_llm_signals"] == []
    assert any("rpc_failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_malformed_output_fails_open_with_distinguishable_warning(monkeypatch, caplog) -> None:
    async def _fake_llm_call(bus, *, prompt):
        return "I refuse to output JSON today."

    monkeypatch.setattr(signals_module, "_llm_call", _fake_llm_call)
    signals_module.bind_current_turn_llm_signals_bus(object())

    ctx = {"user_message": "I'm meeting Sarah tomorrow"}
    with caplog.at_level(logging.WARNING, logger="orion.cortex.current_turn_llm_signals"):
        await populate_current_turn_llm_signals(ctx)
    assert ctx["current_turn_llm_signals"] == []
    assert any("malformed_output" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_three_failure_modes_use_distinct_log_messages(monkeypatch, caplog) -> None:
    """RPC failure, malformed output, and unbound bus must never be
    conflated in logs with each other or with a genuine empty result."""
    ctx = {"user_message": "hello there"}

    # unbound bus
    with caplog.at_level(logging.WARNING, logger="orion.cortex.current_turn_llm_signals"):
        await populate_current_turn_llm_signals(dict(ctx))
    unbound_msgs = [r.message for r in caplog.records]
    caplog.clear()

    signals_module.bind_current_turn_llm_signals_bus(object())

    async def _boom(bus, *, prompt):
        raise RuntimeError("boom")

    monkeypatch.setattr(signals_module, "_llm_call", _boom)
    with caplog.at_level(logging.WARNING, logger="orion.cortex.current_turn_llm_signals"):
        await populate_current_turn_llm_signals(dict(ctx))
    rpc_failed_msgs = [r.message for r in caplog.records]
    caplog.clear()

    async def _garbage(bus, *, prompt):
        return "not json"

    monkeypatch.setattr(signals_module, "_llm_call", _garbage)
    with caplog.at_level(logging.WARNING, logger="orion.cortex.current_turn_llm_signals"):
        await populate_current_turn_llm_signals(dict(ctx))
    malformed_msgs = [r.message for r in caplog.records]

    assert any("bus_unbound" in m for m in unbound_msgs)
    assert any("rpc_failed" in m for m in rpc_failed_msgs)
    assert any("malformed_output" in m for m in malformed_msgs)
    assert set(unbound_msgs) != set(rpc_failed_msgs) != set(malformed_msgs)


def test_non_string_question_is_not_a_natural_question() -> None:
    raw = json.dumps(
        {
            "wants_direct_answer": False,
            "items": [
                {"phrase": "work travel", "type": "plan", "question": True},
                {"phrase": "pottery class", "type": "activity", "question": ["What kind?"]},
            ],
        }
    )
    read = parse_current_turn_llm_read(raw)
    assert read is not None
    assert all("natural_question" not in s for s in read["signals"])


@pytest.mark.asyncio
async def test_populate_records_a_successful_turn_read(monkeypatch) -> None:
    async def _fake_llm_call(bus, *, prompt):
        return json.dumps(
            {
                "wants_direct_answer": False,
                "items": [{"phrase": "work travel", "type": "plan", "question": "Where are you headed?"}],
            }
        )

    monkeypatch.setattr(signals_module, "_llm_call", _fake_llm_call)
    signals_module.bind_current_turn_llm_signals_bus(object())

    ctx = {"user_message": "I'll be pretty busy the next few days with work travel."}
    await populate_current_turn_llm_signals(ctx)
    assert ctx["current_turn_llm_read"] == {"ok": True, "wants_direct_answer": False}
    assert ctx["current_turn_llm_signals"] == [
        {"phrase": "work travel", "type": "plan", "natural_question": "Where are you headed?"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["rpc", "malformed", "unbound"])
async def test_populate_marks_turn_read_unavailable_on_every_failure(monkeypatch, failure) -> None:
    async def _boom(bus, *, prompt):
        raise TimeoutError("rpc timed out")

    async def _garbage(bus, *, prompt):
        return "not json"

    if failure != "unbound":
        monkeypatch.setattr(signals_module, "_llm_call", _boom if failure == "rpc" else _garbage)
        signals_module.bind_current_turn_llm_signals_bus(object())

    ctx = {"user_message": "I'm meeting Sarah tomorrow"}
    await populate_current_turn_llm_signals(ctx)
    assert ctx["current_turn_llm_read"]["ok"] is False
    assert ctx["current_turn_llm_read"]["wants_direct_answer"] is None


def test_probe_request_pins_temperature_and_token_budget():
    """Unpinned sampling made the live probe a coin flip: the same prompt and
    message caught the trip 1/10 at default temperature on the quick lane."""
    import asyncio

    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.core.bus.codec import OrionCodec

    seen: dict = {}
    codec = OrionCodec()

    class _Bus:
        def __init__(self) -> None:
            self.codec = codec

        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec, health_label=None):
            seen["options"] = env.payload["options"]
            reply = BaseEnvelope(kind="llm.chat.result", source=ServiceRef(name="llm-gateway"), payload={"content": "[]"})
            return {"data": codec.encode(reply)}

    asyncio.run(signals_module._llm_call(_Bus(), prompt="hi"))
    assert seen["options"]["temperature"] == settings.current_turn_signal_probe_temperature
    assert seen["options"]["max_tokens"] == settings.current_turn_signal_probe_max_tokens
    assert settings.current_turn_signal_probe_max_tokens >= 200


def test_probe_rpc_uses_its_own_rpc_health_hop_label():
    """The probe's deliberate 3 s deadline must not be counted as LLMGatewayService
    delivery failure (orion/substrate/rpc_delivery.py excludes this label)."""
    import asyncio

    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.core.bus.codec import OrionCodec
    from orion.substrate.rpc_delivery import DEFAULT_EXCLUDE_LABELS

    seen: dict = {}
    codec = OrionCodec()

    class _Bus:
        def __init__(self) -> None:
            self.codec = codec

        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec, health_label=None):
            seen.update(channel=channel, health_label=health_label, timeout_sec=timeout_sec)
            reply = BaseEnvelope(kind="llm.chat.result", source=ServiceRef(name="llm-gateway"), payload={"content": "[]"})
            return {"data": codec.encode(reply)}

    assert asyncio.run(signals_module._llm_call(_Bus(), prompt="hi")) == "[]"
    assert seen["health_label"] == signals_module.PROBE_HEALTH_LABEL == "current_turn_probe"
    assert signals_module.PROBE_HEALTH_LABEL in DEFAULT_EXCLUDE_LABELS


# --- human-turn gate --------------------------------------------------------

from app.current_turn_llm_signals import (  # noqa: E402
    SKIPPED_NOT_HUMAN_TURN,
    human_chat_turn_reason,
    mark_current_turn_llm_skipped,
)
from orion.substrate.attention.policy import direct_answer_cause  # noqa: E402


@pytest.mark.parametrize(
    "ctx,expected",
    [
        ({"verb": "stance_react", "stance_inputs": {"utterance_origin": "juniper"}}, (True, "utterance_origin_juniper")),
        ({"verb": "chat_general"}, (True, "chat_entry_verb")),
        ({"verb": "chat_quick", "options": {"chat_quick_full_stance": True}}, (True, "chat_entry_verb")),
        ({"verb": "stance_react", "stance_inputs": {"utterance_origin": "orion"}}, (False, "utterance_origin_orion")),
        ({"verb": "stance_react", "stance_inputs": {}}, (False, "unified_turn_without_human_origin")),
        ({"verb": "journal.compose"}, (False, "non_chat_verb:journal.compose")),
        ({"verb": "skills.imagination.render_scene.v1"}, (False, "non_chat_verb:skills.imagination.render_scene.v1")),
        ({}, (False, "non_chat_verb:none")),
        ({"verb": "chat_general", "options": {"policy_dispatch_only": True}}, (False, "policy_dispatch")),
        ({"verb": "chat_general", "policy_dispatch_only": True}, (False, "policy_dispatch")),
    ],
)
def test_human_chat_turn_reason(ctx, expected) -> None:
    assert human_chat_turn_reason(ctx) == expected


def test_skipped_state_is_distinct_and_fails_closed() -> None:
    ctx = {"user_message": "Compose today's journal entry."}
    mark_current_turn_llm_skipped(ctx, "non_chat_verb:journal.compose")
    assert ctx["current_turn_llm_signals"] == []
    read = ctx["current_turn_llm_read"]
    assert read == {
        "ok": False,
        "wants_direct_answer": None,
        "skipped": SKIPPED_NOT_HUMAN_TURN,
        "skip_reason": "non_chat_verb:journal.compose",
    }
    # Not "found nothing" (that would be ok=True) and not "wants a direct answer"
    # ("judged"): the policy treats it like any missing read.
    assert direct_answer_cause(ctx, ctx["user_message"]) == "unavailable"
