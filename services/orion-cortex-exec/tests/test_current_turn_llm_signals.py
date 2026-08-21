from __future__ import annotations

import logging

import pytest

import app.current_turn_llm_signals as signals_module
from app.current_turn_llm_signals import (
    build_current_turn_llm_prompt,
    parse_current_turn_llm_signals,
    populate_current_turn_llm_signals,
    reset_current_turn_llm_signals_bus_for_tests,
)


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


def test_multi_word_phrase_survives_regardless_of_type() -> None:
    raw = '[{"phrase": "the reactor rollout plan", "type": "plan"}, {"phrase": "context compaction", "type": "concept"}]'
    assert parse_current_turn_llm_signals(raw) == [
        {"phrase": "the reactor rollout plan", "type": "plan"},
        {"phrase": "context compaction", "type": "concept"},
    ]


# --- build_current_turn_llm_prompt ------------------------------------------


def test_prompt_includes_user_text_and_excludes_filler_instruction() -> None:
    prompt = build_current_turn_llm_prompt("Heck yeah!")
    assert "Heck yeah!" in prompt
    assert "interjection" in prompt.lower()
    assert "JSON array" in prompt


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
