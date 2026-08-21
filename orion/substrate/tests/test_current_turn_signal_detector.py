from __future__ import annotations

from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.detectors.current_turn import CurrentTurnSignalDetector


def _detect(ctx: dict) -> list[AttentionSignalV1]:
    return CurrentTurnSignalDetector().detect(ctx, {}, [])


def test_absent_ctx_key_yields_no_signals() -> None:
    # No ctx["current_turn_llm_signals"] key at all -- e.g. the frame is
    # disabled, so populate_current_turn_llm_signals() was never called.
    assert _detect({"user_message": "Heck yeah!"}) == []


def test_empty_candidate_list_yields_no_signals() -> None:
    # LLM call ran and genuinely found nothing worth tracking -- the fixed
    # bug case: "Heck yeah!" should not surface "Heck" as a candidate the
    # way the deleted LegacyRegexSignalDetector's _PROPER_RE regex did.
    assert _detect({"user_message": "Heck yeah!", "current_turn_llm_signals": []}) == []


def test_real_candidates_produce_correctly_shaped_signals() -> None:
    ctx = {
        "user_message": "I'm meeting Sarah at the coffee shop tomorrow",
        "current_turn_llm_signals": [
            {"phrase": "Sarah", "type": "person"},
            {"phrase": "coffee shop", "type": "place"},
        ],
    }
    signals = _detect(ctx)
    assert len(signals) == 2

    by_text = {s.target_text: s for s in signals}
    assert set(by_text) == {"Sarah", "coffee shop"}

    sarah = by_text["Sarah"]
    assert sarah.target_type_hint == "person"
    assert sarah.signal_kind == "llm_novelty_v1"
    assert sarah.source == "current_turn_v1"
    assert sarah.evidence_refs == ["ctx.user_message"]
    assert sarah.provenance["detector"] == "current_turn_v1"
    assert 0.0 <= sarah.salience <= 1.0
    assert 0.0 <= sarah.confidence <= 1.0

    coffee = by_text["coffee shop"]
    assert coffee.target_type_hint == "place"


def test_deduplicates_case_insensitively() -> None:
    ctx = {
        "user_message": "Sarah, sarah, SARAH",
        "current_turn_llm_signals": [
            {"phrase": "Sarah", "type": "person"},
            {"phrase": "sarah", "type": "person"},
        ],
    }
    signals = _detect(ctx)
    assert len(signals) == 1


def test_stop_phrase_still_filtered_as_defense_in_depth() -> None:
    ctx = {
        "user_message": "juniper is here",
        "current_turn_llm_signals": [{"phrase": "juniper", "type": "person"}],
    }
    assert _detect(ctx) == []


def test_malformed_candidate_entries_are_skipped_not_raised() -> None:
    ctx = {
        "user_message": "whatever",
        "current_turn_llm_signals": [
            "not-a-dict",
            {"phrase": "", "type": "person"},
            {"phrase": "x", "type": "person"},  # too short after strip, len < 2
            {"phrase": "Ok", "type": "person"},
        ],
    }
    signals = _detect(ctx)
    assert len(signals) == 1
    assert signals[0].target_text == "Ok"


def test_missing_type_key_falls_back_to_other() -> None:
    # Type-hint validation against the allowed vocabulary happens upstream,
    # in current_turn_llm_signals.py::parse_current_turn_llm_signals(),
    # before candidates ever land in ctx -- this detector only needs to
    # handle the "key absent entirely" case defensively.
    ctx = {
        "user_message": "whatever",
        "current_turn_llm_signals": [{"phrase": "Something Specific"}],
    }
    signals = _detect(ctx)
    assert signals[0].target_type_hint == "other"
