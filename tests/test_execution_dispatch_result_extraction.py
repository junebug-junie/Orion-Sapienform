from __future__ import annotations

from orion.execution_dispatch.result_extraction import (
    RESULT_KIND_EMPTY,
    RESULT_KIND_OBSERVATION,
    RESULT_KIND_STRUCTURED,
    extract_final_text,
    parse_structured_observation,
)


def test_extract_final_text_from_result_final_text() -> None:
    payload = {"result": {"final_text": "  hello world  "}}
    assert extract_final_text(payload) == "hello world"


def test_extract_final_text_from_step_result_fallback() -> None:
    payload = {
        "result": {
            "final_text": None,
            "steps": [
                {"result": {"llm": {"text": "step output here"}}},
            ],
        }
    }
    assert extract_final_text(payload) == "step output here"


def test_extract_final_text_missing_returns_empty() -> None:
    assert extract_final_text({}) == ""
    assert extract_final_text({"result": {}}) == ""
    assert extract_final_text({"result": None}) == ""


# ---------------------------------------------------------------------------
# Family 1: observation verbs (substrate.inspect / .summarize / .observe).
# Behaviour here is unchanged by the 2026-08-13 structured-result patch; these
# tests exist to prove that, since the patch touches the same function.
# ---------------------------------------------------------------------------


def test_parse_structured_observation_happy_path() -> None:
    data = parse_structured_observation(
        '{"observation": "steady state", "salient_facts": ["a", "b"], "confidence": 0.75}'
    )
    assert data["observation"] == "steady state"
    assert data["salient_facts"] == ["a", "b"]
    assert data["confidence"] == 0.75
    assert data["result_kind"] == RESULT_KIND_OBSERVATION
    assert data["structured_result"] is None


def test_parse_structured_observation_empty_text() -> None:
    data = parse_structured_observation("")
    assert data["observation"] == ""
    assert data["salient_facts"] == []
    assert data["confidence"] == 0.0
    assert data["result_kind"] == RESULT_KIND_EMPTY


def test_parse_structured_observation_malformed_json_degrades_empty() -> None:
    data = parse_structured_observation("not json at all {{{")
    assert data["observation"] == ""
    assert data["result_kind"] == RESULT_KIND_EMPTY


def test_parse_structured_observation_non_dict_json_degrades_empty() -> None:
    data = parse_structured_observation("[1, 2, 3]")
    assert data["observation"] == ""
    assert data["result_kind"] == RESULT_KIND_EMPTY


def test_parse_structured_observation_empty_object_degrades_empty() -> None:
    """`{}` is a JSON object but carries nothing. Still empty, deliberately.

    Without this, the family-2 branch below would classify an empty object as a
    real structured result and report success for a genuinely empty payload --
    reintroducing exactly the empty-shell failure this patch exists to remove,
    in the opposite direction.
    """
    data = parse_structured_observation("{}")
    assert data["result_kind"] == RESULT_KIND_EMPTY
    assert data["structured_result"] is None


def test_parse_structured_observation_present_but_blank_observation_is_empty() -> None:
    """An observation verb that returned `""` genuinely returned nothing.

    The `observation` key is present, so this must NOT be misread as a
    structured skill result. This is the boundary between the two families.
    """
    data = parse_structured_observation('{"observation": "   ", "confidence": 0.9}')
    assert data["observation"] == ""
    assert data["result_kind"] == RESULT_KIND_EMPTY


def test_parse_structured_observation_wrong_types_coerced_safely() -> None:
    data = parse_structured_observation(
        '{"observation": 123, "salient_facts": "not-a-list", "confidence": "n/a"}'
    )
    assert data["observation"] == ""
    assert data["salient_facts"] == []
    assert data["confidence"] == 0.0
    assert data["result_kind"] == RESULT_KIND_EMPTY


def test_parse_structured_observation_bool_confidence_is_not_a_number() -> None:
    """`True` is an `int` subclass in Python; it must not become confidence 1.0."""
    data = parse_structured_observation('{"observation": "x", "confidence": true}')
    assert data["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Family 2: skill verbs (skills.runtime.*). This is the behaviour change.
#
# BEHAVIOUR CHANGED, intentionally: before this patch, every payload in this
# section returned `{"observation": "", ...}` and the caller stored it as
# `status="empty"` with `raw_len=0`. Confirmed live on 2026-08-12: all four
# `skills.runtime.builder_prune.v1` dispatches recorded exactly that, while
# cortex-exec logged `raw_len=739 final_len=739` for the same correlation ids.
# The old `test_parse_structured_observation_missing_fields_defaults` asserted
# that behaviour and therefore locked the bug in; it is replaced below.
# ---------------------------------------------------------------------------


def test_skill_result_is_a_real_result_not_empty() -> None:
    """A real builder_prune decline payload, shaped as the verb actually emits it.

    Key assertion: this is `structured`, not `empty`. A skill that measured the
    host and honestly declined to act produced a real result.
    """
    payload = (
        '{"acted": false, "decision": "declined_no_pressure", '
        '"reason": "used_pct 70.2 < 75.0", "node_name": "athena", '
        '"disk_before": {"used_pct": 70.2, "used_bytes": 1000}}'
    )
    data = parse_structured_observation(payload)
    assert data["result_kind"] == RESULT_KIND_STRUCTURED
    assert data["structured_result"]["decision"] == "declined_no_pressure"
    assert data["structured_result"]["disk_before"]["used_pct"] == 70.2


def test_skill_result_observation_is_synthesised_from_real_fields_only() -> None:
    """Hand-derived: fields present are `decision` and `reason`, in that order.

    STRUCTURED_SUMMARY_FIELDS order is (decision, status, summary, reason), so
    the expected join is "declined_no_pressure: used_pct 70.2 < 75.0" -- no
    invented prose, every character traceable to a field in the payload.
    """
    payload = '{"acted": false, "decision": "declined_no_pressure", "reason": "used_pct 70.2 < 75.0"}'
    data = parse_structured_observation(payload)
    assert data["observation"] == "declined_no_pressure: used_pct 70.2 < 75.0"


def test_skill_result_preserves_a_real_outcome_number() -> None:
    """The acceptance check the whole arc exists for: bytes_reclaimed survives.

    Before this patch this number was discarded at the parse boundary and could
    not be queried from `substrate_dispatch_results` at all.
    """
    payload = '{"acted": true, "decision": "pruned", "bytes_reclaimed": 150999998464}'
    data = parse_structured_observation(payload)
    assert data["structured_result"]["bytes_reclaimed"] == 150999998464
    assert data["result_kind"] == RESULT_KIND_STRUCTURED


def test_skill_result_with_no_summary_fields_is_still_a_real_result() -> None:
    """No `decision`/`status`/`summary`/`reason` -> empty observation, real result.

    This is the case that forces status to be driven by `result_kind` rather
    than by `len(observation)`: the payload is real, the prose is absent, and
    those two facts must stay separable.
    """
    data = parse_structured_observation('{"unrelated": true, "count": 3}')
    assert data["observation"] == ""
    assert data["result_kind"] == RESULT_KIND_STRUCTURED
    assert data["structured_result"] == {"unrelated": True, "count": 3}


def test_skill_result_confidence_is_not_fabricated() -> None:
    """A skill verb reports no epistemic confidence; inventing one is banned."""
    data = parse_structured_observation('{"decision": "pruned", "acted": true}')
    assert data["confidence"] == 0.0
    assert data["salient_facts"] == []


def test_skill_result_observation_is_bounded() -> None:
    """A verbose skill payload must not blow up the stored observation string.

    The record is `structured_result`, stored in full; `observation` is a
    bounded convenience rendering.
    """
    long_reason = "x" * 5000
    data = parse_structured_observation(f'{{"decision": "pruned", "reason": "{long_reason}"}}')
    assert len(data["observation"]) == 1000
    # The full text is not lost -- it survives in the structured record.
    assert len(data["structured_result"]["reason"]) == 5000


def test_skill_result_ignores_non_string_summary_fields() -> None:
    """A numeric `status` must not be stringified into the observation.

    Only real strings are joined; anything else stays in `structured_result`
    where its type is preserved.
    """
    data = parse_structured_observation('{"status": 200, "decision": "pruned"}')
    assert data["observation"] == "pruned"
    assert data["structured_result"]["status"] == 200
